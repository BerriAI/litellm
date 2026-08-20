"""
CLI Token Utilities

SDK-level utilities for reading the credential minted by `lite login`.

Non-secret metadata lives in ~/.litellm/token.json. The secret material (the
bearer key, plus a JWT when one is issued) lives in the OS keychain when the
machine has one, and in that same 0600 file otherwise. This module hides the
split from callers, and migrates a legacy plaintext file into the keychain the
first time it reads one.

This module has no dependencies on proxy code and can be safely imported at the SDK level.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, TypeAlias

from pydantic import BaseModel, ConfigDict, ValidationError

from litellm.litellm_core_utils.cli_keyring import (
    SYSTEM_KEYRING,
    KeyringDisabled,
    KeyringDiscardsWrites,
    KeyringNotInstalled,
    KeyringUnreachable,
    SecretErase,
    SecretErased,
    SecretFound,
    SecretMissing,
    SecretStored,
    SecretStranded,
    SecretVault,
    SecretWrite,
)
from litellm.litellm_core_utils.private_json import (
    commit_staged_json,
    discard_staged_json,
    ensure_private_dir,
    stage_private_json,
    write_private_json,
)


@dataclass(frozen=True, slots=True)
class CredentialNotSaved:
    """The credential was minted but no store would keep it, so this machine has none."""

    detail: str


SecretSave: TypeAlias = SecretWrite | CredentialNotSaved


class CliTokenRecord(BaseModel):
    """A stored CLI credential.

    `key is None` means the metadata was found but the secret could not be
    produced: the keychain holds nothing for us, or we could not reach it.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    base_url: str = ""
    key: str | None = None
    user_id: str = ""
    user_email: str = ""
    user_role: str = ""
    auth_header_name: str = "Authorization"
    jwt_token: str = ""
    timestamp: float = 0.0


class CliTokenSecret(BaseModel):
    """The secret material as stored in the OS keychain.

    `base_url` is duplicated from the metadata file purely as a pairing tag: a
    secret minted for one server is never handed to another, even if the
    metadata file is edited underneath us.
    """

    model_config = ConfigDict(frozen=True)

    base_url: str
    key: str
    jwt_token: str = ""


def get_cli_token_file_path() -> str:
    """Get the path to the CLI token file"""
    home_dir: Final = Path.home()
    config_dir: Final = home_dir / ".litellm"
    return str(config_dir / "token.json")


def load_cli_token(*, vault: SecretVault = SYSTEM_KEYRING) -> CliTokenRecord | None:
    """Load the stored CLI credential, or None when this machine has none"""
    record: Final = _read_token_file()
    if record is None:
        return None
    return _resolve_secret(record, vault)


def save_cli_token(record: CliTokenRecord, *, vault: SecretVault = SYSTEM_KEYRING) -> SecretSave:
    """Store a freshly minted credential. Reports where its secret material ended up, and why.

    The token file is what makes a keychain-backed credential findable again, so a file that will
    not be written takes the keychain copy down with it rather than leaving a live credential
    stored under a machine that has no record of it.
    """
    outcome: Final = (
        SecretStored()
        if record.key is None
        else vault.write(_encode_secret(record.base_url, record.key, record.jwt_token))
    )
    try:
        _write_token_file(_without_secret(record) if isinstance(outcome, SecretStored) else record)
    except OSError as error:
        if record.key is not None and isinstance(outcome, SecretStored):
            vault.erase()
        return CredentialNotSaved(str(error))
    return outcome


def clear_cli_token(*, vault: SecretVault = SYSTEM_KEYRING) -> SecretErase:
    """Remove the credential from both stores. Reports whether the keychain is now free of it"""
    outcome: Final = vault.erase()
    settled: Final = _nothing_left_behind(outcome)
    Path(get_cli_token_file_path()).unlink(missing_ok=True)
    return SecretErased() if settled else outcome


def _nothing_left_behind(outcome: SecretErase) -> bool:
    """Whether the keychain can be trusted to hold no credential of ours once the file is gone.

    A keychain that exists but is out of reach right now is never trusted, whatever the token file
    looks like: the login that stored a secret there and the logout that cannot remove it are
    separate runs, free to differ in whether the keychain was usable at the time.
    """
    match outcome:
        case SecretErased():
            return True
        case SecretStranded() | KeyringDisabled() | KeyringUnreachable() | KeyringDiscardsWrites():
            return False
        case KeyringNotInstalled():
            return _file_holds_its_own_secret()


def _file_holds_its_own_secret() -> bool:
    """Whether the stored login keeps its secret in the token file, ruling out a keychain entry.

    Sound only against a missing `keyring` package, the one way to lose the keychain that had to
    hold at storage time too, since nothing here can reach a keychain without it. A file whose
    secret half is absent went to a keychain by definition, and so rules nothing out.
    """
    record: Final = _read_token_file()
    return record is not None and record.key is not None


def get_litellm_gateway_api_key(
    expected_base_url: str | None = None,
    *,
    vault: SecretVault = SYSTEM_KEYRING,
) -> str | None:
    """
    Get the stored CLI API key for use with LiteLLM SDK.

    This function reads the credential created by `lite login`
    and returns the API key for use in Python scripts.

    Args:
        expected_base_url: When provided, the key is only returned if it was
            originally issued for this URL. Pass the target server URL to
            prevent credential leakage when the client is pointed at a
            different (possibly malicious) server.
        vault: Where the secret material is stored. Defaults to the OS keychain.

    Returns:
        str: The API key if found (and origin matches), None otherwise

    Example:
        >>> import litellm
        >>> api_key = litellm.get_litellm_gateway_api_key()
        >>> if api_key:
        >>>     response = litellm.completion(
        >>>         model="gpt-3.5-turbo",
        >>>         messages=[{"role": "user", "content": "Hello"}],
        >>>         api_key=api_key,
        >>>         base_url="https://your-proxy.com/v1"
        >>>     )
    """
    record: Final = _read_token_file()
    if record is None:
        return None
    if expected_base_url is not None and record.base_url != expected_base_url.rstrip("/"):
        return None
    resolved: Final = _resolve_secret(record, vault)
    return None if resolved is None else resolved.key


def is_cli_token_fresh(token_data: CliTokenRecord, buffer_hours: float = 0.1) -> bool:
    """Check whether a cached CLI token is still within its expiration window.
    Used by `lite auth print-token` to fail fast, without a network round trip,
    once the cached token is past `LITELLM_CLI_JWT_EXPIRATION_HOURS`."""
    from litellm.constants import CLI_JWT_EXPIRATION_HOURS

    age_hours: Final = (time.time() - token_data.timestamp) / 3600
    return age_hours < (CLI_JWT_EXPIRATION_HOURS - buffer_hours)


def _read_token_file() -> CliTokenRecord | None:
    try:
        raw: Final = Path(get_cli_token_file_path()).read_text()
    except (OSError, ValueError):
        return None
    try:
        return CliTokenRecord.model_validate_json(raw)
    except ValidationError:
        return None


def _resolve_secret(record: CliTokenRecord, vault: SecretVault) -> CliTokenRecord | None:
    match vault.read():
        case SecretFound(blob=blob):
            return _apply_vault_secret(record, blob, vault)
        case SecretMissing():
            return _migrate_file_secret(record, vault)
        case KeyringNotInstalled() | KeyringDisabled() | KeyringUnreachable() | KeyringDiscardsWrites():
            return record


def _apply_vault_secret(record: CliTokenRecord, blob: str, vault: SecretVault) -> CliTokenRecord | None:
    """Resolve the credential when both stores hold one.

    A secret still on disk is the fresher of the two, because it is only left there when the
    keychain write that should have removed it failed, so it outranks the vault entry.
    """
    if record.key is not None:
        return _migrate_file_secret(record, vault)
    try:
        secret: Final = CliTokenSecret.model_validate_json(blob)
    except ValidationError:
        return _migrate_file_secret(record, vault)
    if secret.base_url != record.base_url:
        return _migrate_file_secret(record, vault)
    _scrub_file_secret(record)
    return record.model_copy(update=MappingProxyType({"key": secret.key, "jwt_token": secret.jwt_token}))


def _migrate_file_secret(record: CliTokenRecord, vault: SecretVault) -> CliTokenRecord | None:
    """Move a file-held secret into the vault, but only once the file's copy can be taken away.

    The scrubbed file is staged first so a directory that will not accept it stops the migration
    before the keychain is handed anything. Copying the credential into a second store and only
    then discovering the first one cannot be cleaned would widen exposure instead of narrowing it,
    which is the opposite of what moving it into the keychain is for.
    """
    if record.key is None:
        return None
    staged: Final = _stage_scrubbed_file(record)
    if staged is None:
        return record
    if not isinstance(vault.write(_encode_secret(record.base_url, record.key, record.jwt_token)), SecretStored):
        discard_staged_json(staged)
        return record
    if not _commit_scrubbed_file(staged):
        vault.erase()
    return record


def _scrub_file_secret(record: CliTokenRecord) -> bool:
    """Leave no secret material in the token file once the vault holds it"""
    if record.key is None and not record.jwt_token:
        return True
    staged: Final = _stage_scrubbed_file(record)
    return staged is not None and _commit_scrubbed_file(staged)


def _stage_scrubbed_file(record: CliTokenRecord) -> str | None:
    path: Final = Path(get_cli_token_file_path())
    try:
        ensure_private_dir(path.parent)
        return stage_private_json(str(path), _without_secret(record).model_dump(exclude_none=True))
    except OSError:
        return None


def _commit_scrubbed_file(staged: str) -> bool:
    try:
        commit_staged_json(staged, get_cli_token_file_path())
    except OSError:
        return False
    return True


def _without_secret(record: CliTokenRecord) -> CliTokenRecord:
    return record.model_copy(update=MappingProxyType({"key": None, "jwt_token": ""}))


def _encode_secret(base_url: str, key: str, jwt_token: str) -> str:
    return CliTokenSecret(base_url=base_url, key=key, jwt_token=jwt_token).model_dump_json()


def _write_token_file(record: CliTokenRecord) -> None:
    path: Final = Path(get_cli_token_file_path())
    ensure_private_dir(path.parent)
    write_private_json(str(path), record.model_dump(exclude_none=True))
