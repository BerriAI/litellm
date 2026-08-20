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
    overwrite_private_json,
    stage_private_json,
    write_private_json,
)


@dataclass(frozen=True, slots=True)
class CredentialNotSaved:
    """The credential was minted but no store would keep it, so this machine has none.

    Nothing was touched on the way to this, so a login that already worked still does.
    """

    detail: str


@dataclass(frozen=True, slots=True)
class CredentialNotRecorded:
    """The keychain took the credential, but the file that names it could not be replaced.

    The keychain holds one entry, so the secret that was there is already gone and no rollback
    brings it back. Removing the new one as well would only turn a login this machine may still
    be able to use into no login at all, so it stays, and the user is told what is where.
    """


@dataclass(frozen=True, slots=True)
class CredentialNotCleared:
    """The token file still holds the secret, because it could not be removed or rewritten.

    Logging out of the keychain is only half of it. A `~/.litellm` that refuses both the scrubbed
    rewrite and the removal leaves the credential readable on disk, which is the one thing a logout
    is for, so it is reported instead of being counted as a clean sweep.
    """

    detail: str


SecretSave: TypeAlias = SecretWrite | CredentialNotSaved | CredentialNotRecorded

SecretClear: TypeAlias = SecretErase | CredentialNotCleared


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

    The token file is what makes a keychain-backed credential findable again, and it is also the
    half that a read-only or full directory refuses, so it is staged before the keychain is handed
    anything. A save that cannot land then leaves both stores exactly as it found them, which
    matters most when the login it failed to replace is still perfectly good.

    Staging can still succeed and the replacement fail afterwards. That is the one case where the
    keychain has already taken the new secret, and it reports itself as such rather than claiming
    the previous login survived.
    """
    staged: Final = _stage_token_file(_without_secret(record))
    if isinstance(staged, CredentialNotSaved):
        return staged
    outcome: Final = (
        SecretStored()
        if record.key is None
        else vault.write(_encode_secret(record.base_url, record.key, record.jwt_token))
    )
    if isinstance(outcome, SecretStored):
        return outcome if _commit_token_file(staged) else CredentialNotRecorded()
    discard_staged_json(staged)
    return _keep_the_secret_in_the_file(record, outcome)


def _keep_the_secret_in_the_file(record: CliTokenRecord, outcome: SecretWrite) -> SecretSave:
    """Fall back to the owner-only file, which is all that is left when no keychain took the secret"""
    try:
        _write_token_file(record)
    except OSError as error:
        return CredentialNotSaved(str(error))
    return outcome


def clear_cli_token(*, vault: SecretVault = SYSTEM_KEYRING) -> SecretClear:
    """Remove the credential from both stores. Reports whether the keychain is now free of it.

    A logout the keychain never answered keeps the token file, with its secret taken out, because
    that file is the only remaining record that something may still be in there to remove. It is
    what lets a later run tell a machine with a credential it cannot reach apart from one that never
    had a login at all, and taking it away would leave the next logout answering the warning this
    one just issued with a false all-clear. The secret goes either way, and a file that will give up
    neither its copy nor itself is removed rather than kept, with the note written again afterwards
    so the warning still outlives this run.
    """
    outcome: Final = vault.erase()
    record: Final = _read_token_file()
    settled: Final = _nothing_left_behind(outcome, record)
    if not settled and _keep_the_unchecked_keychain_on_record(outcome, record):
        return outcome
    removal: Final = _remove_token_file()
    if removal is not None and record is not None and not _scrub_file_secret(record):
        return removal
    if removal is None and record is not None and _the_keychain_went_unchecked(outcome):
        _write_the_note_the_removal_took_with_it(record)
    return SecretErased() if settled else outcome


def _remove_token_file() -> CredentialNotCleared | None:
    try:
        Path(get_cli_token_file_path()).unlink(missing_ok=True)
    except OSError as error:
        return CredentialNotCleared(str(error))
    return None


def _write_the_note_the_removal_took_with_it(record: CliTokenRecord) -> None:
    """Put the secret-free note back after the file carrying it had to go to get the secret off disk.

    Reaching here means neither rewrite would take, so the file went instead, and its absence is
    what the next logout would read as a keychain already known to be clean. Removing it is also
    what frees the room the rewrite was refused for, so the note usually lands on this second try.
    When it does not, the warning this logout printed is the only one the user gets.
    """
    staged: Final = _stage_scrubbed_file(record)
    if staged is not None:
        _commit_token_file(staged)


def _keep_the_unchecked_keychain_on_record(outcome: SecretErase, record: CliTokenRecord | None) -> bool:
    """Whether the token file, stripped of its secret, is worth keeping as the note that says so.

    Only a keychain that could not be reached leaves the question open. One that answered for itself
    is remembered without any help from the file, and a file it can still pair a live entry with
    would leave the machine signed in to the login that was just ended. A copy that will give up
    its secret neither to a staged replacement nor to an overwrite is not kept either, because the
    secret goes first.
    """
    if record is None or not _the_keychain_went_unchecked(outcome):
        return False
    return _scrub_file_secret(record)


def _the_keychain_went_unchecked(outcome: SecretErase) -> bool:
    """Whether the keychain neither confirmed the erase nor answered that it still holds the secret"""
    match outcome:
        case SecretErased() | SecretStranded():
            return False
        case KeyringDisabled() | KeyringNotInstalled() | KeyringUnreachable():
            return True


def _nothing_left_behind(outcome: SecretErase, record: CliTokenRecord | None) -> bool:
    """Whether the keychain can be trusted to hold no credential of ours once the file is gone.

    A machine with no token file has no stored login to end, and `clear_cli_token` keeps one behind
    whenever the keychain is left unconfirmed, taking the secret out in place when it cannot stage a
    replacement and writing the note again when the file holding it had to go, so a missing file is
    real evidence rather than the absence of it. Past that, a
    keychain that could not be reached is never trusted, whatever the file looks like. Even a file
    holding its own secret says only that the login which wrote it had no keychain to write to, and
    the login before it may well have had one: the entry that login left outlives both the
    uninstalled package and the file that replaced it. `SecretStranded` is
    the keychain answering for itself and outranks the file.
    """
    match outcome:
        case SecretErased():
            return True
        case SecretStranded():
            return False
        case KeyringDisabled() | KeyringNotInstalled() | KeyringUnreachable():
            return record is None


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
        case KeyringNotInstalled() | KeyringDisabled() | KeyringUnreachable():
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

    A staged file that will not go into place is overwritten where it lies before the keychain is
    asked to take the new entry back, so the migration finishes on a directory that would only ever
    have refused it. Rolling back is the last resort, and a rollback the keychain also refuses
    leaves the secret in both stores until the next read, which retries this same migration.
    """
    if record.key is None:
        return None
    staged: Final = _stage_scrubbed_file(record)
    if staged is None:
        return record
    if not isinstance(vault.write(_encode_secret(record.base_url, record.key, record.jwt_token)), SecretStored):
        discard_staged_json(staged)
        return record
    if not _commit_token_file(staged) and not _overwrite_file_secret(record):
        vault.erase()
    return record


def _scrub_file_secret(record: CliTokenRecord) -> bool:
    """Leave no secret material in the token file once the vault holds it"""
    if record.key is None and not record.jwt_token:
        return True
    staged: Final = _stage_scrubbed_file(record)
    if staged is not None and _commit_token_file(staged):
        return True
    return _overwrite_file_secret(record)


def _overwrite_file_secret(record: CliTokenRecord) -> bool:
    """Take the secret out of the token file where it lies, when no replacement can be put in place.

    The atomic rewrite wants room for a second file and a directory that will accept it. A full disk
    refuses the first and a read-only `~/.litellm` the second, and neither stands in the way of
    shortening the file that is already there. It is worth the loss of atomicity because a partial
    write reads as no login at all, which is where the refused rewrite left the next run anyway.
    """
    try:
        overwrite_private_json(get_cli_token_file_path(), _without_secret(record).model_dump(exclude_none=True))
    except OSError:
        return False
    return True


def _stage_scrubbed_file(record: CliTokenRecord) -> str | None:
    staged: Final = _stage_token_file(_without_secret(record))
    return None if isinstance(staged, CredentialNotSaved) else staged


def _stage_token_file(record: CliTokenRecord) -> str | CredentialNotSaved:
    path: Final = Path(get_cli_token_file_path())
    try:
        ensure_private_dir(path.parent)
        return stage_private_json(str(path), record.model_dump(exclude_none=True))
    except OSError as error:
        return CredentialNotSaved(str(error))


def _commit_token_file(staged: str) -> bool:
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
