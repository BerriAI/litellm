"""
CLI Keyring Access

SDK-level access to the OS keychain (macOS Keychain, Windows Credential Manager,
Linux Secret Service) that holds the credential minted by `lite login`.

The `keyring` package is optional and imported lazily, so importing this module
never pulls it in. Every failure is returned as a value, naming which of the
ways the keychain can be out of reach applies, so callers can degrade to the
token file and tell the user what to do about it.

A write is only reported as stored once it has been read back, because keyring's
null backend, which `keyring --disable` and headless CI images both select,
accepts every write and keeps nothing. Writes are also pre-flighted with a
throwaway value, because a keychain can answer neither way and block forever.

A write parks the new secret in a staging entry before it replaces the live one,
because keyring replaces an entry on macOS by deleting it and then adding the new
one. A process stopped between the two would otherwise leave no credential at all,
after a renewal already spent the refresh token that could have minted another.
"""

import hashlib
import os
import threading
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Final, Protocol, TypeAlias

from pydantic import ConfigDict, ValidationError

from litellm.types.llms.base_model import LiteLLMBaseModel

KEYRING_SERVICE: Final = "litellm-cli"
KEYRING_ACCOUNT: Final = "credential"
KEYRING_STAGING_ACCOUNT: Final = "credential-staging"
KEYRING_PREFLIGHT_ACCOUNT: Final = "credential-preflight"
DISABLE_KEYRING_ENV_VAR: Final = "LITELLM_CLI_DISABLE_KEYRING"

_DISABLED_VALUES: Final = frozenset(("1", "true", "yes", "on"))
_PREFLIGHT_VALUE: Final = "preflight"
_PREFLIGHT_TIMEOUT_SECONDS: Final = 5.0


@dataclass(frozen=True, slots=True)
class SecretFound:
    blob: str


@dataclass(frozen=True, slots=True)
class SecretMissing:
    pass


@dataclass(frozen=True, slots=True)
class SecretStored:
    pass


@dataclass(frozen=True, slots=True)
class SecretErased:
    pass


@dataclass(frozen=True, slots=True)
class SecretStranded:
    pass


@dataclass(frozen=True, slots=True)
class KeyringNotInstalled:
    pass


@dataclass(frozen=True, slots=True)
class KeyringDisabled:
    pass


@dataclass(frozen=True, slots=True)
class KeyringUnreachable:
    pass


@dataclass(frozen=True, slots=True)
class KeyringDiscardsWrites:
    pass


KeyringUnusable: TypeAlias = KeyringNotInstalled | KeyringDisabled | KeyringUnreachable
SecretRead: TypeAlias = SecretFound | SecretMissing | KeyringUnusable
SecretWrite: TypeAlias = SecretStored | KeyringUnusable | KeyringDiscardsWrites
SecretErase: TypeAlias = SecretErased | SecretStranded | KeyringUnusable


class SecretVault(Protocol):
    """The single slot holding the CLI credential's secret material."""

    def read(self) -> SecretRead: ...

    def write(self, blob: str) -> SecretWrite: ...

    def erase(self) -> SecretErase: ...


class KeyringApi(Protocol):
    def get_password(self, service_name: str, username: str) -> str | None: ...

    def set_password(self, service_name: str, username: str, password: str) -> None: ...

    def delete_password(self, service_name: str, username: str) -> None: ...


def _keyring_disabled() -> bool:
    return os.getenv(DISABLE_KEYRING_ENV_VAR, "").strip().lower() in _DISABLED_VALUES


def _import_keyring() -> KeyringApi | None:
    try:
        import keyring
    except ImportError:
        return None
    return keyring


def _keyring_api() -> KeyringApi | KeyringNotInstalled | KeyringDisabled:
    if _keyring_disabled():
        return KeyringDisabled()
    api: Final = _import_keyring()
    return KeyringNotInstalled() if api is None else api


def _answers_a_write(api: KeyringApi, timeout_seconds: float) -> bool:
    """Whether the keychain answers a write at all, asked with a value worth nothing.

    macOS derives the login keychain from `$HOME`, and `set_password` against a HOME with no usable
    one blocks forever with no timeout of its own. Containers, CI images, `sudo -H`, and service
    accounts all run there, and reads answer normally, so nothing cheaper tells them apart. Asking
    with a throwaway value keeps a keychain that never answers from taking `lite login` down with
    it, and keeps the real credential out of a store that might accept it long after we gave up.
    A keychain that refuses the probe outright still answered it, so only silence counts against it.
    """
    answered: Final = threading.Event()

    def ask() -> None:
        with suppress(Exception):
            api.set_password(KEYRING_SERVICE, KEYRING_PREFLIGHT_ACCOUNT, _PREFLIGHT_VALUE)
        answered.set()

    threading.Thread(target=ask, daemon=True, name="litellm-cli-keyring-preflight").start()
    return answered.wait(timeout_seconds)


def _forget(api: KeyringApi, account: str) -> None:
    """Take an entry that is worth nothing any more back out: the pre-flight probe, or a staged
    write whose live write has settled. One that is already gone or that the keychain refuses to
    give up costs the caller nothing."""
    with suppress(Exception):
        api.delete_password(KEYRING_SERVICE, account)


class _StagedWrite(LiteLLMBaseModel):
    """A secret parked before the live entry is replaced, with a fingerprint of the entry it replaces.

    The fingerprint is what tells an unfinished write apart from a later one that went to the live
    entry alone, such as a login by a litellm version that predates staging.
    """

    model_config = ConfigDict(frozen=True)

    replaces: str | None
    secret: str


def _fingerprint(blob: str | None) -> str | None:
    return None if blob is None else hashlib.sha256(blob.encode()).hexdigest()


def _parse_staged_write(raw: str | None) -> _StagedWrite | None:
    if raw is None:
        return None
    try:
        return _StagedWrite.model_validate_json(raw)
    except ValidationError:
        return None


def _newest(live: str | None, staged: _StagedWrite | None) -> SecretFound | SecretMissing:
    """The live entry, unless a staged write never got to replace it.

    That is the case while the live entry is gone or still holds what the staged write replaces,
    and then the staged secret is the newest one this machine saved, possibly the only one left
    after a renewal spent the refresh token behind the live entry.
    """
    if staged is not None and (live is None or _fingerprint(live) == staged.replaces):
        return SecretFound(staged.secret)
    return SecretMissing() if live is None else SecretFound(live)


def _store(api: KeyringApi, account: str, value: str) -> SecretWrite:
    try:
        api.set_password(KEYRING_SERVICE, account, value)
        kept: Final = api.get_password(KEYRING_SERVICE, account)
    except Exception:  # noqa: BLE001  # a keychain that refuses the write falls back to the token file
        return KeyringUnreachable()
    return SecretStored() if kept == value else KeyringDiscardsWrites()


def _delete_if_present(api: KeyringApi, account: str) -> None:
    if api.get_password(KEYRING_SERVICE, account) is not None:
        api.delete_password(KEYRING_SERVICE, account)


@dataclass(frozen=True, slots=True)
class KeyringVault:
    """The OS keychain, reached through the optional `keyring` package.

    A keychain that let the pre-flight time out is not asked anything else for the rest of the
    process. The probe that timed out is still sitting in the keychain on a thread of its own, and
    it holds the keychain against every later call, so the read after it would block on the main
    thread with no timeout to save it. One silence is answer enough.
    """

    preflight_timeout_seconds: float = _PREFLIGHT_TIMEOUT_SECONDS
    stopped_answering: threading.Event = field(default_factory=threading.Event, compare=False, repr=False)

    def read(self) -> SecretRead:
        if self.stopped_answering.is_set():
            return KeyringUnreachable()
        api: Final = _keyring_api()
        if isinstance(api, (KeyringNotInstalled, KeyringDisabled)):
            return api
        try:
            live: Final = api.get_password(KEYRING_SERVICE, KEYRING_ACCOUNT)
            staged: Final = api.get_password(KEYRING_SERVICE, KEYRING_STAGING_ACCOUNT)
        except Exception:  # noqa: BLE001  # backends raise outside keyring.errors; never break the SDK
            return KeyringUnreachable()
        return _newest(live, _parse_staged_write(staged))

    def write(self, blob: str) -> SecretWrite:
        """Store the secret, reporting stored only once the keychain hands the same bytes back.

        A backend that accepts writes and keeps nothing, which is exactly what `keyring --disable`
        and `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring` select, raises nothing to
        distinguish itself. Reading the value back is the only way to tell it apart from a keychain
        that really stored the credential, and the caller is about to drop its own copy on our word.

        The secret is staged and read back before the live entry is touched, so a process stopped
        anywhere after that leaves it for the next read. A live write that fails while the process
        is still running takes the staged copy back out, because the caller then keeps the secret
        in the token file instead.

        The keychain is pre-flighted first, because one that blocks rather than answering would
        otherwise hang `lite login` outright.
        """
        if self.stopped_answering.is_set():
            return KeyringUnreachable()
        api: Final = _keyring_api()
        if isinstance(api, (KeyringNotInstalled, KeyringDisabled)):
            return api
        if not _answers_a_write(api, self.preflight_timeout_seconds):
            self.stopped_answering.set()
            return KeyringUnreachable()
        _forget(api, KEYRING_PREFLIGHT_ACCOUNT)
        try:
            replaced: Final = api.get_password(KEYRING_SERVICE, KEYRING_ACCOUNT)
        except Exception:  # noqa: BLE001  # backends raise outside keyring.errors; never break the SDK
            return KeyringUnreachable()
        staging: Final = _StagedWrite(replaces=_fingerprint(replaced), secret=blob).model_dump_json()
        staged: Final = _store(api, KEYRING_STAGING_ACCOUNT, staging)
        if not isinstance(staged, SecretStored):
            return staged
        stored: Final = _store(api, KEYRING_ACCOUNT, blob)
        _forget(api, KEYRING_STAGING_ACCOUNT)
        return stored

    def erase(self) -> SecretErase:
        """Remove our entries, reporting whether the keychain is guaranteed to be free of them.

        A keychain out of reach is never an erasure: the entry belongs to the OS, not to this
        install, so it outlives an uninstalled `keyring` package and a kill switch set after login.
        Those cases are reported apart from a confirmed entry that would not delete, because only
        the caller knows whether this machine ever put a secret in a keychain.

        A staged write goes too, since the next read would otherwise hand back the login this ended.
        """
        match self.read():
            case KeyringNotInstalled() | KeyringDisabled() | KeyringUnreachable() as unusable:
                return unusable
            case SecretMissing():
                return SecretErased()
            case SecretFound():
                return self._delete()

    def _delete(self) -> SecretErase:
        api: Final = _keyring_api()
        if isinstance(api, (KeyringNotInstalled, KeyringDisabled)):
            return api
        try:
            _delete_if_present(api, KEYRING_ACCOUNT)
            _delete_if_present(api, KEYRING_STAGING_ACCOUNT)
        except Exception:  # noqa: BLE001  # report the failure as a value so `lite logout` can warn
            return SecretStranded()
        return SecretErased()


SYSTEM_KEYRING: Final[SecretVault] = KeyringVault()
