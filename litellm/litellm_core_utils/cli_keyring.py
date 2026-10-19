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
"""

import os
import threading
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Final, Protocol, TypeAlias

KEYRING_SERVICE: Final = "litellm-cli"
KEYRING_ACCOUNT: Final = "credential"
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


def _forget_the_preflight(api: KeyringApi) -> None:
    """Take the throwaway probe back out.

    A backend that kept nothing has nothing to remove, and the probe is worth nothing either way,
    so a keychain that refuses to give it up costs the caller nothing.
    """
    with suppress(Exception):
        api.delete_password(KEYRING_SERVICE, KEYRING_PREFLIGHT_ACCOUNT)


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
            blob: Final = api.get_password(KEYRING_SERVICE, KEYRING_ACCOUNT)
        except Exception:  # noqa: BLE001  # backends raise outside keyring.errors; never break the SDK
            return KeyringUnreachable()
        return SecretMissing() if blob is None else SecretFound(blob)

    def write(self, blob: str) -> SecretWrite:
        """Store the secret, reporting stored only once the keychain hands the same bytes back.

        A backend that accepts writes and keeps nothing, which is exactly what `keyring --disable`
        and `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring` select, raises nothing to
        distinguish itself. Reading the value back is the only way to tell it apart from a keychain
        that really stored the credential, and the caller is about to drop its own copy on our word.

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
        _forget_the_preflight(api)
        try:
            api.set_password(KEYRING_SERVICE, KEYRING_ACCOUNT, blob)
        except Exception:  # noqa: BLE001  # a keychain that refuses the write falls back to the token file
            return KeyringUnreachable()
        return SecretStored() if self.read() == SecretFound(blob) else KeyringDiscardsWrites()

    def erase(self) -> SecretErase:
        """Remove our entry, reporting whether the keychain is guaranteed to be free of it.

        A keychain out of reach is never an erasure: the entry belongs to the OS, not to this
        install, so it outlives an uninstalled `keyring` package and a kill switch set after login.
        Those cases are reported apart from a confirmed entry that would not delete, because only
        the caller knows whether this machine ever put a secret in a keychain.
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
            api.delete_password(KEYRING_SERVICE, KEYRING_ACCOUNT)
        except Exception:  # noqa: BLE001  # report the failure as a value so `lite logout` can warn
            return SecretStranded()
        return SecretErased()


SYSTEM_KEYRING: Final[SecretVault] = KeyringVault()
