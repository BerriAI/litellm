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
accepts every write and keeps nothing.
"""

import os
from dataclasses import dataclass
from typing import Final, Protocol, TypeAlias

KEYRING_SERVICE: Final = "litellm-cli"
KEYRING_ACCOUNT: Final = "credential"
DISABLE_KEYRING_ENV_VAR: Final = "LITELLM_CLI_DISABLE_KEYRING"

_DISABLED_VALUES: Final = frozenset(("1", "true", "yes", "on"))


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


KeyringUnusable: TypeAlias = KeyringNotInstalled | KeyringDisabled | KeyringUnreachable | KeyringDiscardsWrites
SecretRead: TypeAlias = SecretFound | SecretMissing | KeyringUnusable
SecretWrite: TypeAlias = SecretStored | KeyringUnusable
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


@dataclass(frozen=True, slots=True)
class KeyringVault:
    """The OS keychain, reached through the optional `keyring` package."""

    def read(self) -> SecretRead:
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
        """
        api: Final = _keyring_api()
        if isinstance(api, (KeyringNotInstalled, KeyringDisabled)):
            return api
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
            case KeyringNotInstalled() | KeyringDisabled() | KeyringUnreachable() | KeyringDiscardsWrites() as unusable:
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
