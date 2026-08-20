"""
CLI Keyring Access

SDK-level access to the OS keychain (macOS Keychain, Windows Credential Manager,
Linux Secret Service) that holds the credential minted by `lite login`.

The `keyring` package is optional and imported lazily, so importing this module
never pulls it in. Every failure is returned as a value, naming which of the
three ways the keychain can be out of reach applies, so callers can degrade to
the token file and tell the user what to do about it.
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
class KeyringNotInstalled:
    pass


@dataclass(frozen=True, slots=True)
class KeyringDisabled:
    pass


@dataclass(frozen=True, slots=True)
class KeyringUnreachable:
    pass


KeyringUnusable: TypeAlias = KeyringNotInstalled | KeyringDisabled | KeyringUnreachable
SecretRead: TypeAlias = SecretFound | SecretMissing | KeyringUnusable
SecretWrite: TypeAlias = SecretStored | KeyringUnusable


class SecretVault(Protocol):
    """The single slot holding the CLI credential's secret material."""

    def read(self) -> SecretRead: ...

    def write(self, blob: str) -> SecretWrite: ...

    def erase(self) -> bool: ...


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
        api: Final = _keyring_api()
        if isinstance(api, (KeyringNotInstalled, KeyringDisabled)):
            return api
        try:
            api.set_password(KEYRING_SERVICE, KEYRING_ACCOUNT, blob)
        except Exception:  # noqa: BLE001  # a keychain that refuses the write falls back to the token file
            return KeyringUnreachable()
        return SecretStored()

    def erase(self) -> bool:
        """Whether the keychain is guaranteed to hold no credential afterwards.

        An uninstalled `keyring` package can never have stored one. A kill switch set after
        a credential was stored leaves that entry out of reach, so erasure cannot be promised.
        """
        if _import_keyring() is None:
            return True
        if _keyring_disabled():
            return False
        match self.read():
            case KeyringNotInstalled() | KeyringDisabled() | KeyringUnreachable():
                return False
            case SecretMissing():
                return True
            case SecretFound():
                return self._delete()

    def _delete(self) -> bool:
        api: Final = _keyring_api()
        if isinstance(api, (KeyringNotInstalled, KeyringDisabled)):
            return False
        try:
            api.delete_password(KEYRING_SERVICE, KEYRING_ACCOUNT)
        except Exception:  # noqa: BLE001  # report the failure as a value so `lite logout` can warn
            return False
        return True


SYSTEM_KEYRING: Final[SecretVault] = KeyringVault()
