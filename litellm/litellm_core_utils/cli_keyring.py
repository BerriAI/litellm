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
from collections.abc import Callable
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Final, Generic, Protocol, TypeAlias, TypeVar

KEYRING_SERVICE: Final = "litellm-cli"
KEYRING_ACCOUNT: Final = "credential"
KEYRING_PREFLIGHT_ACCOUNT: Final = "credential-preflight"
DISABLE_KEYRING_ENV_VAR: Final = "LITELLM_CLI_DISABLE_KEYRING"

_DISABLED_VALUES: Final = frozenset(("1", "true", "yes", "on"))
_PREFLIGHT_VALUE: Final = "preflight"
_PREFLIGHT_TIMEOUT_SECONDS: Final = 5.0
_MAX_CREDENTIAL_BLOB_BYTES: Final = 5 * 512

_T = TypeVar("_T")


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


@dataclass(frozen=True, slots=True)
class SecretTooLarge:
    pass


KeyringUnusable: TypeAlias = KeyringNotInstalled | KeyringDisabled | KeyringUnreachable
SecretRead: TypeAlias = SecretFound | SecretMissing | KeyringUnusable
SecretWrite: TypeAlias = SecretStored | KeyringUnusable | KeyringDiscardsWrites | SecretTooLarge
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


@dataclass(frozen=True, slots=True)
class _KeyringCallAnswered(Generic[_T]):
    value: _T


@dataclass(frozen=True, slots=True)
class _KeyringCallFailed:
    error: Exception


@dataclass(frozen=True, slots=True)
class _KeyringCallTimedOut:
    pass


_KeyringCallResult: TypeAlias = _KeyringCallAnswered[_T] | _KeyringCallFailed | _KeyringCallTimedOut


@dataclass(frozen=True, slots=True)
class _ProbeStored:
    pass


@dataclass(frozen=True, slots=True)
class _ProbeRefused:
    pass


@dataclass(frozen=True, slots=True)
class _ProbeSilent:
    pass


_WriteProbe: TypeAlias = _ProbeStored | _ProbeRefused | _ProbeSilent


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


def _bounded_keyring_call(call: Callable[[], _T], timeout_seconds: float) -> _KeyringCallResult[_T]:
    answers: Final = Queue[_KeyringCallResult[_T]](maxsize=1)

    def run() -> None:
        try:
            answers.put(_KeyringCallAnswered(call()))
        except Exception as error:  # noqa: BLE001  # keyring backends raise outside keyring.errors
            answers.put(_KeyringCallFailed(error))

    threading.Thread(target=run, daemon=True, name="litellm-cli-keyring-call").start()
    try:
        return answers.get(timeout=timeout_seconds)
    except Empty:
        return _KeyringCallTimedOut()


def _probe_a_write(api: KeyringApi, timeout_seconds: float) -> _WriteProbe:
    """Whether the keychain answers a write at all, asked with a value worth nothing.

    macOS derives the login keychain from `$HOME`, and `set_password` against a HOME with no usable
    one blocks forever with no timeout of its own. Containers, CI images, `sudo -H`, and service
    accounts all run there, and reads answer normally, so nothing cheaper tells them apart. Asking
    with a throwaway value keeps a keychain that never answers from taking `lite login` down with
    it, and keeps the real credential out of a store that might accept it long after we gave up.
    A keychain that refuses the probe outright still answered it, so only silence counts against it.
    """
    result: Final = _bounded_keyring_call(
        lambda: api.set_password(KEYRING_SERVICE, KEYRING_PREFLIGHT_ACCOUNT, _PREFLIGHT_VALUE),
        timeout_seconds,
    )
    match result:
        case _KeyringCallAnswered():
            return _ProbeStored()
        case _KeyringCallFailed():
            return _ProbeRefused()
        case _KeyringCallTimedOut():
            return _ProbeSilent()


def _forget_the_preflight(api: KeyringApi, timeout_seconds: float) -> bool:
    """Take the throwaway probe back out.

    A backend that kept nothing has nothing to remove, and the probe is worth nothing either way,
    so a keychain that refuses to give it up costs the caller nothing.
    """
    result: Final = _bounded_keyring_call(
        lambda: api.delete_password(KEYRING_SERVICE, KEYRING_PREFLIGHT_ACCOUNT),
        timeout_seconds,
    )
    return not isinstance(result, _KeyringCallTimedOut)


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
    keyring_api: Callable[[], KeyringApi | KeyringNotInstalled | KeyringDisabled] = _keyring_api

    def read(self) -> SecretRead:
        if self.stopped_answering.is_set():
            return KeyringUnreachable()
        api: Final = self.keyring_api()
        if isinstance(api, (KeyringNotInstalled, KeyringDisabled)):
            return api
        result: Final = _bounded_keyring_call(
            lambda: api.get_password(KEYRING_SERVICE, KEYRING_ACCOUNT),
            self.preflight_timeout_seconds,
        )
        match result:
            case _KeyringCallAnswered(value=blob):
                return SecretMissing() if blob is None else SecretFound(blob)
            case _KeyringCallFailed():
                return KeyringUnreachable()
            case _KeyringCallTimedOut():
                self.stopped_answering.set()
                return KeyringUnreachable()

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
        api: Final = self.keyring_api()
        if isinstance(api, (KeyringNotInstalled, KeyringDisabled)):
            return api
        probe: Final = _probe_a_write(api, self.preflight_timeout_seconds)
        match probe:
            case _ProbeStored() | _ProbeRefused():
                pass
            case _ProbeSilent():
                self.stopped_answering.set()
                return KeyringUnreachable()
        if not _forget_the_preflight(api, self.preflight_timeout_seconds):
            self.stopped_answering.set()
            return KeyringUnreachable()
        result: Final = _bounded_keyring_call(
            lambda: api.set_password(KEYRING_SERVICE, KEYRING_ACCOUNT, blob),
            self.preflight_timeout_seconds,
        )
        match result:
            case _KeyringCallFailed():
                match probe:
                    case _ProbeStored():
                        return SecretTooLarge() if _payload_exceeds_windows_limit(blob) else KeyringUnreachable()
                    case _ProbeRefused():
                        return KeyringUnreachable()
            case _KeyringCallTimedOut():
                self.stopped_answering.set()
                return KeyringUnreachable()
            case _KeyringCallAnswered():
                read_back: Final = self.read()
                if read_back == SecretFound(blob):
                    return SecretStored()
                if read_back == KeyringUnreachable():
                    return KeyringUnreachable()
                return KeyringDiscardsWrites()

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
        api: Final = self.keyring_api()
        if isinstance(api, (KeyringNotInstalled, KeyringDisabled)):
            return api
        result: Final = _bounded_keyring_call(
            lambda: api.delete_password(KEYRING_SERVICE, KEYRING_ACCOUNT),
            self.preflight_timeout_seconds,
        )
        match result:
            case _KeyringCallAnswered():
                return SecretErased()
            case _KeyringCallFailed():
                return SecretStranded()
            case _KeyringCallTimedOut():
                self.stopped_answering.set()
                return SecretStranded()


def _payload_exceeds_windows_limit(blob: str) -> bool:
    return len(blob.encode("utf-16-le")) > _MAX_CREDENTIAL_BLOB_BYTES


SYSTEM_KEYRING: Final[SecretVault] = KeyringVault()
