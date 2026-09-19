import threading
import time

from litellm.litellm_core_utils.cli_keyring import (
    KEYRING_ACCOUNT,
    KEYRING_PREFLIGHT_ACCOUNT,
    KeyringApi,
    KeyringUnreachable,
    KeyringVault,
    SecretErased,
    SecretFound,
    SecretStranded,
    SecretTooLarge,
)


class _BlockingReadKeyring:
    def __init__(self) -> None:
        self.blocked = threading.Event()
        self.calls: list[tuple[str, str]] = []

    def get_password(self, service_name: str, username: str) -> str | None:
        self.calls.append(("get", username))
        self.blocked.set()
        threading.Event().wait()
        return None

    def set_password(self, service_name: str, username: str, password: str) -> None:
        self.calls.append(("set", username))

    def delete_password(self, service_name: str, username: str) -> None:
        self.calls.append(("delete", username))


class _BlockingDeleteKeyring:
    def __init__(self) -> None:
        self.blocked = threading.Event()
        self.calls: list[tuple[str, str]] = []

    def get_password(self, service_name: str, username: str) -> str | None:
        self.calls.append(("get", username))
        return "stored-blob"

    def set_password(self, service_name: str, username: str, password: str) -> None:
        self.calls.append(("set", username))

    def delete_password(self, service_name: str, username: str) -> None:
        self.calls.append(("delete", username))
        self.blocked.set()
        threading.Event().wait()


class _OversizedWriteKeyring:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def get_password(self, service_name: str, username: str) -> str | None:
        self.calls.append(("get", username))
        return None

    def set_password(self, service_name: str, username: str, password: str) -> None:
        self.calls.append(("set", username))
        if username != KEYRING_PREFLIGHT_ACCOUNT:
            raise ValueError("credential blob too large")

    def delete_password(self, service_name: str, username: str) -> None:
        self.calls.append(("delete", username))


class _RefusingWriteKeyring:
    def set_password(self, service_name: str, username: str, password: str) -> None:
        raise ValueError("keychain refused write")

    def get_password(self, service_name: str, username: str) -> str | None:
        return None

    def delete_password(self, service_name: str, username: str) -> None:
        return None


def _vault(api: KeyringApi, timeout_seconds: float = 0.02) -> KeyringVault:
    return KeyringVault(
        preflight_timeout_seconds=timeout_seconds,
        keyring_api=lambda: api,
    )


def test_blocking_read_is_bounded_and_latches():
    keyring = _BlockingReadKeyring()
    vault = _vault(keyring)

    started = time.monotonic()
    outcome = vault.read()

    assert outcome == KeyringUnreachable()
    assert time.monotonic() - started < 1
    assert keyring.blocked.is_set()
    assert vault.read() == KeyringUnreachable()
    assert keyring.calls == [("get", KEYRING_ACCOUNT)]


def test_blocking_delete_is_bounded_and_stranded():
    keyring = _BlockingDeleteKeyring()
    vault = _vault(keyring)

    started = time.monotonic()
    outcome = vault.erase()

    assert outcome == SecretStranded()
    assert time.monotonic() - started < 1
    assert keyring.blocked.is_set()
    assert vault.read() == KeyringUnreachable()
    assert keyring.calls == [("get", KEYRING_ACCOUNT), ("delete", KEYRING_ACCOUNT)]


def test_successful_read_and_delete_use_the_injected_backend():
    class _WorkingKeyring:
        def get_password(self, service_name: str, username: str) -> str | None:
            return "stored-blob"

        def set_password(self, service_name: str, username: str, password: str) -> None:
            return None

        def delete_password(self, service_name: str, username: str) -> None:
            return None

    vault = _vault(_WorkingKeyring())

    assert vault.read() == SecretFound("stored-blob")
    assert vault.erase() == SecretErased()


def test_oversized_write_is_classified_from_utf16_payload_size():
    keyring = _OversizedWriteKeyring()
    vault = _vault(keyring)

    assert vault.write("a" * 1281) == SecretTooLarge()
    assert ("set", KEYRING_PREFLIGHT_ACCOUNT) in keyring.calls
    assert ("set", KEYRING_ACCOUNT) in keyring.calls


def test_small_write_failure_remains_unreachable():
    keyring = _OversizedWriteKeyring()

    assert _vault(keyring).write("small") == KeyringUnreachable()


def test_oversized_write_from_refusing_probe_remains_unreachable():
    keyring = _RefusingWriteKeyring()

    assert _vault(keyring).write("a" * 1281) == KeyringUnreachable()
