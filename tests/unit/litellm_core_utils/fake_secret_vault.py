from litellm.litellm_core_utils.cli_keyring import (
    KeyringDiscardsWrites,
    KeyringUnreachable,
    KeyringUnusable,
    SecretErase,
    SecretErased,
    SecretFound,
    SecretMissing,
    SecretRead,
    SecretStored,
    SecretStranded,
    SecretWrite,
)


class FakeSecretVault:
    """In-memory stand-in for the OS keychain, injected wherever CLI credential storage is exercised.

    `available=False` models a keychain that is locked or has no backend, `writable=False` one that
    refuses to store, `erasable=False` one that will not release what it already holds, and `failure`
    picks which unusable state those report. `discards=True` is keyring's null backend, which answers
    reads and erases like any other yet keeps nothing it is given, so only writes report it.
    """

    def __init__(
        self,
        blob: str | None = None,
        *,
        available: bool = True,
        writable: bool = True,
        erasable: bool = True,
        discards: bool = False,
        failure: KeyringUnusable = KeyringUnreachable(),
    ) -> None:
        self.blob: str | None = blob
        self.available: bool = available
        self.writable: bool = writable
        self.erasable: bool = erasable
        self.discards: bool = discards
        self.failure: KeyringUnusable = failure
        self.reads: int = 0
        self.writes: list[str] = []
        self.erases: int = 0

    def read(self) -> SecretRead:
        self.reads += 1
        if not self.available:
            return self.failure
        return SecretMissing() if self.blob is None else SecretFound(self.blob)

    def write(self, blob: str) -> SecretWrite:
        self.writes.append(blob)
        if not (self.available and self.writable):
            return self.failure
        if self.discards:
            return KeyringDiscardsWrites()
        self.blob = blob
        return SecretStored()

    def erase(self) -> SecretErase:
        self.erases += 1
        if not self.available:
            return self.failure
        if not self.erasable:
            return SecretStranded() if self.blob is not None else SecretErased()
        self.blob = None
        return SecretErased()
