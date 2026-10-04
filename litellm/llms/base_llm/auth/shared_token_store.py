"""Same-host token store for the JWT-bearer exchange engine.

Anthropic accepts an assertion carrying a ``jti`` once per issuer, so every uvicorn worker that
reads the same projected token file must share the token the first exchange minted instead of
re-sending the same assertion. The engine keys the store by cache key and only reuses a stored
token minted from the assertion it currently holds; a rotated assertion always buys a fresh token.
"""

import contextlib
import os
import sys
import tempfile
import threading
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

from pydantic import BaseModel, SecretStr, ValidationError

from litellm._logging import verbose_logger

CACHE_DIR_ENV: Final = "LITELLM_TOKEN_EXCHANGE_CACHE_DIR"


@dataclass(frozen=True, slots=True)
class StoredToken:
    access_token: SecretStr
    expires_at_epoch: float | None
    assertion_sha256: str


class SharedTokenStore(Protocol):
    """Every method is best-effort: a store that cannot read, write, or lock degrades to a per-process
    cache and never raises into the mint path."""

    def load(self, key: str) -> StoredToken | None: ...

    def save(self, key: str, token: StoredToken) -> None: ...

    def delete(self, key: str) -> None: ...

    def lock(self, key: str) -> contextlib.AbstractContextManager[None]: ...


class _StoredTokenFile(BaseModel):
    access_token: str
    expires_at_epoch: float | None
    assertion_sha256: str


def _directory_is_private(directory: Path) -> bool:
    try:
        directory.mkdir(mode=0o700, exist_ok=True)
        stat: Final = directory.stat()
    except OSError as e:
        verbose_logger.warning("Token exchange cache directory %s is unusable (%s); caching per process", directory, e)
        return False
    if stat.st_uid != os.getuid() or stat.st_mode & 0o077:
        verbose_logger.warning(
            "Token exchange cache directory %s must be owned by uid %d with mode 0700; caching per process",
            directory,
            os.getuid(),
        )
        return False
    return True


def _unlink(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink()


def _write_token_file(directory: Path, key: str, body: bytes) -> None:
    """The token is staged in its own file and renamed over the entry, so a reader never sees a
    half-written one. Every failure unlinks the staging file, including the buffered write that only
    reaches the disk when the handle closes: nothing else sweeps this directory, and that file holds
    a token that still works. The rename leaves nothing behind for the unlink to find."""
    descriptor, name = tempfile.mkstemp(dir=directory, prefix=f"{key}.")
    os.close(descriptor)
    staged: Final = Path(name)
    try:
        staged.write_bytes(body)
        os.replace(staged, directory / f"{key}.json")
    finally:
        _unlink(staged)


class FileTokenStore:
    """One ``<cache key>.json`` (mode 0600) and one ``<cache key>.lock`` (flock) per identity under a
    directory only the proxy's uid can enter; the directory is checked on first use, not at import."""

    def __init__(self, directory: Path) -> None:
        self._directory: Final = directory
        self._ready_lock: Final = threading.Lock()
        self._ready: bool | None = None

    @property
    def directory(self) -> Path:
        return self._directory

    def _usable(self) -> bool:
        with self._ready_lock:
            if self._ready is None:
                self._ready = _directory_is_private(self._directory)
            return self._ready

    def load(self, key: str) -> StoredToken | None:
        if not self._usable():
            return None
        try:
            raw: Final = (self._directory / f"{key}.json").read_bytes()
            parsed: Final = _StoredTokenFile.model_validate_json(raw)
        except FileNotFoundError:
            return None
        except (OSError, ValidationError) as e:
            verbose_logger.debug("Ignoring unreadable token exchange cache entry: %s", e)
            return None
        return StoredToken(
            access_token=SecretStr(parsed.access_token),
            expires_at_epoch=parsed.expires_at_epoch,
            assertion_sha256=parsed.assertion_sha256,
        )

    def save(self, key: str, token: StoredToken) -> None:
        if not self._usable():
            return
        body: Final = (
            _StoredTokenFile(
                access_token=token.access_token.get_secret_value(),
                expires_at_epoch=token.expires_at_epoch,
                assertion_sha256=token.assertion_sha256,
            )
            .model_dump_json()
            .encode()
        )
        try:
            _write_token_file(self._directory, key, body)
        except OSError as e:
            verbose_logger.debug("Token exchange cache entry not written: %s", e)

    def delete(self, key: str) -> None:
        if not self._usable():
            return
        with contextlib.suppress(FileNotFoundError, OSError):
            (self._directory / f"{key}.json").unlink()

    @contextlib.contextmanager
    def lock(self, key: str) -> Generator[None]:
        if sys.platform == "win32" or not self._usable():
            yield
            return
        import fcntl

        try:
            fd: Final = os.open(self._directory / f"{key}.lock", os.O_RDWR | os.O_CREAT, 0o600)
        except OSError as e:
            verbose_logger.debug("Token exchange cache lock unavailable (%s); minting without it", e)
            yield
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


def default_shared_token_store() -> SharedTokenStore | None:
    """``LITELLM_TOKEN_EXCHANGE_CACHE_DIR`` relocates the store; setting it empty disables it. Without
    it the store lives under the temp directory, keyed by uid, so the workers of one proxy share it and
    other users on the host cannot read it. Windows has no ``flock``, so it caches per process there."""
    if sys.platform == "win32":
        return None
    configured: Final = os.environ.get(CACHE_DIR_ENV)
    if configured == "":
        return None
    if configured is not None:
        return FileTokenStore(Path(configured))
    return FileTokenStore(Path(tempfile.gettempdir()) / f"litellm-token-exchange-{os.getuid()}")
