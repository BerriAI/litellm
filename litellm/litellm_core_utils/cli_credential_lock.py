import ctypes
import errno
import os
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from ctypes import wintypes
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from filelock import Timeout

if TYPE_CHECKING or sys.platform != "win32":
    import fcntl


@contextmanager
def credential_lock(home: Path, timeout: float = 30) -> Generator[None, None, None]:
    """Serialize credential changes without creating or modifying a lock file."""
    if sys.platform == "win32":
        with _windows_mutex(home, timeout):
            yield
        return
    fd: Final = os.open(home, os.O_RDONLY | os.O_DIRECTORY)
    try:
        deadline: Final = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK):
                    raise
                if time.monotonic() >= deadline:
                    raise Timeout(str(home)) from None
                time.sleep(0.05)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


@contextmanager
def _windows_mutex(home: Path, timeout: float) -> Generator[None, None, None]:
    kernel: Final = ctypes.WinDLL("kernel32", use_last_error=True)
    create: Final = ctypes.WINFUNCTYPE(
        wintypes.HANDLE, ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR, use_last_error=True
    )(("CreateMutexW", kernel))
    wait: Final = ctypes.WINFUNCTYPE(wintypes.DWORD, wintypes.HANDLE, wintypes.DWORD, use_last_error=True)(
        ("WaitForSingleObject", kernel)
    )
    release: Final = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HANDLE, use_last_error=True)(("ReleaseMutex", kernel))
    close: Final = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HANDLE, use_last_error=True)(("CloseHandle", kernel))
    identity: Final = sha256(os.path.normcase(str(home.resolve())).encode()).hexdigest()
    handle: Final = cast(int | None, create(None, False, f"Global\\litellm-cli-{identity}"))
    if handle is None:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        result: Final = cast(int, wait(handle, max(0, int(timeout * 1000))))
        if result == 0x102:
            raise Timeout(str(home))
        if result not in (0, 0x80):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            yield
        finally:
            if not release(handle) and sys.exc_info()[0] is None:
                raise ctypes.WinError(ctypes.get_last_error())
    finally:
        close(handle)
