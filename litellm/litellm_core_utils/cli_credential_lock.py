import ctypes.wintypes
import errno
import os
import stat
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from filelock import Timeout

if TYPE_CHECKING or sys.platform != "win32":
    import fcntl


@contextmanager
def credential_lock(home: Path, timeout: float = 30) -> Generator[None, None, None]:
    """Serialize credential changes on this host without writing inside the home directory."""
    if sys.platform == "win32":
        with _windows_mutex(home, timeout):
            yield
        return
    fd: Final = _open_posix_lock(home)
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
            os.utime(fd, None)
            if os.fstat(fd).st_nlink != 1:
                raise OSError("The CLI lock file was removed while waiting")
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _open_posix_lock(home: Path) -> int:
    if sys.platform == "win32":
        raise OSError("POSIX lock files are unavailable on Windows")
    directory: Final = Path("/tmp") / f"litellm-cli-{os.getuid()}"
    directory.mkdir(mode=0o700, exist_ok=True)
    directory_fd: Final = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        directory_stat: Final = os.fstat(directory_fd)
        if directory_stat.st_uid != os.getuid() or stat.S_IMODE(directory_stat.st_mode) & 0o077:
            raise PermissionError("The CLI lock directory must be private and owned by the current user")
        identity: Final = sha256(str(home.resolve()).encode()).hexdigest()
        fd: Final = os.open(
            f"{identity}.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=directory_fd
        )
        try:
            file_stat: Final = os.fstat(fd)
            if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_uid != os.getuid() or file_stat.st_nlink != 1:
                raise PermissionError("The CLI lock must be a regular file owned only by the current user")
            return fd
        except OSError:
            os.close(fd)
            raise
    finally:
        os.close(directory_fd)


@contextmanager
def _windows_mutex(home: Path, timeout: float) -> Generator[None, None, None]:
    kernel: Final = ctypes.WinDLL("kernel32", use_last_error=True)
    create: Final = ctypes.WINFUNCTYPE(
        ctypes.wintypes.HANDLE, ctypes.c_void_p, ctypes.wintypes.BOOL, ctypes.wintypes.LPCWSTR, use_last_error=True
    )(("CreateMutexW", kernel))
    wait: Final = ctypes.WINFUNCTYPE(
        ctypes.wintypes.DWORD, ctypes.wintypes.HANDLE, ctypes.wintypes.DWORD, use_last_error=True
    )(("WaitForSingleObject", kernel))
    release: Final = ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HANDLE, use_last_error=True)(
        ("ReleaseMutex", kernel)
    )
    close: Final = ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HANDLE, use_last_error=True)(
        ("CloseHandle", kernel)
    )
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
