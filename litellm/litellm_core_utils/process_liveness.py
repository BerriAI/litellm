import ctypes
import os
import sys
from typing import Final

_SYNCHRONIZE: Final = 0x00100000
_PROCESS_QUERY_LIMITED_INFORMATION: Final = 0x1000
_ERROR_ACCESS_DENIED: Final = 5
_WAIT_TIMEOUT: Final = 0x102


def pid_exists(pid: int) -> bool:
    if sys.platform == "win32":
        return _pid_exists_windows(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _pid_exists_windows(pid: int) -> bool:
    # os.kill(pid, 0) on Windows calls TerminateProcess, so liveness must come from the Win32 process handle
    from ctypes import wintypes

    kernel32: Final = ctypes.WinDLL("kernel32", use_last_error=True)
    open_process: Final = ctypes.WINFUNCTYPE(
        wintypes.HANDLE, wintypes.DWORD, wintypes.BOOL, wintypes.DWORD, use_last_error=True
    )(("OpenProcess", kernel32))
    wait_for_single_object: Final = ctypes.WINFUNCTYPE(wintypes.DWORD, wintypes.HANDLE, wintypes.DWORD)(
        ("WaitForSingleObject", kernel32)
    )
    close_handle: Final = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HANDLE)(("CloseHandle", kernel32))

    handle: Final = open_process(_SYNCHRONIZE | _PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return ctypes.get_last_error() == _ERROR_ACCESS_DENIED
    try:
        return wait_for_single_object(handle, 0) == _WAIT_TIMEOUT
    finally:
        close_handle(handle)
