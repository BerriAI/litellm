import contextlib
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import click
import requests
from pydantic import TypeAdapter, ValidationError

from ..up import UpError, secure_create

AUTOROUTE_DIR: Final = Path.home() / ".litellm" / "autorouter"
CONFIG_PATH: Final = AUTOROUTE_DIR / "config.yaml"
LOG_PATH: Final = AUTOROUTE_DIR / "proxy.log"
PID_RECORD_PATH: Final = AUTOROUTE_DIR / "proxy.pid.json"


class ProcessLaunchError(Exception):
    """Raised when the ephemeral proxy subprocess fails to come up healthy."""


@dataclass(frozen=True, slots=True)
class PidRecord:
    pid: int
    port: int
    config_path: str
    log_path: str


_PID_RECORD_ADAPTER: Final = TypeAdapter(PidRecord)


_PROXY_RUNTIME_MODULES: tuple[str, ...] = ("fastapi", "uvicorn", "backoff", "orjson", "websockets", "apscheduler")


def missing_proxy_runtime_modules() -> tuple[str, ...]:
    """Proxy-server modules that ``lite autoroute up`` needs but the thin CLI install lacks.

    ``launch_proxy`` runs the full ``litellm.proxy.proxy_cli`` server, whose dependencies live in
    the ``proxy`` extra, not the ``cli`` extra that installs the ``lite`` command. On a thin
    ``litellm[cli]`` install the subprocess dies with a bare ``ModuleNotFoundError``; detecting the
    gap here lets ``up`` fail with an actionable message instead.
    """
    return tuple(name for name in _PROXY_RUNTIME_MODULES if importlib.util.find_spec(name) is None)


DEFAULT_AUTOROUTE_PORT: Final = 5483


def is_port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
        return True


def launch_proxy(config_path: Path, port: int, log_path: Path) -> "subprocess.Popen[bytes]":
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log_file:
        return subprocess.Popen(
            [
                sys.executable,
                "-m",
                "litellm.proxy.proxy_cli",
                "--config",
                str(config_path),
                "--port",
                str(port),
                "--host",
                "127.0.0.1",
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )


def _tail(log_path: Path, lines: int = 40) -> str:
    if not log_path.exists():
        return "(no log output captured)"
    return "\n".join(log_path.read_text(errors="replace").splitlines()[-lines:])


def poll_liveliness(base_url: str, log_path: Path, process: "subprocess.Popen[bytes]", timeout: float = 30.0) -> None:
    """Poll /health/liveliness until it responds, the process dies, or timeout elapses."""
    deadline: Final = time.monotonic() + timeout
    url: Final = base_url.rstrip("/") + "/health/liveliness"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise ProcessLaunchError(
                f"Ephemeral proxy exited early (code {process.returncode}). Last log lines:\n{_tail(log_path)}"
            )
        with contextlib.suppress(requests.RequestException):
            if requests.get(url, timeout=2).status_code == 200:
                return
        time.sleep(0.5)
    raise ProcessLaunchError(
        f"Ephemeral proxy never became healthy within {timeout}s. Last log lines:\n{_tail(log_path)}"
    )


def write_pid_record(record: PidRecord, path: Path | None = None) -> None:
    resolved_path: Final = path if path is not None else PID_RECORD_PATH
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved_path, "w") as f:
        json.dump(
            {"pid": record.pid, "port": record.port, "config_path": record.config_path, "log_path": record.log_path},
            f,
            indent=2,
        )


def read_pid_record(path: Path | None = None) -> PidRecord | None:
    resolved_path: Final = path if path is not None else PID_RECORD_PATH
    if not resolved_path.exists():
        return None
    with open(resolved_path, "r") as f:
        content: Final = f.read()
    try:
        return _PID_RECORD_ADAPTER.validate_json(content)
    except ValidationError:
        raise UpError(f"{resolved_path} contains invalid or unexpected JSON; cannot proceed safely.")


def clear_pid_record(path: Path | None = None) -> None:
    resolved_path: Final = path if path is not None else PID_RECORD_PATH
    resolved_path.unlink(missing_ok=True)


_PROCESS_QUERY_LIMITED_INFORMATION: Final = 0x1000
_STILL_ACTIVE: Final = 259
_ERROR_ACCESS_DENIED: Final = 5


@dataclass(frozen=True, slots=True)
class _WindowsProcessApi:
    """The kernel32 surface ``_windows_pid_exists`` needs, injected so the probe is testable off Windows."""

    open_query_handle: Callable[[int], int]
    exit_code: Callable[[int], int | None]
    close_handle: Callable[[int], None]
    last_error: Callable[[], int]


def _load_windows_process_api() -> _WindowsProcessApi:  # pragma: no cover - kernel32 is Windows-only; CI is Linux
    import ctypes
    from ctypes import wintypes

    kernel32: Final = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL

    def exit_code(handle: int) -> int | None:
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return None
        return code.value

    return _WindowsProcessApi(
        open_query_handle=lambda pid: kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid),
        exit_code=exit_code,
        close_handle=kernel32.CloseHandle,
        last_error=ctypes.get_last_error,
    )


def _windows_pid_exists(pid: int, api: _WindowsProcessApi | None = None) -> bool:
    """Liveness check for Windows that does not signal the process.

    Opens a query-only handle and asks for the exit code.  ``OpenProcess``
    failing with ``ERROR_ACCESS_DENIED`` means the pid exists but is not ours
    to open, which is the case the ``PermissionError`` arm covers on POSIX.
    An unreadable exit code is reported as alive rather than guessed away.

    Caveat kept deliberately: a process whose real exit code is 259 reads as
    alive, because ``GetExitCodeProcess`` reports ``STILL_ACTIVE`` (259) for a
    running process and cannot distinguish the two.  That is the standard
    trade-off for this API and it is strictly better than the previous
    behaviour, which killed the process it was asked about.
    """
    resolved: Final = api if api is not None else _load_windows_process_api()
    handle: Final = resolved.open_query_handle(pid)
    if not handle:
        return resolved.last_error() == _ERROR_ACCESS_DENIED
    try:
        code: Final = resolved.exit_code(handle)
        return code is None or code == _STILL_ACTIVE
    finally:
        resolved.close_handle(handle)


def is_running(pid: int) -> bool:
    """Report whether ``pid`` names a live process, without signalling it.

    ``os.kill(pid, 0)`` is the POSIX idiom and is a genuine no-op there, but it
    is not a probe on Windows.  ``signal.CTRL_C_EVENT`` is 0, so ``os.kill(pid,
    0)`` *is* ``os.kill(pid, CTRL_C_EVENT)`` and reaches
    ``GenerateConsoleCtrlEvent``.  That API's second argument is a process
    *group*, and ``launch_proxy`` above starts the proxy through
    ``subprocess.Popen`` with no ``CREATE_NEW_PROCESS_GROUP``, so the child
    shares this console's group: the Ctrl-C is delivered to the proxy, to the
    ``lite`` process asking the question, and to anything else on the console.

    Measured on Windows 11, Python 3.12.10.  Probing a sleeping child returns
    ``True`` with no exception raised; the child then exits with 3221225786
    (``0xC000013A``, ``STATUS_CONTROL_C_EXIT``) and a ``KeyboardInterrupt``
    arrives in the caller a moment later -- so the traceback does not point at
    the probe.  ``KeyboardInterrupt`` is a ``BaseException``, so neither the
    ``ProcessLookupError`` nor the ``PermissionError`` arm below ever sees it.
    """
    if sys.platform == "win32":
        return _windows_pid_exists(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminate(pid: int, grace_period: float = 5.0) -> None:
    """Terminate a process by pid, escalating from SIGTERM to SIGKILL if needed.

    Both kills tolerate the same errors ``prisma_client`` already tolerates for
    this exact case -- "already dead or inaccessible".  ``ProcessLookupError``
    alone is not enough: on Windows ``os.kill`` routes to ``TerminateProcess``,
    and a process that has already exited answers ``ERROR_ACCESS_DENIED``, so
    the call raises ``PermissionError`` and escapes the suppression.
    """
    if not is_running(pid):
        return
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.kill(pid, signal.SIGTERM)
    deadline: Final = time.monotonic() + grace_period
    while time.monotonic() < deadline and is_running(pid):
        time.sleep(0.2)
    if is_running(pid):
        # signal.SIGKILL does not exist on Windows.  Referencing it raises
        # AttributeError, which contextlib.suppress(ProcessLookupError) does
        # not catch, so this escalation path crashed instead of hard-killing
        # a proxy that ignored SIGTERM.  os.kill with any signal other than 0
        # or 1 routes to TerminateProcess on Windows, so SIGTERM is a real
        # kill there.  litellm/proxy/db/prisma_client.py already resolves the
        # signal this way.
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))


def stream_log(log_path: Path, stop_event: threading.Event) -> None:
    """Print new lines appended to log_path until stop_event is set. Blocks the calling thread."""
    while not log_path.exists() and not stop_event.is_set():
        time.sleep(0.1)
    if stop_event.is_set() or not log_path.exists():
        return
    with open(log_path, "r") as f:
        while not stop_event.is_set():
            line = f.readline()
            if line:
                click.echo(line, nl=False)
            else:
                time.sleep(0.2)


__all__ = [
    "AUTOROUTE_DIR",
    "CONFIG_PATH",
    "DEFAULT_AUTOROUTE_PORT",
    "LOG_PATH",
    "PID_RECORD_PATH",
    "PidRecord",
    "ProcessLaunchError",
    "clear_pid_record",
    "is_port_available",
    "is_running",
    "launch_proxy",
    "missing_proxy_runtime_modules",
    "poll_liveliness",
    "read_pid_record",
    "secure_create",
    "stream_log",
    "terminate",
    "write_pid_record",
]
