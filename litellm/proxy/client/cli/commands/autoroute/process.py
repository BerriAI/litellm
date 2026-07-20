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
from dataclasses import dataclass
from pathlib import Path

import click
import requests
from pydantic import TypeAdapter, ValidationError

from ..up import UpError, secure_create

AUTOROUTE_DIR = Path.home() / ".litellm" / "autorouter"
CONFIG_PATH = AUTOROUTE_DIR / "config.yaml"
LOG_PATH = AUTOROUTE_DIR / "proxy.log"
PID_RECORD_PATH = AUTOROUTE_DIR / "proxy.pid.json"


class ProcessLaunchError(Exception):
    """Raised when the ephemeral proxy subprocess fails to come up healthy."""


@dataclass(frozen=True, slots=True)
class PidRecord:
    pid: int
    port: int
    config_path: str
    log_path: str


_PID_RECORD_ADAPTER = TypeAdapter(PidRecord)


_PROXY_RUNTIME_MODULES: tuple[str, ...] = ("fastapi", "uvicorn", "backoff", "orjson", "websockets", "apscheduler")


def missing_proxy_runtime_modules() -> tuple[str, ...]:
    """Proxy-server modules that ``lite autoroute up`` needs but the thin CLI install lacks.

    ``launch_proxy`` runs the full ``litellm.proxy.proxy_cli`` server, whose dependencies live in
    the ``proxy`` extra, not the ``cli`` extra that installs the ``lite`` command. On a thin
    ``litellm[cli]`` install the subprocess dies with a bare ``ModuleNotFoundError``; detecting the
    gap here lets ``up`` fail with an actionable message instead.
    """
    return tuple(name for name in _PROXY_RUNTIME_MODULES if importlib.util.find_spec(name) is None)


def allocate_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def launch_proxy(
    config_path: Path, port: int, log_path: Path, *, debug: bool = False, detailed_debug: bool = False
) -> "subprocess.Popen[bytes]":
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = (
        (
            sys.executable,
            "-m",
            "litellm.proxy.proxy_cli",
            "--config",
            str(config_path),
            "--port",
            str(port),
            "--host",
            "127.0.0.1",
        )
        + (("--debug",) if debug else ())
        + (("--detailed_debug",) if detailed_debug else ())
    )
    with open(log_path, "w") as log_file:
        return subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)


def _tail(log_path: Path, lines: int = 40) -> str:
    if not log_path.exists():
        return "(no log output captured)"
    return "\n".join(log_path.read_text(errors="replace").splitlines()[-lines:])


def poll_liveliness(base_url: str, log_path: Path, process: "subprocess.Popen[bytes]", timeout: float = 30.0) -> None:
    """Poll /health/liveliness until it responds, the process dies, or timeout elapses."""
    deadline = time.monotonic() + timeout
    url = base_url.rstrip("/") + "/health/liveliness"
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
    resolved_path = path if path is not None else PID_RECORD_PATH
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved_path, "w") as f:
        json.dump(
            {"pid": record.pid, "port": record.port, "config_path": record.config_path, "log_path": record.log_path},
            f,
            indent=2,
        )


def read_pid_record(path: Path | None = None) -> PidRecord | None:
    resolved_path = path if path is not None else PID_RECORD_PATH
    if not resolved_path.exists():
        return None
    with open(resolved_path, "r") as f:
        content = f.read()
    try:
        return _PID_RECORD_ADAPTER.validate_json(content)
    except ValidationError:
        raise UpError(f"{resolved_path} contains invalid or unexpected JSON; cannot proceed safely.")


def clear_pid_record(path: Path | None = None) -> None:
    resolved_path = path if path is not None else PID_RECORD_PATH
    resolved_path.unlink(missing_ok=True)


def is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminate(pid: int, grace_period: float = 5.0) -> None:
    """Terminate a process by pid, escalating from SIGTERM to SIGKILL if needed."""
    if not is_running(pid):
        return
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_period
    while time.monotonic() < deadline and is_running(pid):
        time.sleep(0.2)
    if is_running(pid):
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)


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
    "LOG_PATH",
    "PID_RECORD_PATH",
    "PidRecord",
    "ProcessLaunchError",
    "allocate_free_port",
    "clear_pid_record",
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
