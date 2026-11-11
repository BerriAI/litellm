import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Final, TypeVar

import httpx
import psutil
import pytest

from tests.integration._support.client import eventually
from tests.integration._support.ports import allocate_port
from tests.integration._support.process import kill_stray_process, send_signal

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="drives POSIX signals")

APP: Final = """
import asyncio


async def app(scope, receive, send):
    if scope["type"] != "http":
        return
    await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/plain")]})
    if scope["path"] != "/stream":
        await send({"type": "http.response.body", "body": b"ok"})
        return
    while True:
        await send({"type": "http.response.body", "body": b"tick\\n", "more_body": True})
        await asyncio.sleep(0.1)
"""

SERVE: Final = """
import os
import pathlib
import sys

import uvicorn



def listener():
    from litellm.proxy.rust_forward.mode import Supervised, rust_forward_mode, stop_with_supervisor

    mode = rust_forward_mode(os.environ)
    if not isinstance(mode, Supervised):
        return {"host": "127.0.0.1", "port": int(sys.argv[2])}
    stop_with_supervisor(mode)
    pathlib.Path(os.environ["MARKERS"], "python.pid").write_text(str(os.getpid()))
    return {"fd": mode.listener.fd}


if __name__ == "__main__":
    uvicorn.run("app:app", app_dir=os.path.dirname(__file__), workers=int(sys.argv[1]), log_level="warning", **listener())
"""

GATEWAY: Final = """
import asyncio
import os
import pathlib
import signal
import threading


def wait_for_eof(loop, stop):
    while os.read(0, 4096):
        pass
    loop.call_soon_threadsafe(stop.set)


async def pipe(reader, writer):
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except OSError:
        pass
    finally:
        writer.close()


async def main():
    pathlib.Path(os.environ["MARKERS"], "rust.pid").write_text(str(os.getpid()))
    host, port = os.environ["LITELLM_FORWARD_UPSTREAM"].removeprefix("http://").rsplit(":", 1)
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, stop.set)
    threading.Thread(target=wait_for_eof, args=(loop, stop), daemon=True).start()
    relays = set()

    async def relay(client_reader, client_writer):
        relays.add(asyncio.current_task())
        try:
            upstream_reader, upstream_writer = await asyncio.open_connection(host, int(port))
            await asyncio.gather(pipe(client_reader, upstream_writer), pipe(upstream_reader, client_writer))
        finally:
            relays.discard(asyncio.current_task())

    server = await asyncio.start_server(
        relay, os.environ["LITELLM_FORWARD_HOST"], int(os.environ["LITELLM_FORWARD_PORT"])
    )
    await stop.wait()
    server.close()
    if relays:
        await asyncio.wait(relays, timeout=float(os.environ["LITELLM_FORWARD_SHUTDOWN_DRAIN"]))


asyncio.run(main())
"""

SUPERVISE: Final = """
import sys
from pathlib import Path

from litellm.proxy.rust_forward.server import Listener, PythonServer, Server
from litellm.proxy.rust_forward.supervisor import supervise

workers, port, scripts = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
server = PythonServer(Server.UVICORN, Listener.TCP, "127.0.0.1", port, 5, 30.0, "127.0.0.1")
result = supervise(server, (sys.executable, str(scripts / "serve.py"), workers, str(port)), scripts_dir=str(scripts))
raise SystemExit(result if isinstance(result, int) else 3)
"""

T = TypeVar("T")
EXIT_SECONDS: Final = 15
TIME_TOLERANCE_SECONDS: Final = 5


class Scenario(Enum):
    TERM = "SIGTERM"
    INT = "SIGINT"
    INT_TWICE_WHILE_STREAMING = "SIGINT twice during an open stream"
    HUP = "SIGHUP"


@dataclass(frozen=True, slots=True)
class Outcome:
    status: int | None
    survived_hup: bool


@dataclass(frozen=True, slots=True)
class Run:
    outcome: Outcome
    seconds: float


def _scripts(directory: Path) -> Path:
    (directory / "app.py").write_text(APP)
    (directory / "serve.py").write_text(SERVE)
    (directory / "supervise.py").write_text(SUPERVISE)
    gateway: Final = directory / "litellm-gateway"
    gateway.write_text(f"#!{sys.executable}\n{GATEWAY}")
    gateway.chmod(0o755)
    return directory


def _serving(port: int) -> bool:
    try:
        return httpx.get(f"http://127.0.0.1:{port}/", timeout=1, trust_env=False).text == "ok"
    except httpx.TransportError:
        return False


def _exit_status(process: subprocess.Popen[bytes], seconds: float) -> int | None:
    try:
        return process.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        return None


def _streaming(port: int) -> socket.socket:
    stream: Final = socket.create_connection(("127.0.0.1", port), timeout=10)
    stream.sendall(b"GET /stream HTTP/1.1\r\nHost: test\r\n\r\n")
    assert eventually(lambda: stream.recv(4096), lambda received: b"tick" in received, seconds=10)
    return stream


def _signal_after_ready(process: subprocess.Popen[bytes], port: int, scenario: Scenario) -> Run:
    assert eventually(lambda: _serving(port), bool, seconds=120)
    started: Final = time.monotonic()
    match scenario:
        case Scenario.TERM | Scenario.INT:
            send_signal(process, signal.SIGTERM if scenario is Scenario.TERM else signal.SIGINT)
            return Run(Outcome(_exit_status(process, EXIT_SECONDS), False), time.monotonic() - started)
        case Scenario.INT_TWICE_WHILE_STREAMING:
            with _streaming(port):
                send_signal(process, signal.SIGINT)
                time.sleep(0.5)
                send_signal(process, signal.SIGINT)
                status: Final = _exit_status(process, EXIT_SECONDS)
            return Run(Outcome(status, False), time.monotonic() - started)
        case Scenario.HUP:
            send_signal(process, signal.SIGHUP)
            after_hup: Final = _exit_status(process, 3)
            if after_hup is not None:
                return Run(Outcome(after_hup, False), time.monotonic() - started)
            assert eventually(lambda: _serving(port), bool, seconds=30)
            send_signal(process, signal.SIGTERM)
            return Run(Outcome(_exit_status(process, EXIT_SECONDS), True), time.monotonic() - started)


def _run(argv: tuple[str, ...], environment: dict[str, str], act: Callable[[subprocess.Popen[bytes]], T]) -> T:
    with subprocess.Popen(argv, env=environment, start_new_session=True) as process:
        try:
            return act(process)
        finally:
            if process.poll() is None:
                for leftover in (*psutil.Process(process.pid).children(recursive=True), psutil.Process(process.pid)):
                    kill_stray_process(leftover.pid)


def _environment(markers: Path) -> dict[str, str]:
    return {**os.environ, "MARKERS": str(markers), "PYTHONUNBUFFERED": "1"}


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("scenario", list(Scenario), ids=[scenario.name for scenario in Scenario])
def test_the_supervisor_ends_as_python_alone_would(tmp_path: Path, workers: int, scenario: Scenario) -> None:
    scripts: Final = _scripts(tmp_path)
    alone_port: Final = allocate_port()
    alone: Final = _run(
        (sys.executable, str(scripts / "serve.py"), str(workers), str(alone_port)),
        _environment(tmp_path),
        lambda process: _signal_after_ready(process, alone_port, scenario),
    )
    supervised_port: Final = allocate_port()
    supervised: Final = _run(
        (sys.executable, str(scripts / "supervise.py"), str(workers), str(supervised_port), str(scripts)),
        _environment(tmp_path),
        lambda process: _signal_after_ready(process, supervised_port, scenario),
    )
    assert supervised.outcome == alone.outcome
    assert abs(supervised.seconds - alone.seconds) < TIME_TOLERANCE_SECONDS, (alone, supervised)


def _alive(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


@pytest.mark.parametrize("workers", [1, 2])
def test_children_exit_when_the_supervisor_is_killed(tmp_path: Path, workers: int) -> None:
    scripts: Final = _scripts(tmp_path)
    port: Final = allocate_port()

    def survivors_of_killing_the_supervisor(process: subprocess.Popen[bytes]) -> tuple[int, ...]:
        assert eventually(lambda: _serving(port), bool, seconds=120)
        python: Final = int((tmp_path / "python.pid").read_text())
        rust: Final = int((tmp_path / "rust.pid").read_text())
        descendants: Final = (python, rust, *(child.pid for child in psutil.Process(python).children(recursive=True)))
        kill_stray_process(process.pid)
        return eventually(lambda: tuple(pid for pid in descendants if _alive(pid)), lambda alive: not alive)

    survivors: Final = _run(
        (sys.executable, str(scripts / "supervise.py"), str(workers), str(port), str(scripts)),
        _environment(tmp_path),
        survivors_of_killing_the_supervisor,
    )
    assert survivors == ()
