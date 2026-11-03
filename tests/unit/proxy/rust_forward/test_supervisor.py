import functools
import signal
import socket
import subprocess
import sys
from pathlib import Path
from typing import Final

import pytest

from litellm.constants import GRACEFUL_SHUTDOWN_TEARDOWN_HEADROOM_SECONDS
from litellm.proxy.proxy_cli import run_server
from litellm.proxy.rust_forward.mode import AssignedPort, InheritedFd, Supervised, rust_forward_mode
from litellm.proxy.rust_forward.platform import Platform, PrivateListener, current_platform
from litellm.proxy.rust_forward.platform.posix import PosixPlatform
from litellm.proxy.rust_forward.platform.windows import (
    CREATE_NEW_PROCESS_GROUP,
    CTRL_BREAK_EVENT,
    SIGBREAK,
    WindowsPlatform,
)
from litellm.proxy.rust_forward.server import Listener, PythonServer, Server
from litellm.proxy.rust_forward.supervisor import (
    Child,
    Event,
    ExitCode,
    Exited,
    Finished,
    Ready,
    Running,
    Signal,
    Start,
    StartupError,
    Stop,
    Stopping,
    Transition,
    boot,
    gateway_binary,
    python_spec,
    rust_spec,
    startup_check,
    step,
    supervise,
)

TERM: Final = signal.SIGTERM
INT: Final = signal.SIGINT
POSIX: Final = PosixPlatform()
WINDOWS: Final = WindowsPlatform()
POSIX_ONLY: Final = pytest.mark.skipif(sys.platform == "win32", reason="runs a POSIX shebang gateway")


def _server(
    server: Server = Server.UVICORN, listener: Listener = Listener.TCP, trusted_proxies: str | None = None
) -> PythonServer:
    return PythonServer(server, listener, "0.0.0.0", 4000, 2, 2.5, False, trusted_proxies)


def test_python_child_serves_the_private_socket_with_the_operators_trust() -> None:
    private: Final = PrivateListener(InheritedFd(17), 45555, (17,))
    spec: Final = python_spec(POSIX, ("python",), private, _server(trusted_proxies="10.0.0.5"))
    assert rust_forward_mode(spec.env_overrides) == Supervised(InheritedFd(17), False, "10.0.0.5")
    assert (spec.spawn.pass_fds, spec.spawn.start_new_session) == ((17,), True)


def test_windows_python_child_binds_the_assigned_port_in_its_own_process_group() -> None:
    with WINDOWS.private_listener() as private:
        pass
    spec: Final = python_spec(WINDOWS, ("python",), private, _server())
    assert rust_forward_mode(spec.env_overrides) == Supervised(AssignedPort(private.port), False, "127.0.0.1")
    assert spec.argv[-2:] == ("--port", str(private.port))
    assert (spec.spawn.pass_fds, spec.spawn.creationflags) == ((), CREATE_NEW_PROCESS_GROUP)
    with socket.create_server(("127.0.0.1", private.port)):
        pass


@pytest.mark.skipif(sys.platform != "win32", reason="these names exist only on Windows")
def test_windows_constants_match_the_stdlib() -> None:
    assert (CTRL_BREAK_EVENT, SIGBREAK) == (signal.CTRL_BREAK_EVENT, signal.SIGBREAK)
    assert CREATE_NEW_PROCESS_GROUP == subprocess.CREATE_NEW_PROCESS_GROUP


def test_python_child_keeps_the_public_host_and_its_options() -> None:
    arguments: Final = ("--host", "0.0.0.0", "--port", "4000", "--num_workers", "2", "--reload")
    spec: Final = python_spec(POSIX, arguments, PrivateListener(InheritedFd(17), 45555, (17,)), _server())
    params: Final = run_server.make_context("litellm", list(spec.argv)).params
    assert (params["host"], params["port"]) == ("0.0.0.0", 45555)
    assert (params["num_workers"], params["reload"]) == (2, True)


def test_rust_child_serves_public_port_and_forwards_to_private_listener() -> None:
    spec: Final = rust_spec(POSIX, "gateway", 45555, _server())
    assert spec.argv == ("gateway",)
    assert spec.spawn.pass_fds == ()
    assert dict(spec.env_overrides) == {
        "LITELLM_FORWARD_UPSTREAM": "http://127.0.0.1:45555",
        "LITELLM_FORWARD_HOST": "0.0.0.0",
        "LITELLM_FORWARD_PORT": "4000",
        "LITELLM_FORWARD_UPSTREAM_KEEPALIVE": "2",
        "LITELLM_FORWARD_SHUTDOWN_DRAIN": str(2.5 + GRACEFUL_SHUTDOWN_TEARDOWN_HEADROOM_SECONDS),
    }


@pytest.mark.parametrize(
    ("server", "reason"),
    [
        (_server(Server.HYPERCORN), "LITELLM_RUST does not support Hypercorn"),
        (_server(Server.GRANIAN), "LITELLM_RUST does not support Granian"),
        (_server(listener=Listener.TLS), "LITELLM_RUST does not support TLS on the public port"),
        (_server(listener=Listener.UNIX_SOCKET), "LITELLM_RUST does not support a Unix socket on the public port"),
        (
            _server(Server.GUNICORN, Listener.INHERITED_FD),
            "LITELLM_RUST does not support an inherited file descriptor on the public port",
        ),
    ],
)
def test_unsupported_servers_are_refused_before_anything_starts(
    server: PythonServer, reason: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert startup_check(POSIX, server, "gateway") == StartupError(reason)
    prepared: Final[list[bool]] = []
    refused: Final = supervise(server, ("python",), before_start=lambda: prepared.append(True), platform=POSIX)
    assert refused == ExitCode.CHILD_FAILED
    assert (capsys.readouterr().err, prepared) == (f"{reason}\n", [])


@pytest.mark.parametrize("server", [Server.UVICORN, Server.GUNICORN])
def test_plain_tcp_uvicorn_and_gunicorn_can_be_forwarded(server: Server) -> None:
    assert startup_check(POSIX, _server(server), "gateway") == "gateway"


def test_windows_forwards_uvicorn_and_refuses_gunicorn() -> None:
    assert startup_check(WINDOWS, _server(), "gateway") == "gateway"
    assert startup_check(WINDOWS, _server(Server.GUNICORN), "gateway") == StartupError(
        "LITELLM_RUST does not support gunicorn on Windows"
    )


def test_missing_binary_gives_install_instruction() -> None:
    assert startup_check(POSIX, _server(), None) == StartupError(
        "LITELLM_RUST requires the litellm-gateway binary. Install litellm[proxy] in this environment"
    )


@POSIX_ONLY
def test_binary_is_found_only_in_the_environment_scripts_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scripts: Final = tmp_path / "scripts"
    elsewhere: Final = tmp_path / "elsewhere"
    scripts.mkdir()
    elsewhere.mkdir()
    binary: Final = elsewhere / "litellm-gateway"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", str(elsewhere))
    assert gateway_binary(POSIX, str(scripts)) is None
    assert gateway_binary(POSIX, str(elsewhere)) == str(binary)


PYTHON_CHILD: Final = """
import os, pathlib, socket, time
time.sleep(0.5)
pathlib.Path(os.environ["MARKERS"], "listening").touch()
fd = os.environ.get("_LITELLM_GATEWAY_FD")
port = os.environ.get("_LITELLM_GATEWAY_PORT")
listener = socket.socket(fileno=int(fd)) if fd else socket.create_server(("127.0.0.1", int(port)))
listener.listen()
time.sleep(10)
"""
RUST_CHILD: Final = """
import os, pathlib
markers = pathlib.Path(os.environ["MARKERS"])
(markers / "rust_saw_python_listening").write_text(str((markers / "listening").exists()))
"""


def _fake_gateway(scripts: Path) -> Platform:
    if sys.platform != "win32":
        gateway: Final = scripts / "litellm-gateway"
        gateway.write_text(f"#!{sys.executable} -I\n{RUST_CHILD}")
        gateway.chmod(0o755)
        return current_platform()
    script: Final = scripts / "rust_child.py"
    script.write_text(RUST_CHILD)
    (scripts / "litellm-gateway.cmd").write_text(f'@"{sys.executable}" -I "{script}"\r\n')
    return WindowsPlatform(gateway_name="litellm-gateway.cmd")


def test_rust_opens_the_public_port_only_once_python_listens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MARKERS", str(tmp_path))
    command: Final = (sys.executable, "-I", "-c", PYTHON_CHILD)
    supervise(_server(), command, scripts_dir=str(tmp_path), platform=_fake_gateway(tmp_path))
    assert (tmp_path / "rust_saw_python_listening").read_text() == "True"


def _fold(events: tuple[Event, ...]) -> Transition:
    def advance(acc: Transition, event: Event) -> Transition:
        state, done = acc
        following, actions = step(state, event)
        return following, (*done, *actions)

    return functools.reduce(advance, events, boot())


RUST: Final = Child.RUST
PYTHON: Final = Child.PYTHON
PYTHON_READY: Final = Ready(PYTHON)
STARTED: Final = (Start(PYTHON), Start(RUST))
BOTH_TERM: Final = (*STARTED, Signal(RUST, TERM), Signal(PYTHON, TERM))
CLEAN: Final = Finished(ExitCode.CLEAN)
FAILED: Final = Finished(ExitCode.CHILD_FAILED)


@pytest.mark.parametrize(
    ("events", "expected"),
    [
        ((), (Running((PYTHON,)), (Start(PYTHON),))),
        ((Ready(RUST),), (Running((PYTHON,)), (Start(PYTHON),))),
        ((PYTHON_READY,), (Running((PYTHON, RUST)), STARTED)),
        ((PYTHON_READY, PYTHON_READY, Ready(RUST)), (Running((PYTHON, RUST)), STARTED)),
        ((PYTHON_READY, Stop(TERM)), (Stopping(frozenset({RUST, PYTHON})), BOTH_TERM)),
        ((PYTHON_READY, Stop(TERM), Exited(RUST, 0), Exited(PYTHON, 0)), (CLEAN, BOTH_TERM)),
        ((PYTHON_READY, Stop(TERM), Exited(PYTHON, 0), Exited(RUST, 0)), (CLEAN, BOTH_TERM)),
        (
            (PYTHON_READY, Stop(INT), Stop(INT), Exited(RUST, 0), Exited(PYTHON, 0)),
            (CLEAN, (*STARTED, Signal(RUST, INT), Signal(PYTHON, INT))),
        ),
        ((PYTHON_READY, Stop(TERM), Exited(RUST, 0), Exited(RUST, 0)), (FAILED, BOTH_TERM)),
        ((PYTHON_READY, Stop(TERM), Exited(RUST, 1)), (FAILED, BOTH_TERM)),
        ((PYTHON_READY, Stop(TERM), Exited(RUST, 0), Exited(PYTHON, 1)), (FAILED, BOTH_TERM)),
        ((PYTHON_READY, Exited(PYTHON, 0)), (FAILED, STARTED)),
        ((PYTHON_READY, Exited(RUST, 1)), (FAILED, STARTED)),
        ((PYTHON_READY, Exited(RUST, 1), Stop(TERM), PYTHON_READY, Exited(PYTHON, 0)), (FAILED, STARTED)),
        ((Exited(PYTHON, 1),), (FAILED, (Start(PYTHON),))),
        ((Stop(TERM), PYTHON_READY, Exited(PYTHON, 0)), (CLEAN, (Start(PYTHON), Signal(PYTHON, TERM)))),
        ((Stop(TERM), Exited(PYTHON, 1)), (FAILED, (Start(PYTHON), Signal(PYTHON, TERM)))),
    ],
)
def test_children_start_in_order_once_ready_and_stop_together_in_reverse(
    events: tuple[Event, ...], expected: Transition
) -> None:
    assert _fold(events) == expected
