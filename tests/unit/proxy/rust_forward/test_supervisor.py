import functools
import signal
import socket
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

import pytest
from hypothesis import given
from hypothesis import strategies as st

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
    FAILURE,
    Action,
    Caught,
    Child,
    ChildExited,
    Event,
    Lifecycle,
    PythonListening,
    Send,
    Spawn,
    Spawned,
    StartupError,
    boot,
    exit_status,
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
    assert refused == StartupError(reason)
    assert (capsys.readouterr().err, prepared) == (
        f"{reason}, so Python serves the public port without Rust forwarding\n",
        [],
    )


@pytest.mark.parametrize("server", [Server.UVICORN, Server.GUNICORN])
def test_plain_tcp_uvicorn_and_gunicorn_can_be_forwarded(server: Server) -> None:
    assert startup_check(POSIX, _server(server), "gateway") == "gateway"


def test_windows_forwards_uvicorn_and_refuses_gunicorn() -> None:
    assert startup_check(WINDOWS, _server(), "gateway") == "gateway"
    assert startup_check(WINDOWS, _server(Server.GUNICORN), "gateway") == StartupError(
        "LITELLM_RUST does not support gunicorn on Windows"
    )


def test_missing_binary_is_refused_before_anything_starts(tmp_path: Path) -> None:
    prepared: Final[list[bool]] = []
    refused: Final = supervise(
        _server(), ("python",), before_start=lambda: prepared.append(True), scripts_dir=str(tmp_path), platform=POSIX
    )
    assert (refused, prepared) == (
        StartupError("LITELLM_RUST requires the litellm-gateway binary from the litellm-proxy package"),
        [],
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


@dataclass(frozen=True, slots=True)
class World:
    """What the operating system knows about the children, kept independently of ``Lifecycle``."""

    alive: frozenset[Child] = frozenset({Child.PYTHON})
    spawned: frozenset[Child] = frozenset({Child.PYTHON})
    reported_listening: bool = False
    stop_requested: bool = False
    python_status: int | None = None
    rust_lost: bool = False


def _expected_exit(world: World) -> int | None:
    if world.alive or world.python_status is None:
        return None
    return FAILURE if world.rust_lost else world.python_status


def _expected_relay(world: World, signum: signal.Signals) -> tuple[Action, ...]:
    to_rust: Final = (Send(Child.RUST, signum),) if signum in STOPS and Child.RUST in world.alive else ()
    to_python: Final = (Send(Child.PYTHON, signum),) if Child.PYTHON in world.alive else ()
    return (*to_rust, *to_python)


def _expected_on_exit(world: World, child: Child) -> tuple[Action, ...]:
    match child:
        case Child.PYTHON:
            return (Send(Child.RUST, TERM),) if Child.RUST in world.alive else ()
        case Child.RUST:
            serving: Final = Child.PYTHON in world.alive and not world.stop_requested
            return (Send(Child.PYTHON, TERM),) if serving else ()


def _after(world: World, event: Event) -> World:
    match event:
        case Caught(signum):
            return replace(world, stop_requested=world.stop_requested or signum in STOPS)
        case PythonListening():
            return replace(world, reported_listening=world.reported_listening or Child.PYTHON in world.alive)
        case ChildExited(child, status) if child in world.alive:
            lost: Final = child is Child.RUST and Child.PYTHON in world.alive and not world.stop_requested
            return replace(
                world,
                alive=world.alive - {child},
                python_status=status if child is Child.PYTHON else world.python_status,
                rust_lost=world.rust_lost or lost,
            )
        case ChildExited():
            return world


def _spawned(world: World, actions: tuple[Action, ...]) -> World:
    started: Final = frozenset(action.child for action in actions if isinstance(action, Spawn))
    return replace(world, alive=world.alive | started, spawned=world.spawned | started)


SIGNALS: Final = POSIX.relayed_signals
STOPS: Final = POSIX.stop_signals
EVENTS: Final = st.lists(
    st.one_of(
        st.sampled_from(SIGNALS).map(Caught),
        st.just(PythonListening()),
        st.builds(ChildExited, st.sampled_from(Child), st.integers(-15, 3)),
    ),
    max_size=40,
)


@given(EVENTS)
def test_lifecycle_relays_every_signal_and_ends_with_pythons_status(events: list[Event]) -> None:
    state, actions = boot()
    world = World()
    assert actions == (Spawn(Child.PYTHON),)
    for event in events:
        before = _after(world, event)
        state, actions = step(state, event, STOPS)
        match event:
            case Caught(signum):
                assert actions == _expected_relay(before, signum)
            case PythonListening():
                starts_rust = Child.PYTHON in world.alive and not world.reported_listening and not world.stop_requested
                assert actions == ((Spawn(Child.RUST),) if starts_rust else ())
            case ChildExited(child) if child in world.alive:
                assert actions == _expected_on_exit(world, child)
            case ChildExited():
                assert actions == ()
        assert all(action.child in before.alive for action in actions if isinstance(action, Send))
        assert all(action.child not in before.spawned for action in actions if isinstance(action, Spawn))
        world = _spawned(before, actions)
        assert exit_status(state) == _expected_exit(world)


@given(EVENTS)
def test_rust_starts_at_most_once_and_only_while_python_listens_unstopped(events: list[Event]) -> None:
    def advance(acc: tuple[Lifecycle, tuple[tuple[Event, Lifecycle, Action], ...]], event: Event):
        state, seen = acc
        following, actions = step(state, event, STOPS)
        return following, (*seen, *((event, state, action) for action in actions))

    _, history = functools.reduce(advance, events, (boot()[0], ()))
    spawns: Final = tuple((event, state) for event, state, action in history if isinstance(action, Spawn))
    assert len(spawns) <= 1
    assert all(
        isinstance(event, PythonListening) and isinstance(state.python, Spawned) and not state.stop_requested
        for event, state in spawns
    )


@pytest.mark.parametrize(
    ("events", "expected"),
    [
        ((PythonListening(), Caught(TERM), ChildExited(Child.RUST, 0), ChildExited(Child.PYTHON, -15)), -15),
        ((PythonListening(), Caught(INT), Caught(INT), ChildExited(Child.PYTHON, 1), ChildExited(Child.RUST, 0)), 1),
        ((PythonListening(), Caught(signal.SIGHUP), ChildExited(Child.PYTHON, -1), ChildExited(Child.RUST, 0)), -1),
        ((PythonListening(), ChildExited(Child.RUST, 0), ChildExited(Child.PYTHON, 0)), FAILURE),
        ((ChildExited(Child.PYTHON, 3),), 3),
    ],
)
def test_supervisor_exits_as_python_did_unless_rust_was_lost(events: tuple[Event, ...], expected: int) -> None:
    final: Final = functools.reduce(lambda state, event: step(state, event, STOPS)[0], events, boot()[0])
    assert exit_status(final) == expected
