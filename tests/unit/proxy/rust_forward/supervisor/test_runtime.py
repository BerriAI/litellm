import signal
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Final

import anyio
import pytest

from litellm.proxy.rust_forward.platform import Platform, current_platform
from litellm.proxy.rust_forward.platform.windows import WindowsPlatform
from litellm.proxy.rust_forward.supervisor import supervise
from litellm.proxy.rust_forward.supervisor.children import ChildSpec
from litellm.proxy.rust_forward.supervisor.lifecycle import (
    Action,
    Caught,
    Child,
    ChildExited,
    Event,
    Probe,
    Send,
    Spawn,
)
from litellm.proxy.rust_forward.supervisor.runtime import SpawnFailed, drive, open_children
from tests.unit.proxy.rust_forward.supervisor._support import python_server


PYTHON_CHILD: Final = """
import json, os, pathlib, socket, time
time.sleep(0.5)
pathlib.Path(os.environ["MARKERS"], "listening").touch()
private = json.loads(os.environ["_LITELLM_GATEWAY_SUPERVISED"])["listener"]
listener = socket.socket(fileno=private["fd"]) if "fd" in private else socket.create_server(("127.0.0.1", private["port"]))
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
    supervise(python_server(), command, scripts_dir=str(tmp_path), platform=_fake_gateway(tmp_path))
    assert (tmp_path / "rust_saw_python_listening").read_text() == "True"


TERM: Final = signal.SIGTERM
POSIX_ONLY: Final = pytest.mark.skipif(sys.platform == "win32", reason="the child traps SIGTERM")
STOPS: Final = frozenset({signal.SIGTERM, signal.SIGINT, signal.SIGQUIT})
RUST_MISSING: Final = SpawnFailed(FileNotFoundError("litellm-gateway"))


@dataclass
class ScriptedExecutor:
    """Replays a fixed event script and records every action the driver carries out."""

    events: list[Event]
    failing: frozenset[Child]
    done: list[Action] = field(default_factory=list)

    async def spawn(self, child: Child) -> SpawnFailed | None:
        self.done.append(Spawn(child))
        return RUST_MISSING if child in self.failing else None

    def send(self, child: Child, signum: signal.Signals) -> None:
        self.done.append(Send(child, signum))

    async def next_event(self) -> Event:
        return self.events.pop(0)


@pytest.mark.parametrize(
    ("events", "failing", "done", "outcome", "unread"),
    [
        pytest.param(
            [Probe.PYTHON_LISTENING, Caught(TERM), ChildExited(Child.RUST, 0), ChildExited(Child.PYTHON, -15)],
            frozenset(),
            [Spawn(Child.PYTHON), Spawn(Child.RUST), Send(Child.RUST, TERM), Send(Child.PYTHON, TERM)],
            -15,
            [],
            id="carries out each action in order and ends with Python's status",
        ),
        pytest.param(
            [ChildExited(Child.PYTHON, 3), Caught(TERM)],
            frozenset(),
            [Spawn(Child.PYTHON)],
            3,
            [Caught(TERM)],
            id="reads no event once the status is settled",
        ),
        pytest.param(
            [Probe.PYTHON_LISTENING, Caught(TERM)],
            frozenset({Child.RUST}),
            [Spawn(Child.PYTHON), Spawn(Child.RUST)],
            RUST_MISSING,
            [Caught(TERM)],
            id="returns a failed spawn without waiting for more events",
        ),
    ],
)
async def test_driver_executes_lifecycle_decisions(
    events: list[Event], failing: frozenset[Child], done: list[Action], outcome: int | SpawnFailed, unread: list[Event]
) -> None:
    executor: Final = ScriptedExecutor(events, failing)
    assert await drive(executor, STOPS) == outcome
    assert (executor.done, executor.events) == (done, unread)


def _children(argv: tuple[str, ...]) -> MappingProxyType[Child, ChildSpec]:
    spec: Final = ChildSpec(argv, MappingProxyType({}), current_platform().spawn_options(()))
    return MappingProxyType({Child.PYTHON: spec, Child.RUST: spec})


SLOW_TO_STOP: Final = (
    sys.executable,
    "-c",
    "import signal, time; signal.signal(signal.SIGTERM, lambda *_: (time.sleep(0.5), exit(0))); time.sleep(30)",
)


@POSIX_ONLY
async def test_a_cancelled_supervisor_still_waits_for_the_child_it_stops() -> None:
    with anyio.CancelScope() as scope:
        async with open_children(current_platform(), _children(SLOW_TO_STOP), 1) as children:
            assert await children.spawn(Child.RUST) is None
            process: Final = children.processes[Child.RUST]
            await anyio.sleep(0.3)
            scope.cancel()
            await anyio.sleep(1)
    assert process.returncode == 0


async def test_exiting_children_are_reported_as_events() -> None:
    async with open_children(current_platform(), _children((sys.executable, "-c", "exit(7)")), 1) as children:
        assert await children.spawn(Child.RUST) is None
        with anyio.fail_after(5):
            assert await children.next_event() == ChildExited(Child.RUST, 7)


async def test_a_child_that_cannot_start_is_returned_as_a_failure(tmp_path: Path) -> None:
    async with open_children(current_platform(), _children((str(tmp_path / "missing"),)), 1) as children:
        outcome: Final = await children.spawn(Child.RUST)
    assert isinstance(outcome, SpawnFailed)
    assert isinstance(outcome.error, FileNotFoundError)
