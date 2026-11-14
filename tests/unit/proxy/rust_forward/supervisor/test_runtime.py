import signal
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Final

import anyio
import pytest

from litellm.proxy.rust_forward.platform import ChildProcess, Platform, current_platform
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
from litellm.proxy.rust_forward.supervisor.runtime import SpawnFailed, drive, exit_like, open_children, spawn_subprocess
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


@dataclass(slots=True)
class HeldChild:
    exits: anyio.Event
    waiting: int = 0
    stdin: None = None

    @property
    def pid(self) -> int:
        raise NotImplementedError

    def send_signal(self, signum: int, /) -> None:
        raise NotImplementedError

    async def wait(self) -> int:
        self.waiting += 1
        try:
            await self.exits.wait()
        finally:
            self.waiting -= 1
        return 0


@dataclass(frozen=True, slots=True)
class StopRecorder:
    stopped: list[ChildProcess] = field(default_factory=list)

    def signal_child(self, process: ChildProcess, signum: signal.Signals) -> None:
        raise NotImplementedError

    def stop_group(self, process: ChildProcess) -> None:
        self.stopped.append(process)


async def test_a_cancelled_supervisor_still_waits_for_the_child_it_stops() -> None:
    child: Final = HeldChild(anyio.Event())
    control: Final = StopRecorder()
    supervisor: Final = anyio.CancelScope()
    returned: Final = anyio.Event()

    async def spawn(_spec: ChildSpec) -> HeldChild:
        return child

    async def supervise_one_child() -> None:
        with supervisor:
            async with open_children(control, spawn, _children(("held",)), 1) as children:
                assert await children.spawn(Child.RUST) is None
                await anyio.sleep_forever()
        returned.set()

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(supervise_one_child)
        await anyio.wait_all_tasks_blocked()
        supervisor.cancel()
        await anyio.wait_all_tasks_blocked()
        while_child_runs: Final = (tuple(control.stopped), child.waiting, returned.is_set())
        child.exits.set()
        await anyio.wait_all_tasks_blocked()
        once_child_exits: Final = returned.is_set()
    assert (while_child_runs, once_child_exits) == (((child,), 1, False), True)


async def test_exiting_children_are_reported_as_events() -> None:
    async with open_children(
        current_platform(), spawn_subprocess, _children((sys.executable, "-c", "exit(7)")), 1
    ) as children:
        assert await children.spawn(Child.RUST) is None
        with anyio.fail_after(5):
            assert await children.next_event() == ChildExited(Child.RUST, 7)


async def test_a_child_that_cannot_start_is_returned_as_a_failure(tmp_path: Path) -> None:
    async with open_children(
        current_platform(), spawn_subprocess, _children((str(tmp_path / "missing"),)), 1
    ) as children:
        outcome: Final = await children.spawn(Child.RUST)
    assert isinstance(outcome, SpawnFailed)
    assert isinstance(outcome.error, FileNotFoundError)


def _dropped(_signum: signal.Signals) -> None:
    pass


@POSIX_ONLY
@pytest.mark.parametrize(
    ("status", "exits"),
    [
        pytest.param(3, 3, id="an exit status is passed on"),
        pytest.param(-signal.SIGTERM, 0, id="a stop signal ends it the way uvicorn alone ends as PID 1"),
        pytest.param(-signal.SIGKILL, 137, id="a kill is reported the way a shell reports it"),
    ],
)
def test_a_supervisor_the_kernel_will_not_signal_still_exits_as_python_alone_would(status: int, exits: int) -> None:
    assert exit_like(status, STOPS, die_by=_dropped) == exits


@POSIX_ONLY
@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGKILL])
def test_the_supervisor_dies_by_the_signal_that_killed_python(signum: signal.Signals) -> None:
    script: Final = (
        "from litellm.proxy.rust_forward.supervisor.runtime import exit_like\n"
        f"raise SystemExit(exit_like({-signum}, frozenset()))"
    )
    assert subprocess.run((sys.executable, "-I", "-c", script), check=False, timeout=60).returncode == -signum
