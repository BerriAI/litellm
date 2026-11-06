import asyncio
import contextlib
import math
import os
import signal
from collections.abc import AsyncGenerator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Protocol

import anyio
from anyio.abc import TaskGroup
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from typing_extensions import assert_never

from litellm.proxy.rust_forward.mode import RUST_HOP
from litellm.proxy.rust_forward.platform import Platform
from litellm.proxy.rust_forward.server import PythonServer
from litellm.proxy.rust_forward.supervisor.children import ChildSpec, python_spec, rust_spec
from litellm.proxy.rust_forward.supervisor.lifecycle import (
    Caught,
    Child,
    ChildExited,
    Event,
    Probe,
    Send,
    Spawn,
    boot,
    exit_status,
    step,
)

PROBE_INTERVAL_SECONDS: Final = 0.05


@dataclass(frozen=True, slots=True)
class SpawnFailed:
    error: OSError


class Executor(Protocol):
    async def spawn(self, child: Child) -> SpawnFailed | None: ...

    def send(self, child: Child, signum: signal.Signals) -> None: ...

    async def next_event(self) -> Event: ...


async def run(platform: Platform, binary: str, command: Sequence[str], server: PythonServer) -> int | SpawnFailed:
    with platform.private_listener() as private:
        specs: Final = MappingProxyType(
            {
                Child.PYTHON: python_spec(platform, command, private, server),
                Child.RUST: rust_spec(platform, binary, private.port, server),
            }
        )
        async with open_children(platform, specs, private.port) as children:
            platform.catch(asyncio.get_running_loop(), children.caught)
            return await drive(children, platform.stop_signals)


async def drive(executor: Executor, stop_signals: frozenset[signal.Signals]) -> int | SpawnFailed:
    """Carry out what ``lifecycle.step`` decides until it settles the exit status."""
    state, actions = boot()
    while True:
        for action in actions:
            match action:
                case Spawn(child):
                    if (failed := await executor.spawn(child)) is not None:
                        return failed
                case Send(child, signum):
                    executor.send(child, signum)
                case _:
                    assert_never(action)
        if (status := exit_status(state)) is not None:
            return status
        state, actions = step(state, await executor.next_event(), stop_signals)


def exit_like(status: int) -> int:
    """A child killed by a signal kills the supervisor with the same signal, as uvicorn re-raises the one it caught."""
    if status >= 0:
        return status
    signum: Final = signal.Signals(-status)
    signal.signal(signum, signal.SIG_DFL)
    signal.raise_signal(signum)
    return 128 + signum


@dataclass(frozen=True, slots=True)
class Children:
    platform: Platform
    specs: Mapping[Child, ChildSpec]
    private_port: int
    stack: contextlib.AsyncExitStack
    tasks: TaskGroup
    events_in: MemoryObjectSendStream[Event]
    events_out: MemoryObjectReceiveStream[Event]
    processes: dict[Child, asyncio.subprocess.Process]  # mutable-ok: the running children the stack owns

    def caught(self, signum: signal.Signals) -> None:
        with contextlib.suppress(anyio.ClosedResourceError):
            self.events_in.send_nowait(Caught(signum))

    async def spawn(self, child: Child) -> SpawnFailed | None:
        try:
            process: Final = await self.stack.enter_async_context(_process(self.platform, self.specs[child]))
        except OSError as error:
            return SpawnFailed(error)
        self.processes[child] = process
        self.tasks.start_soon(_report_exit, child, process, self.events_in.clone())
        if child is Child.PYTHON:
            probe: Final = self.tasks.start_soon(_report_once_listening, self.private_port, self.events_in.clone())
            self.stack.callback(probe.cancel)
        return None

    def send(self, child: Child, signum: signal.Signals) -> None:
        self.platform.signal_child(self.processes[child], signum)

    async def next_event(self) -> Event:
        return await self.events_out.receive()


# The task group is entered before any child, so on the way out every child is stopped, last spawned first, before
# the task group waits for the tasks reporting their exits. Signal handlers cannot wait for room, hence the
# unbounded event buffer.
@contextlib.asynccontextmanager
async def open_children(
    platform: Platform, specs: Mapping[Child, ChildSpec], private_port: int
) -> AsyncGenerator[Children]:
    events_in, events_out = anyio.create_memory_object_stream[Event](math.inf)
    with events_in, events_out:
        async with contextlib.AsyncExitStack() as stack:
            tasks: Final = await stack.enter_async_context(anyio.create_task_group())
            yield Children(platform, specs, private_port, stack, tasks, events_in, events_out, {})


# A child's stdin is a pipe only the supervisor holds open, so the child sees EOF as soon as the supervisor is gone,
# however it went, and shuts down instead of outliving it. Stopping is shielded so a cancelled supervisor still waits
# for the child it is stopping.
@contextlib.asynccontextmanager
async def _process(platform: Platform, spec: ChildSpec) -> AsyncGenerator[asyncio.subprocess.Process]:
    process: Final = await asyncio.create_subprocess_exec(
        *spec.argv,
        stdin=asyncio.subprocess.PIPE,
        env={**os.environ, **spec.env_overrides},
        pass_fds=spec.spawn.pass_fds,
        start_new_session=spec.spawn.start_new_session,
        creationflags=spec.spawn.creationflags,
    )
    try:
        yield process
    finally:
        with anyio.CancelScope(shield=True):
            platform.stop_group(process)
            await process.wait()
        if process.stdin is not None:
            process.stdin.close()


async def _report_exit(
    child: Child, process: asyncio.subprocess.Process, events: MemoryObjectSendStream[Event]
) -> None:
    with events:
        events.send_nowait(ChildExited(child, await process.wait()))


async def _report_once_listening(port: int, events: MemoryObjectSendStream[Event]) -> None:
    with events:
        while not await _accepts(port):
            await anyio.sleep(PROBE_INTERVAL_SECONDS)
        events.send_nowait(Probe.PYTHON_LISTENING)


async def _accepts(port: int) -> bool:
    with anyio.move_on_after(PROBE_INTERVAL_SECONDS):
        try:
            stream: Final = await anyio.connect_tcp(RUST_HOP, port)
        except OSError:
            return False
        await stream.aclose()
        return True
    return False
