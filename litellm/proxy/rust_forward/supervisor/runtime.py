import asyncio
import contextlib
import os
import signal
from collections.abc import AsyncGenerator, Callable, Coroutine, Mapping, Sequence
from functools import partial
from types import MappingProxyType
from typing import Final, TypeAlias

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

Launch: TypeAlias = Callable[[asyncio.Queue[Event]], contextlib.AbstractAsyncContextManager[asyncio.subprocess.Process]]


async def run(platform: Platform, binary: str, command: Sequence[str], server: PythonServer) -> int:
    """Run Python and Rust under ``lifecycle.step`` until it decides the exit status."""
    events: Final[asyncio.Queue[Event]] = asyncio.Queue()
    platform.catch(asyncio.get_running_loop(), lambda signum: events.put_nowait(Caught(signum)))
    with platform.private_listener() as private:
        launches: Final[Mapping[Child, Launch]] = MappingProxyType(
            {
                Child.PYTHON: partial(
                    _launch_python, platform, python_spec(platform, command, private, server), private.port
                ),
                Child.RUST: partial(_launch, platform, Child.RUST, rust_spec(platform, binary, private.port, server)),
            }
        )
        return await _drive(platform, events, launches)


def exit_like(status: int) -> int:
    """A child killed by a signal kills the supervisor with the same signal, as uvicorn re-raises the one it caught."""
    if status >= 0:
        return status
    signum: Final = signal.Signals(-status)
    signal.signal(signum, signal.SIG_DFL)
    signal.raise_signal(signum)
    return 128 + signum


async def _drive(platform: Platform, events: asyncio.Queue[Event], launches: Mapping[Child, Launch]) -> int:
    async with contextlib.AsyncExitStack() as stack:
        children: Final[dict[Child, asyncio.subprocess.Process]] = {}  # mutable-ok: indexes the stack's children
        state, actions = boot()
        while True:
            for action in actions:
                match action:
                    case Spawn(child):
                        children[child] = await stack.enter_async_context(launches[child](events))
                    case Send(child, signum):
                        platform.signal_child(children[child], signum)
                    case _:
                        assert_never(action)
            if (status := exit_status(state)) is not None:
                return status
            state, actions = step(state, await events.get(), platform.stop_signals)


# A child's stdin is a pipe only the supervisor holds open, so the child sees EOF as soon as the supervisor is gone,
# however it went, and shuts down instead of outliving it.
@contextlib.asynccontextmanager
async def _launch(
    platform: Platform,
    child: Child,
    spec: ChildSpec,
    events: asyncio.Queue[Event],
) -> AsyncGenerator[asyncio.subprocess.Process]:
    process: Final = await asyncio.create_subprocess_exec(
        *spec.argv,
        stdin=asyncio.subprocess.PIPE,
        env={**os.environ, **spec.env_overrides},
        pass_fds=spec.spawn.pass_fds,
        start_new_session=spec.spawn.start_new_session,
        creationflags=spec.spawn.creationflags,
    )
    waiter: Final = asyncio.create_task(_report_exit(child, process, events))
    try:
        yield process
    finally:
        platform.stop_group(process)
        await waiter
        if process.stdin is not None:
            process.stdin.close()


@contextlib.asynccontextmanager
async def _launch_python(
    platform: Platform, spec: ChildSpec, port: int, events: asyncio.Queue[Event]
) -> AsyncGenerator[asyncio.subprocess.Process]:
    async with (
        _launch(platform, Child.PYTHON, spec, events) as process,
        _in_background(_report_once_listening(port, events)),
    ):
        yield process


async def _report_exit(child: Child, process: asyncio.subprocess.Process, events: asyncio.Queue[Event]) -> None:
    events.put_nowait(ChildExited(child, await process.wait()))


async def _report_once_listening(port: int, events: asyncio.Queue[Event]) -> None:
    while not await _accepts(port):
        await asyncio.sleep(PROBE_INTERVAL_SECONDS)
    events.put_nowait(Probe.PYTHON_LISTENING)


async def _accepts(port: int) -> bool:
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection(RUST_HOP, port), PROBE_INTERVAL_SECONDS)
    except (OSError, asyncio.TimeoutError):
        return False
    writer.close()
    return True


@contextlib.asynccontextmanager
async def _in_background(work: Coroutine[None, None, None]) -> AsyncGenerator[None]:
    task: Final = asyncio.create_task(work)
    try:
        yield
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
