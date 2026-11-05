import asyncio
import contextlib
import os
import shutil
import signal
import sys
import sysconfig
from collections.abc import AsyncGenerator, Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from functools import partial
from types import MappingProxyType
from typing import Final, TypeAlias

from typing_extensions import assert_never

from litellm.constants import GRACEFUL_SHUTDOWN_TEARDOWN_HEADROOM_SECONDS
from litellm.proxy.rust_forward.mode import RUST_HOP, supervised_environment
from litellm.proxy.rust_forward.platform import Platform, PrivateListener, SpawnOptions, current_platform
from litellm.proxy.rust_forward.server import Listener, PythonServer, Server
from litellm.rust_bridge.configuration import ENV_NAME

PROBE_INTERVAL_SECONDS: Final = 0.05


def gateway_binary(platform: Platform, scripts_dir: str) -> str | None:
    return shutil.which(platform.gateway_name, path=scripts_dir)


@dataclass(frozen=True, slots=True)
class StartupError:
    reason: str


def startup_check(platform: Platform, server: PythonServer, binary: str | None) -> str | StartupError:
    if server.server in (Server.HYPERCORN, Server.GRANIAN):
        return StartupError(f"{ENV_NAME} does not support {server.server.value}")
    if server.server not in platform.servers:
        return StartupError(f"{ENV_NAME} does not support {server.server.value} on {platform.name}")
    if server.listener is not Listener.TCP:
        return StartupError(f"{ENV_NAME} does not support {server.listener.value} on the public port")
    if binary is None:
        return StartupError(f"{ENV_NAME} requires the litellm-gateway binary from the litellm-proxy package")
    return binary


class Child(Enum):
    RUST = "rust"
    PYTHON = "python"


FAILURE: Final = 1


@dataclass(frozen=True, slots=True)
class NotStarted:
    pass


@dataclass(frozen=True, slots=True)
class Spawned:
    pass


@dataclass(frozen=True, slots=True)
class Listening:
    pass


@dataclass(frozen=True, slots=True)
class Exited:
    status: int


@dataclass(frozen=True, slots=True)
class Lost:
    """Rust exited while Python was serving and nobody had asked either to stop."""

    status: int


PythonState: TypeAlias = Spawned | Listening | Exited
RustState: TypeAlias = NotStarted | Spawned | Exited | Lost


@dataclass(frozen=True, slots=True)
class Lifecycle:
    python: PythonState
    rust: RustState
    stop_requested: bool


@dataclass(frozen=True, slots=True)
class Caught:
    signum: signal.Signals


@dataclass(frozen=True, slots=True)
class PythonListening:
    pass


@dataclass(frozen=True, slots=True)
class ChildExited:
    child: Child
    status: int


@dataclass(frozen=True, slots=True)
class Spawn:
    child: Child


@dataclass(frozen=True, slots=True)
class Send:
    child: Child
    signum: signal.Signals


Event: TypeAlias = Caught | PythonListening | ChildExited
Action: TypeAlias = Spawn | Send
Transition: TypeAlias = tuple[Lifecycle, tuple[Action, ...]]


def boot() -> Transition:
    return Lifecycle(Spawned(), NotStarted(), stop_requested=False), (Spawn(Child.PYTHON),)


def _relay(state: Lifecycle, signum: signal.Signals, stops: bool) -> Transition:
    to_rust: Final = (Send(Child.RUST, signum),) if stops and isinstance(state.rust, Spawned) else ()
    to_python: Final = (Send(Child.PYTHON, signum),) if not isinstance(state.python, Exited) else ()
    return replace(state, stop_requested=state.stop_requested or stops), (*to_rust, *to_python)


# Python alone takes no connections until it has booted, so Rust only opens the public port once Python listens.
def _python_listening(state: Lifecycle) -> Transition:
    if not isinstance(state.python, Spawned):
        return state, ()
    if state.stop_requested:
        return replace(state, python=Listening()), ()
    return replace(state, python=Listening(), rust=Spawned()), (Spawn(Child.RUST),)


def _python_exited(state: Lifecycle, status: int) -> Transition:
    if isinstance(state.python, Exited):
        return state, ()
    exited: Final = replace(state, python=Exited(status))
    return exited, ((Send(Child.RUST, signal.SIGTERM),) if isinstance(state.rust, Spawned) else ())


def _rust_exited(state: Lifecycle, status: int) -> Transition:
    if not isinstance(state.rust, Spawned):
        return state, ()
    if state.stop_requested or isinstance(state.python, Exited):
        return replace(state, rust=Exited(status)), ()
    return replace(state, rust=Lost(status)), (Send(Child.PYTHON, signal.SIGTERM),)


def step(state: Lifecycle, event: Event, stop_signals: frozenset[signal.Signals]) -> Transition:
    """The supervisor never gives a signal or an exit a meaning of its own. Every signal reaches Python as sent,
    Python's exit status is the supervisor's, and Rust only follows Python, except that losing Rust stops Python."""
    match event:
        case Caught(signum):
            return _relay(state, signum, signum in stop_signals)
        case PythonListening():
            return _python_listening(state)
        case ChildExited(child, status):
            match child:
                case Child.PYTHON:
                    return _python_exited(state, status)
                case Child.RUST:
                    return _rust_exited(state, status)
                case _:
                    assert_never(child)
        case _:
            assert_never(event)


def exit_status(state: Lifecycle) -> int | None:
    match state.python, state.rust:
        case Exited(status), NotStarted() | Exited():
            return status
        case Exited(), Lost():
            return FAILURE
        case _:
            return None


@dataclass(frozen=True, slots=True)
class ChildSpec:
    argv: tuple[str, ...]
    env_overrides: Mapping[str, str]
    spawn: SpawnOptions


def python_spec(
    platform: Platform, command: Sequence[str], private: PrivateListener, server: PythonServer
) -> ChildSpec:
    return ChildSpec(
        argv=(*command, "--port", str(private.port)),
        env_overrides=supervised_environment(private.listener, server.proxy_headers, server.trusted_proxies),
        spawn=platform.spawn_options(private.pass_fds),
    )


def rust_spec(platform: Platform, binary: str, upstream_port: int, server: PythonServer) -> ChildSpec:
    return ChildSpec(
        argv=(binary,),
        env_overrides=MappingProxyType(
            {
                "LITELLM_FORWARD_UPSTREAM": f"http://{RUST_HOP}:{upstream_port}",
                "LITELLM_FORWARD_HOST": server.host,
                "LITELLM_FORWARD_PORT": str(server.port),
                "LITELLM_FORWARD_UPSTREAM_KEEPALIVE": str(server.keepalive_seconds),
                "LITELLM_FORWARD_SHUTDOWN_DRAIN": str(
                    server.graceful_timeout_seconds + GRACEFUL_SHUTDOWN_TEARDOWN_HEADROOM_SECONDS
                ),
            }
        ),
        spawn=platform.spawn_options(()),
    )


async def _spawn(spec: ChildSpec) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        *spec.argv,
        stdin=asyncio.subprocess.PIPE,
        env={**os.environ, **spec.env_overrides},
        pass_fds=spec.spawn.pass_fds,
        start_new_session=spec.spawn.start_new_session,
        creationflags=spec.spawn.creationflags,
    )


Launch: TypeAlias = Callable[[asyncio.Queue[Event]], contextlib.AbstractAsyncContextManager[asyncio.subprocess.Process]]


async def _report_exit(child: Child, process: asyncio.subprocess.Process, events: asyncio.Queue[Event]) -> None:
    events.put_nowait(ChildExited(child, await process.wait()))


async def _accepts(port: int) -> bool:
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection(RUST_HOP, port), PROBE_INTERVAL_SECONDS)
    except (OSError, asyncio.TimeoutError):
        return False
    writer.close()
    return True


async def _report_once_listening(port: int, events: asyncio.Queue[Event]) -> None:
    while not await _accepts(port):
        await asyncio.sleep(PROBE_INTERVAL_SECONDS)
    events.put_nowait(PythonListening())


@contextlib.asynccontextmanager
async def _in_background(work: Coroutine[None, None, None]) -> AsyncGenerator[None]:
    task: Final = asyncio.create_task(work)
    try:
        yield
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


# A child's stdin is a pipe only the supervisor holds open, so the child sees EOF as soon as the supervisor is gone,
# however it went, and shuts down instead of outliving it.
@contextlib.asynccontextmanager
async def _launch(
    platform: Platform,
    child: Child,
    spec: ChildSpec,
    events: asyncio.Queue[Event],
) -> AsyncGenerator[asyncio.subprocess.Process]:
    process: Final = await _spawn(spec)
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


@dataclass(frozen=True, slots=True)
class _Launching:
    state: Lifecycle
    child: Child
    rest: tuple[Action, ...]


async def _advance(
    platform: Platform,
    transition: Transition,
    events: asyncio.Queue[Event],
    children: Mapping[Child, asyncio.subprocess.Process],
) -> int | _Launching:
    state, actions = transition
    while True:
        for index, action in enumerate(actions):
            match action:
                case Send(child, signum):
                    platform.signal_child(children[child], signum)
                case Spawn(child):
                    return _Launching(state, child, actions[index + 1 :])
                case _:
                    assert_never(action)
        status = exit_status(state)
        if status is not None:
            return status
        state, actions = step(state, await events.get(), platform.stop_signals)


async def _run(
    platform: Platform,
    transition: Transition,
    events: asyncio.Queue[Event],
    children: Mapping[Child, asyncio.subprocess.Process],
    launches: Mapping[Child, Launch],
) -> int:
    advanced: Final = await _advance(platform, transition, events, children)
    match advanced:
        case int():
            return advanced
        case _Launching(state, child, rest):
            async with launches[child](events) as process:
                return await _run(
                    platform, (state, rest), events, MappingProxyType({**children, child: process}), launches
                )
        case _:
            assert_never(advanced)


async def _supervise(platform: Platform, binary: str, command: Sequence[str], server: PythonServer) -> int:
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
        return await _run(platform, boot(), events, MappingProxyType({}), launches)


def _exit_like(status: int) -> int:
    """A child killed by a signal kills the supervisor with the same signal, as uvicorn re-raises the one it caught."""
    if status >= 0:
        return status
    signum: Final = signal.Signals(-status)
    signal.signal(signum, signal.SIG_DFL)
    signal.raise_signal(signum)
    return 128 + signum


def _nothing_to_prepare() -> None:
    pass


def supervise(
    server: PythonServer,
    command: Sequence[str],
    before_start: Callable[[], None] = _nothing_to_prepare,
    scripts_dir: str = sysconfig.get_path("scripts"),
    platform: Platform | None = None,
) -> int | StartupError:
    """Serve ``server``'s public port from Rust and run ``command`` behind it, or say why that cannot start.

    A refusal is returned for the caller to serve from Python alone, and ``before_start`` runs only once forwarding
    is known to be possible, so a refused start leaves nothing behind.
    """
    chosen: Final = current_platform() if platform is None else platform
    checked: Final = startup_check(chosen, server, gateway_binary(chosen, scripts_dir))
    if isinstance(checked, StartupError):
        sys.stderr.write(f"{checked.reason}, so Python serves the public port without Rust forwarding\n")
        return checked
    before_start()
    try:
        status: Final = asyncio.run(_supervise(chosen, checked, command, server))
    except OSError as error:
        sys.stderr.write(f"Rust gateway startup failed: {error}\n")
        return FAILURE
    return _exit_like(status)
