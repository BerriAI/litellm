import asyncio
import contextlib
import os
import shutil
import signal
import sys
import sysconfig
from collections.abc import AsyncGenerator, Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, IntEnum
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
        return StartupError(
            f"{ENV_NAME} requires the litellm-gateway binary. Install litellm[proxy] in this environment"
        )
    return binary


class Child(Enum):
    RUST = "rust"
    PYTHON = "python"


# Each child starts once the one before it is ready. Python alone takes no connections until it has booted, so Rust
# only opens the public port once Python listens.
START_ORDER: Final = (Child.PYTHON, Child.RUST)


class ExitCode(IntEnum):
    CLEAN = 0
    CHILD_FAILED = 1


@dataclass(frozen=True, slots=True)
class Ready:
    child: Child


@dataclass(frozen=True, slots=True)
class Stop:
    signum: signal.Signals


@dataclass(frozen=True, slots=True)
class Exited:
    child: Child
    code: int


@dataclass(frozen=True, slots=True)
class Running:
    up: tuple[Child, ...]


@dataclass(frozen=True, slots=True)
class Stopping:
    up: frozenset[Child]


@dataclass(frozen=True, slots=True)
class Finished:
    code: ExitCode


@dataclass(frozen=True, slots=True)
class Start:
    child: Child


@dataclass(frozen=True, slots=True)
class Signal:
    child: Child
    signum: signal.Signals


Event: TypeAlias = Ready | Stop | Exited
State: TypeAlias = Running | Stopping | Finished
Action: TypeAlias = Start | Signal
Transition: TypeAlias = tuple[State, tuple[Action, ...]]


def _start_after(up: tuple[Child, ...]) -> Transition:
    following: Final = START_ORDER[len(up) : len(up) + 1]
    return Running((*up, *following)), tuple(Start(child) for child in following)


def boot() -> Transition:
    return _start_after(())


# Every child gets the signal at once, as Python alone stopped. Python ends its long streams within its graceful
# timeout, which Rust's drain already covers, so stopping Python only after Rust exits would hold Python's shutdown,
# spend flush included, behind streams only Python can end. Reverse start order lets Rust stop accepting first.
def _stop(up: tuple[Child, ...], signum: signal.Signals) -> Transition:
    return Stopping(frozenset(up)), tuple(Signal(child, signum) for child in reversed(up))


def step(state: State, event: Event) -> Transition:
    match state:
        case Running(up):
            match event:
                case Ready(child) if up[-1:] == (child,):
                    return _start_after(up)
                case Ready():
                    return state, ()
                case Stop(signum):
                    return _stop(up, signum)
                case Exited():
                    return Finished(ExitCode.CHILD_FAILED), ()
                case _:
                    assert_never(event)
        case Stopping(up):
            match event:
                case Ready() | Stop():
                    return state, ()
                case Exited(child, code) if child in up and code == 0:
                    remaining: Final = up - {child}
                    return (Stopping(remaining) if remaining else Finished(ExitCode.CLEAN)), ()
                case Exited():
                    return Finished(ExitCode.CHILD_FAILED), ()
                case _:
                    assert_never(event)
        case Finished():
            return state, ()
        case _:
            assert_never(state)


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
        env={**os.environ, **spec.env_overrides},
        pass_fds=spec.spawn.pass_fds,
        start_new_session=spec.spawn.start_new_session,
        creationflags=spec.spawn.creationflags,
    )


Launch: TypeAlias = Callable[[asyncio.Queue[Event]], contextlib.AbstractAsyncContextManager[asyncio.subprocess.Process]]


async def _report_exit(child: Child, process: asyncio.subprocess.Process, events: asyncio.Queue[Event]) -> None:
    events.put_nowait(Exited(child, await process.wait()))


async def _accepts(port: int) -> bool:
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection(RUST_HOP, port), PROBE_INTERVAL_SECONDS)
    except (OSError, asyncio.TimeoutError):
        return False
    writer.close()
    return True


async def _report_ready_once_listening(child: Child, port: int, events: asyncio.Queue[Event]) -> None:
    while not await _accepts(port):
        await asyncio.sleep(PROBE_INTERVAL_SECONDS)
    events.put_nowait(Ready(child))


@contextlib.asynccontextmanager
async def _in_background(work: Coroutine[None, None, None]) -> AsyncGenerator[None]:
    task: Final = asyncio.create_task(work)
    try:
        yield
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


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


@contextlib.asynccontextmanager
async def _launch_until_listening(
    platform: Platform, child: Child, spec: ChildSpec, port: int, events: asyncio.Queue[Event]
) -> AsyncGenerator[asyncio.subprocess.Process]:
    async with (
        _launch(platform, child, spec, events) as process,
        _in_background(_report_ready_once_listening(child, port, events)),
    ):
        yield process


async def _run(
    platform: Platform,
    transition: Transition,
    events: asyncio.Queue[Event],
    children: Mapping[Child, asyncio.subprocess.Process],
    launches: Mapping[Child, Launch],
) -> ExitCode:
    state, actions = transition
    if not actions:
        if isinstance(state, Finished):
            return state.code
        return await _run(platform, step(state, await events.get()), events, children, launches)
    action: Final = actions[0]
    rest: Final = (state, actions[1:])
    match action:
        case Signal(child, signum):
            platform.signal_child(children[child], signum)
            return await _run(platform, rest, events, children, launches)
        case Start(child):
            async with launches[child](events) as process:
                return await _run(platform, rest, events, MappingProxyType({**children, child: process}), launches)
        case _:
            assert_never(action)


async def _supervise(platform: Platform, binary: str, command: Sequence[str], server: PythonServer) -> ExitCode:
    events: Final[asyncio.Queue[Event]] = asyncio.Queue()
    platform.catch_stop(asyncio.get_running_loop(), lambda signum: events.put_nowait(Stop(signum)))
    with platform.private_listener() as private:
        launches: Final[Mapping[Child, Launch]] = MappingProxyType(
            {
                Child.PYTHON: partial(
                    _launch_until_listening,
                    platform,
                    Child.PYTHON,
                    python_spec(platform, command, private, server),
                    private.port,
                ),
                Child.RUST: partial(_launch, platform, Child.RUST, rust_spec(platform, binary, private.port, server)),
            }
        )
        return await _run(platform, boot(), events, MappingProxyType({}), launches)


def _nothing_to_prepare() -> None:
    pass


def supervise(
    server: PythonServer,
    command: Sequence[str],
    before_start: Callable[[], None] = _nothing_to_prepare,
    scripts_dir: str = sysconfig.get_path("scripts"),
    platform: Platform | None = None,
) -> int:
    """Serve ``server``'s public port from Rust and run ``command`` behind it, or say why that cannot start.

    ``before_start`` runs only once forwarding is known to be possible, so a refused start leaves nothing behind.
    """
    chosen: Final = current_platform() if platform is None else platform
    checked: Final = startup_check(chosen, server, gateway_binary(chosen, scripts_dir))
    if isinstance(checked, StartupError):
        sys.stderr.write(f"{checked.reason}\n")
        return ExitCode.CHILD_FAILED
    before_start()
    try:
        return asyncio.run(_supervise(chosen, checked, command, server))
    except OSError as error:
        sys.stderr.write(f"Rust gateway startup failed: {error}\n")
        return ExitCode.CHILD_FAILED
