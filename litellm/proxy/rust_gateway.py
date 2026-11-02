import asyncio
import contextlib
import os
import shutil
import signal
import socket
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, IntEnum
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter
from typing_extensions import assert_never

ENABLE: Final = "LITELLM_RUST_GATEWAY"
GATEWAY_FD: Final = "_LITELLM_GATEWAY_FD"
METRICS_HOST: Final = "_LITELLM_GATEWAY_METRICS_HOST"
TRUSTED_PROXIES: Final = "FORWARDED_ALLOW_IPS"
RUST_HOP: Final = "127.0.0.1"

_FD: Final = TypeAdapter(int)
_ENABLED: Final = TypeAdapter(bool)
_ADDRESS: Final = TypeAdapter(tuple[str, int])


@dataclass(frozen=True, slots=True)
class Disabled:
    pass


@dataclass(frozen=True, slots=True)
class Supervisor:
    pass


@dataclass(frozen=True, slots=True)
class Supervised:
    listener_fd: int
    metrics_host: str
    trusted_proxies: tuple[str, ...]


RustGatewayMode = Disabled | Supervisor | Supervised


def trusted_proxies(operator_value: str | None) -> tuple[str, ...]:
    configured: Final = tuple(host.strip() for host in (operator_value or "").split(",") if host.strip())
    return ("*",) if "*" in configured else (RUST_HOP, *(host for host in configured if host != RUST_HOP))


def rust_gateway_mode(environ: Mapping[str, str]) -> RustGatewayMode:
    fd: Final = environ.get(GATEWAY_FD)
    if fd is not None:
        return Supervised(_FD.validate_python(fd), environ[METRICS_HOST], trusted_proxies(environ.get(TRUSTED_PROXIES)))
    if _ENABLED.validate_python(environ.get(ENABLE, "false")):
        return Supervisor()
    return Disabled()


def gateway_binary(scripts_dir: str) -> str | None:
    name: Final = "litellm-gateway.exe" if os.name == "nt" else "litellm-gateway"
    return shutil.which(name, path=scripts_dir)


def startup_error(binary: str | None, unsupported: Mapping[str, object]) -> str | None:
    flag: Final = next((name for name, value in unsupported.items() if value), None)
    if flag is not None:
        return f"{ENABLE} does not support {flag}"
    if os.name != "posix":
        return f"{ENABLE} requires POSIX inherited sockets and process signals"
    if binary is None:
        return f"{ENABLE} requires the litellm-gateway binary. Install litellm[proxy] in this environment"
    return None


class Child(Enum):
    RUST = "rust"
    PYTHON = "python"


DRAIN_ORDER: Final = (Child.RUST, Child.PYTHON)


class ExitCode(IntEnum):
    CLEAN = 0
    CHILD_FAILED = 1


@dataclass(frozen=True, slots=True)
class Stop:
    signum: signal.Signals


@dataclass(frozen=True, slots=True)
class Exited:
    child: Child
    code: int


@dataclass(frozen=True, slots=True)
class Running:
    pass


@dataclass(frozen=True, slots=True)
class Draining:
    signum: signal.Signals
    signalled: Child
    remaining: tuple[Child, ...]


@dataclass(frozen=True, slots=True)
class Finished:
    code: ExitCode


@dataclass(frozen=True, slots=True)
class Signal:
    child: Child
    signum: signal.Signals


Event = Stop | Exited
State = Running | Draining | Finished


def _signal_next(signum: signal.Signals, order: tuple[Child, ...]) -> tuple[State, Signal | None]:
    if not order:
        return Finished(ExitCode.CLEAN), None
    return Draining(signum, order[0], order[1:]), Signal(order[0], signum)


def step(state: State, event: Event) -> tuple[State, Signal | None]:
    match state:
        case Running():
            match event:
                case Stop(signum):
                    return _signal_next(signum, DRAIN_ORDER)
                case Exited():
                    return Finished(ExitCode.CHILD_FAILED), None
                case _:
                    assert_never(event)
        case Draining(signum, signalled, remaining):
            match event:
                case Stop():
                    return state, None
                case Exited(child, code) if child is signalled and code == 0:
                    return _signal_next(signum, remaining)
                case Exited():
                    return Finished(ExitCode.CHILD_FAILED), None
                case _:
                    assert_never(event)
        case Finished():
            return state, None
        case _:
            assert_never(state)


@dataclass(frozen=True, slots=True)
class ChildSpec:
    argv: tuple[str, ...]
    env: Mapping[str, str]
    pass_fds: tuple[int, ...]


def python_spec(
    command: Sequence[str], listener_fd: int, upstream_port: int, public_host: str, environ: Mapping[str, str]
) -> ChildSpec:
    return ChildSpec(
        argv=(*command, "--host", RUST_HOP, "--port", str(upstream_port)),
        env=MappingProxyType(
            {
                **environ,
                GATEWAY_FD: str(listener_fd),
                METRICS_HOST: public_host,
                TRUSTED_PROXIES: ",".join(trusted_proxies(environ.get(TRUSTED_PROXIES))),
            }
        ),
        pass_fds=(listener_fd,),
    )


def rust_spec(binary: str, upstream_port: int, host: str, port: int, environ: Mapping[str, str]) -> ChildSpec:
    return ChildSpec(
        argv=(binary,),
        env=MappingProxyType(
            {
                **environ,
                "HOST": host,
                "PORT": str(port),
                "LITELLM_FORWARD_UPSTREAM": f"http://{RUST_HOP}:{upstream_port}",
            }
        ),
        pass_fds=(),
    )


async def _spawn(spec: ChildSpec) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        *spec.argv, env=dict(spec.env), pass_fds=spec.pass_fds, start_new_session=True
    )


def _terminate_group(process: asyncio.subprocess.Process) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)


async def _report_exit(child: Child, process: asyncio.subprocess.Process, events: asyncio.Queue[Event]) -> None:
    events.put_nowait(Exited(child, await process.wait()))


async def _drive(
    state: State, events: asyncio.Queue[Event], children: Mapping[Child, asyncio.subprocess.Process]
) -> ExitCode:
    if isinstance(state, Finished):
        return state.code
    following, action = step(state, await events.get())
    if action is not None:
        with contextlib.suppress(ProcessLookupError):
            children[action.child].send_signal(action.signum)
    return await _drive(following, events, children)


async def _supervise(binary: str, command: Sequence[str], host: str, port: int) -> ExitCode:
    loop: Final = asyncio.get_running_loop()
    events: Final[asyncio.Queue[Event]] = asyncio.Queue()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, events.put_nowait, Stop(signum))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((RUST_HOP, 0))
        listener.listen()
        _, upstream_port = _ADDRESS.validate_python(listener.getsockname())
        python: Final = await _spawn(python_spec(command, listener.fileno(), upstream_port, host, os.environ))
        try:
            rust: Final = await _spawn(rust_spec(binary, upstream_port, host, port, os.environ))
        except OSError:
            _terminate_group(python)
            await python.wait()
            raise
    children: Final = MappingProxyType({Child.RUST: rust, Child.PYTHON: python})
    waiters: Final = tuple(
        asyncio.create_task(_report_exit(child, process, events)) for child, process in children.items()
    )
    try:
        return await _drive(Running(), events, children)
    finally:
        for process in children.values():
            _terminate_group(process)
        await asyncio.gather(*waiters)


def supervise(binary: str, command: Sequence[str], host: str, port: int) -> int:
    try:
        return asyncio.run(_supervise(binary, command, host, port))
    except OSError as error:
        sys.stderr.write(f"Rust gateway startup failed: {error}\n")
        return ExitCode.CHILD_FAILED


def cli_command(argv: Sequence[str]) -> tuple[str, ...]:
    return (sys.executable, "-m", "litellm.proxy.proxy_cli", *argv)
