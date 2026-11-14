import asyncio
import contextlib
import signal
import socket
from collections.abc import Callable, Generator
from dataclasses import dataclass
from types import FrameType
from typing import Final

from pydantic import TypeAdapter

from litellm.proxy.rust_forward.mode import RUST_HOP
from litellm.proxy.rust_forward.platform import ChildProcess, PlatformListener, SpawnOptions
from litellm.proxy.rust_forward.server import Server
from litellm.proxy.server_listener import AssignedPort

_ADDRESS: Final = TypeAdapter(tuple[str, int])
# signal.CTRL_BREAK_EVENT, signal.SIGBREAK and subprocess.CREATE_NEW_PROCESS_GROUP exist only on Windows, so they are
# spelled out to keep this module importable and type checked everywhere. A Windows-only test pins them to the stdlib.
CTRL_BREAK_EVENT: Final = 1
SIGBREAK: Final = 21
CREATE_NEW_PROCESS_GROUP: Final = 0x00000200


# Windows support exists only for running the proxy locally from the CLI, never for deploying it. That is why the
# supervisor picks a free loopback port and hands Python its number instead of an open socket, which Windows uvicorn
# cannot serve. Another process could take that port before Python binds it, and the readiness probe would then
# accept it as Python. Python's failed bind still ends the run, which is acceptable on a developer's machine.
@dataclass(frozen=True, slots=True)
class WindowsPlatform:
    name: str = "Windows"
    gateway_name: str = "litellm-gateway.exe"
    servers: frozenset[Server] = frozenset({Server.UVICORN})

    @contextlib.contextmanager
    def private_listener(self) -> Generator[PlatformListener]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((RUST_HOP, 0))
            _, port = _ADDRESS.validate_python(probe.getsockname())
        yield PlatformListener(AssignedPort(port), port, ())

    def spawn_options(self, pass_fds: tuple[int, ...]) -> SpawnOptions:
        return SpawnOptions(pass_fds=pass_fds, start_new_session=False, creationflags=CREATE_NEW_PROCESS_GROUP)

    @property
    def relayed_signals(self) -> tuple[signal.Signals, ...]:
        return (signal.SIGINT, signal.Signals(SIGBREAK))

    @property
    def stop_signals(self) -> frozenset[signal.Signals]:
        return frozenset(self.relayed_signals)

    def catch(self, loop: asyncio.AbstractEventLoop, on_signal: Callable[[signal.Signals], None]) -> None:
        def handle(signum: int, _frame: FrameType | None) -> None:
            loop.call_soon_threadsafe(on_signal, signal.Signals(signum))

        for signum in self.relayed_signals:
            signal.signal(signum, handle)

    # A child in its own process group can only be sent CTRL_BREAK_EVENT, since Windows never delivers CTRL_C_EVENT
    # to a new group. Both children shut down on it as on Ctrl-C, but uvicorn force-exits only on a repeated SIGINT,
    # so a second Ctrl-C does not cut a Windows shutdown short the way it does on POSIX.
    def signal_child(self, process: ChildProcess, signum: signal.Signals) -> None:
        self.stop_group(process)

    def stop_group(self, process: ChildProcess) -> None:
        with contextlib.suppress(ProcessLookupError):
            process.send_signal(CTRL_BREAK_EVENT)
