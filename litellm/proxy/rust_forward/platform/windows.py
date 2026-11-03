import asyncio
import contextlib
import signal
import socket
from collections.abc import Callable, Generator
from dataclasses import dataclass
from types import FrameType
from typing import Final

from pydantic import TypeAdapter

from litellm.proxy.rust_forward.mode import RUST_HOP, AssignedPort
from litellm.proxy.rust_forward.platform import PrivateListener, SpawnOptions
from litellm.proxy.rust_forward.server import Server

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
    def private_listener(self) -> Generator[PrivateListener]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((RUST_HOP, 0))
            _, port = _ADDRESS.validate_python(probe.getsockname())
        yield PrivateListener(AssignedPort(port), port, ())

    def spawn_options(self, pass_fds: tuple[int, ...]) -> SpawnOptions:
        return SpawnOptions(pass_fds=pass_fds, start_new_session=False, creationflags=CREATE_NEW_PROCESS_GROUP)

    def catch_stop(self, loop: asyncio.AbstractEventLoop, on_stop: Callable[[signal.Signals], None]) -> None:
        def handle(signum: int, _frame: FrameType | None) -> None:
            loop.call_soon_threadsafe(on_stop, signal.Signals(signum))

        for signum in (signal.SIGINT, SIGBREAK):
            signal.signal(signum, handle)

    def signal_child(self, process: asyncio.subprocess.Process, signum: signal.Signals) -> None:
        self.stop_group(process)

    def stop_group(self, process: asyncio.subprocess.Process) -> None:
        with contextlib.suppress(ProcessLookupError):
            process.send_signal(CTRL_BREAK_EVENT)
