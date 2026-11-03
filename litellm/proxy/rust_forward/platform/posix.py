import asyncio
import contextlib
import os
import signal
import socket
from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import Final

from pydantic import TypeAdapter

from litellm.proxy.rust_forward.mode import RUST_HOP, InheritedFd
from litellm.proxy.rust_forward.platform import PrivateListener, SpawnOptions
from litellm.proxy.rust_forward.server import Server

_ADDRESS: Final = TypeAdapter(tuple[str, int])


@dataclass(frozen=True, slots=True)
class PosixPlatform:
    name: str = "POSIX"
    gateway_name: str = "litellm-gateway"
    servers: frozenset[Server] = frozenset({Server.UVICORN, Server.GUNICORN})

    @contextlib.contextmanager
    def private_listener(self) -> Generator[PrivateListener]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as private:
            private.bind((RUST_HOP, 0))
            _, port = _ADDRESS.validate_python(private.getsockname())
            yield PrivateListener(InheritedFd(private.fileno()), port, (private.fileno(),))

    def spawn_options(self, pass_fds: tuple[int, ...]) -> SpawnOptions:
        return SpawnOptions(pass_fds=pass_fds, start_new_session=True, creationflags=0)

    def catch_stop(self, loop: asyncio.AbstractEventLoop, on_stop: Callable[[signal.Signals], None]) -> None:
        for signum in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(signum, on_stop, signum)

    def signal_child(self, process: asyncio.subprocess.Process, signum: signal.Signals) -> None:
        with contextlib.suppress(ProcessLookupError):
            process.send_signal(signum)

    def stop_group(self, process: asyncio.subprocess.Process) -> None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
