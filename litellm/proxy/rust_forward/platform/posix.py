import asyncio
import contextlib
import os
import signal
import socket
from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import Final

from pydantic import TypeAdapter

from litellm.proxy.rust_forward.mode import RUST_HOP
from litellm.proxy.rust_forward.platform import PlatformListener, SpawnOptions
from litellm.proxy.rust_forward.server import Server
from litellm.proxy.server_listener import InheritedFd

_ADDRESS: Final = TypeAdapter(tuple[str, int])


@dataclass(frozen=True, slots=True)
class PosixPlatform:
    name: str = "POSIX"
    gateway_name: str = "litellm-gateway"
    servers: frozenset[Server] = frozenset({Server.UVICORN, Server.GUNICORN})

    @contextlib.contextmanager
    def private_listener(self) -> Generator[PlatformListener]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as private:
            private.bind((RUST_HOP, 0))
            _, port = _ADDRESS.validate_python(private.getsockname())
            yield PlatformListener(InheritedFd(private.fileno()), port, (private.fileno(),))

    def spawn_options(self, pass_fds: tuple[int, ...]) -> SpawnOptions:
        return SpawnOptions(pass_fds=pass_fds, start_new_session=True, creationflags=0)

    @property
    def relayed_signals(self) -> tuple[signal.Signals, ...]:
        return (
            signal.SIGTERM,
            signal.SIGINT,
            signal.SIGHUP,
            signal.SIGQUIT,
            signal.SIGUSR1,
            signal.SIGUSR2,
            signal.SIGTTIN,
            signal.SIGTTOU,
            signal.SIGWINCH,
        )

    @property
    def stop_signals(self) -> frozenset[signal.Signals]:
        return frozenset({signal.SIGTERM, signal.SIGINT, signal.SIGQUIT})

    def catch(self, loop: asyncio.AbstractEventLoop, on_signal: Callable[[signal.Signals], None]) -> None:
        for signum in self.relayed_signals:
            loop.add_signal_handler(signum, on_signal, signum)

    def signal_child(self, process: asyncio.subprocess.Process, signum: signal.Signals) -> None:
        with contextlib.suppress(ProcessLookupError):
            process.send_signal(signum)

    def stop_group(self, process: asyncio.subprocess.Process) -> None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
