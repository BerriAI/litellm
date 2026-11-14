import asyncio
import os
import signal
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Protocol

from litellm.proxy.rust_forward.server import Server
from litellm.proxy.server_listener import PrivateListener


@dataclass(frozen=True, slots=True)
class PlatformListener:
    listener: PrivateListener
    port: int
    pass_fds: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SpawnOptions:
    pass_fds: tuple[int, ...]
    start_new_session: bool
    creationflags: int


class ChildProcess(Protocol):
    @property
    def pid(self) -> int: ...

    @property
    def stdin(self) -> asyncio.StreamWriter | None: ...

    def send_signal(self, signum: int, /) -> None: ...

    async def wait(self) -> int: ...


class Platform(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def gateway_name(self) -> str: ...

    @property
    def servers(self) -> frozenset[Server]: ...

    def private_listener(self) -> AbstractContextManager[PlatformListener]: ...

    def spawn_options(self, pass_fds: tuple[int, ...]) -> SpawnOptions: ...

    @property
    def relayed_signals(self) -> tuple[signal.Signals, ...]: ...

    @property
    def stop_signals(self) -> frozenset[signal.Signals]: ...

    def catch(self, loop: asyncio.AbstractEventLoop, on_signal: Callable[[signal.Signals], None]) -> None: ...

    def signal_child(self, process: ChildProcess, signum: signal.Signals) -> None: ...

    def stop_group(self, process: ChildProcess) -> None: ...


def current_platform() -> Platform:
    from litellm.proxy.rust_forward.platform.posix import PosixPlatform
    from litellm.proxy.rust_forward.platform.windows import WindowsPlatform

    return WindowsPlatform() if os.name == "nt" else PosixPlatform()
