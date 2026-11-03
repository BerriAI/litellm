import asyncio
import os
import signal
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Protocol

from litellm.proxy.rust_forward.mode import ChildListener
from litellm.proxy.rust_forward.server import Server


@dataclass(frozen=True, slots=True)
class PrivateListener:
    listener: ChildListener
    port: int
    pass_fds: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SpawnOptions:
    pass_fds: tuple[int, ...]
    start_new_session: bool
    creationflags: int


class Platform(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def gateway_name(self) -> str: ...

    @property
    def servers(self) -> frozenset[Server]: ...

    def private_listener(self) -> AbstractContextManager[PrivateListener]: ...

    def spawn_options(self, pass_fds: tuple[int, ...]) -> SpawnOptions: ...

    def catch_stop(self, loop: asyncio.AbstractEventLoop, on_stop: Callable[[signal.Signals], None]) -> None: ...

    def signal_child(self, process: asyncio.subprocess.Process, signum: signal.Signals) -> None: ...

    def stop_group(self, process: asyncio.subprocess.Process) -> None: ...


def current_platform() -> Platform:
    from litellm.proxy.rust_forward.platform.posix import PosixPlatform
    from litellm.proxy.rust_forward.platform.windows import WindowsPlatform

    return WindowsPlatform() if os.name == "nt" else PosixPlatform()
