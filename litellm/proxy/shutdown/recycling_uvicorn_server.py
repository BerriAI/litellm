import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from socket import socket
from types import FrameType
from typing import Final

from uvicorn import Config, Server

DEFAULT_RECYCLE_DRAIN_TIMEOUT: Final = 600


class RecyclingUvicornServer(Server):
    def __init__(self, config: Config, recycle_drain_timeout: int = DEFAULT_RECYCLE_DRAIN_TIMEOUT) -> None:
        super().__init__(config)
        self._shutdown_signal: Final = asyncio.Event()
        self._recycle_drain_timeout: Final = recycle_drain_timeout
        self._recycle_signal_timeout: float | None = None

    def handle_exit(self, sig: int, frame: FrameType | None) -> None:
        super().handle_exit(sig, frame)
        self._shutdown_signal.set()

    async def shutdown(self, sockets: Sequence[socket] | None = None) -> None:
        shared_sockets: Final = list(sockets) if sockets is not None else None
        timeout: Final = self.config.timeout_graceful_shutdown
        if self.should_exit or self.config.limit_max_requests is None or timeout is None:
            await super().shutdown(shared_sockets)
            return

        logging.getLogger("uvicorn.error").info("Draining in-flight requests before worker recycle")
        callback: Final = self.config.callback_notify
        interval: Final = self.config.timeout_notify
        heartbeat: Final = (
            asyncio.create_task(self._notify_during_recycle(callback, interval))
            if callback is not None and interval > 0
            else None
        )
        self._recycle_signal_timeout = timeout
        self.config.timeout_graceful_shutdown = max(self._recycle_drain_timeout, timeout)
        try:
            await super().shutdown(shared_sockets)
        finally:
            self.config.timeout_graceful_shutdown = timeout
            self._recycle_signal_timeout = None
            if heartbeat is not None:
                heartbeat.cancel()
                await asyncio.gather(heartbeat, return_exceptions=True)

    @staticmethod
    async def _notify_during_recycle(callback: Callable[[], Awaitable[None]], interval: float) -> None:
        while True:
            await callback()
            await asyncio.sleep(interval)

    async def _wait_tasks_to_complete(self) -> None:
        timeout: Final = self._recycle_signal_timeout
        if timeout is None:
            await super()._wait_tasks_to_complete()
            return

        drain: Final = asyncio.create_task(super()._wait_tasks_to_complete())
        signal_received: Final = asyncio.create_task(self._shutdown_signal.wait())
        try:
            await asyncio.wait((drain, signal_received), return_when=asyncio.FIRST_COMPLETED)
            if drain.done():
                await drain
            else:
                await asyncio.wait_for(drain, timeout=timeout)
        finally:
            signal_received.cancel()
            if not drain.done():
                drain.cancel()
            await asyncio.gather(drain, signal_received, return_exceptions=True)
