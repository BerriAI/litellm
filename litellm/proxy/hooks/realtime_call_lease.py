import asyncio
from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Final

_realtime_call_attachment: Final[ContextVar[object | None]] = ContextVar("realtime_call_attachment", default=None)


@contextmanager
def realtime_call_attachment(websocket: object) -> Generator[None]:
    token: Final = _realtime_call_attachment.set(websocket)
    try:
        yield
    finally:
        _realtime_call_attachment.reset(token)


def is_realtime_call_attachment(websocket: object) -> bool:
    bound: Final = _realtime_call_attachment.get()
    return bound is not None and bound is websocket


class RealtimeCallLease:
    def __init__(
        self,
        *,
        renew: Callable[[], Awaitable[bool]],
        release: Callable[[], Awaitable[None]],
        interval: float = 300,
        renewal_timeout: float = 10,
    ) -> None:
        self._renew = renew
        self._release = release
        self._interval = interval
        self._renewal_timeout = renewal_timeout
        self._failed = asyncio.Event()
        self._heartbeat: asyncio.Task[None] | None = None
        self._closing: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._heartbeat is None and self._closing is None:
            self._heartbeat = asyncio.create_task(self._run())

    async def renew(self) -> bool:
        if self._closing is not None or self._failed.is_set():
            return False
        try:
            renewed: Final = await asyncio.wait_for(self._renew(), timeout=self._renewal_timeout)
        except Exception:  # noqa: BLE001  # fail closed without exposing cache credentials
            self._failed.set()
            return False
        if not renewed:
            self._failed.set()
        return renewed and not self._failed.is_set() and self._closing is None

    async def wait_failed(self) -> None:
        await self._failed.wait()

    async def _run(self) -> None:
        while await self.renew():
            await asyncio.sleep(self._interval)
        self._failed.set()

    async def close(self) -> None:
        if self._closing is None:
            self._closing = asyncio.create_task(self._close())
        await asyncio.shield(self._closing)

    async def _close(self) -> None:
        if self._heartbeat is not None:
            self._heartbeat.cancel()
            await asyncio.gather(self._heartbeat, return_exceptions=True)
        await self._release()
