import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Final


class ChargeReceiptStream(AsyncIterator[object]):
    def __init__(self, source: AsyncIterator[object], close: Callable[[], Awaitable[None]]) -> None:
        self.source: Final = source
        self.close_source: Final = close
        self.pending: asyncio.Task[object] | None = None

    async def __anext__(self) -> object:
        task: Final = self.pending if self.pending is not None else asyncio.create_task(self._read_next())
        self.pending = task
        try:
            return await asyncio.shield(task)
        finally:
            if task.done():
                self.pending = None

    async def _read_next(self) -> object:
        return await anext(self.source)

    async def drain(self) -> None:
        async for _ in self:
            pass

    async def aclose(self) -> None:
        pending: Final = self.pending
        if pending is not None and not pending.done():
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        await self.close_source()
