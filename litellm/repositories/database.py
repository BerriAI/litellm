import asyncio
from collections.abc import AsyncGenerator, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from litellm._logging import verbose_proxy_logger

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


@dataclass(frozen=True, slots=True)
class Database:
    prisma: "PrismaClient"

    async def aclose(self) -> None:
        verbose_proxy_logger.debug("Disconnecting from Prisma")
        try:
            await self.prisma.stop_background_tasks()
        finally:
            await self.prisma.disconnect()


@asynccontextmanager
async def background_task(work: Coroutine[object, object, None]) -> AsyncGenerator[None, None]:
    task: Final = asyncio.create_task(work)
    try:
        yield
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
