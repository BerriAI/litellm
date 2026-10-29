import asyncio
from collections.abc import AsyncGenerator, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from litellm.rust_bridge._native import NativeDatabase

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


@dataclass(frozen=True, slots=True)
class Database:
    prisma: "PrismaClient"
    native: NativeDatabase

    @classmethod
    async def open(cls, prisma: "PrismaClient") -> "Database":
        try:
            native: Final = await NativeDatabase.connect()
        except BaseException:
            await prisma.aclose()
            raise
        return cls(prisma=prisma, native=native)

    async def aclose(self) -> None:
        try:
            await self.prisma.aclose()
        finally:
            await self.native.close()


@asynccontextmanager
async def background_task(work: Coroutine[object, object, None]) -> AsyncGenerator[None, None]:
    task: Final = asyncio.create_task(work)
    try:
        yield
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
