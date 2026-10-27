from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


@dataclass(frozen=True, slots=True)
class Database:
    prisma: "PrismaClient"

    async def aclose(self) -> None:
        await self.prisma.disconnect()


@asynccontextmanager
async def lifespan(database: Database | None) -> AsyncGenerator[Database | None, None]:
    try:
        yield database
    finally:
        if database is not None:
            await database.aclose()
