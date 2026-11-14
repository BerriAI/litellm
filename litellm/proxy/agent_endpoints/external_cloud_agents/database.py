import asyncio
from typing import TYPE_CHECKING, Final, Protocol

from typing_extensions import LiteralString

if TYPE_CHECKING:
    from litellm.proxy.db.prisma_client import PrismaWrapper
    from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper

statements: Final = asyncio.Semaphore(4)


class Database(Protocol):
    async def query_raw(self, query: LiteralString, *args: object) -> object: ...

    async def execute_raw(self, query: LiteralString, *args: object) -> int: ...


class ProxyDatabase:
    def __init__(self, db: "PrismaWrapper | RoutingPrismaWrapper | None") -> None:
        self.db = db

    async def query_raw(self, query: LiteralString, *args: object) -> object:
        from litellm.proxy.utils import query_raw_rows

        if self.db is None:
            raise RuntimeError("Database is unavailable")
        async with statements:
            return await query_raw_rows(self.db, query, *args)

    async def execute_raw(self, query: LiteralString, *args: object) -> int:
        if self.db is None:
            raise RuntimeError("Database is unavailable")
        async with statements:
            return await self.db.execute_raw(query, *args)
