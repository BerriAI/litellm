import asyncio
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Final, LiteralString, Protocol

if TYPE_CHECKING:
    from litellm.proxy.db.prisma_client import PrismaWrapper
    from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper


hook_statements: Final = asyncio.Semaphore(4)


class Session(Protocol):
    async def query_raw(self, query: LiteralString, *args: object) -> object: ...

    async def execute_raw(self, query: LiteralString, *args: object) -> int: ...


class Database(Session, Protocol):
    def tx(self, *, timeout: int = 5000) -> AbstractAsyncContextManager[Session]: ...


class ProxyDatabase:
    def __init__(
        self, db: "PrismaWrapper | RoutingPrismaWrapper", limiter: asyncio.Semaphore = hook_statements
    ) -> None:
        self.db = db
        self.limiter = limiter

    async def query_raw(self, query: LiteralString, *args: object) -> object:
        from litellm.proxy.utils import query_raw_rows

        async with self.limiter:
            return await query_raw_rows(self.db, query, *args)

    async def execute_raw(self, query: LiteralString, *args: object) -> int:
        async with self.limiter:
            return await self.db.execute_raw(query, *args)

    def tx(self, *, timeout: int = 5000) -> AbstractAsyncContextManager[Session]:
        return self.db.tx(timeout=timeout)
