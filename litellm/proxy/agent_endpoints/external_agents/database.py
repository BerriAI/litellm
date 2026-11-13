from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, LiteralString, Protocol

if TYPE_CHECKING:
    from litellm.proxy.db.prisma_client import PrismaWrapper
    from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper


class Session(Protocol):
    async def query_raw(self, query: LiteralString, *args: object) -> object: ...

    async def execute_raw(self, query: LiteralString, *args: object) -> int: ...


class Database(Session, Protocol):
    def tx(self, *, timeout: int = 5000) -> AbstractAsyncContextManager[Session]: ...


class ProxyDatabase:
    def __init__(self, db: "PrismaWrapper | RoutingPrismaWrapper") -> None:
        self.db = db

    async def query_raw(self, query: LiteralString, *args: object) -> object:
        from litellm.proxy.utils import query_raw_rows

        return await query_raw_rows(self.db, query, *args)

    async def execute_raw(self, query: LiteralString, *args: object) -> int:
        return await self.db.execute_raw(query, *args)

    def tx(self, *, timeout: int = 5000) -> AbstractAsyncContextManager[Session]:
        return self.db.tx(timeout=timeout)
