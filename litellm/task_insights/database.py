from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import (
    Final,
    Protocol,
    cast,  # noqa: TID251  # the generated Prisma client is untyped and cannot express the raw-query protocol
)

from litellm.repositories.task_insights_repository import TaskInsightsDatabase


class _Connection(Protocol):
    async def connect(self) -> None: ...

    async def disconnect(self) -> None: ...


def _prisma() -> _Connection:
    from prisma import Prisma

    return Prisma()


@asynccontextmanager
async def connected_database(
    client: Callable[[], _Connection] = _prisma,
) -> AsyncGenerator[TaskInsightsDatabase, None]:
    """The worker and seed processes own one Prisma connection each, opened from DATABASE_URL and closed on exit."""
    prisma: Final = client()
    await prisma.connect()
    try:
        yield cast(  # cast-ok: Prisma exposes query_raw, execute_raw and tx with these signatures
            TaskInsightsDatabase, prisma
        )
    finally:
        await prisma.disconnect()
