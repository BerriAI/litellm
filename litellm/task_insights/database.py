from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import (
    Final,
    cast,  # noqa: TID251  # the generated Prisma client is untyped and cannot express the raw-query protocol
)

from litellm.repositories.task_insights_repository import TaskInsightsDatabase


@asynccontextmanager
async def connected_database() -> AsyncIterator[TaskInsightsDatabase]:
    """The worker and seed processes own one Prisma connection each, opened from DATABASE_URL and closed on exit."""
    from prisma import Prisma

    prisma: Final = Prisma()
    await prisma.connect()
    try:
        yield cast(  # cast-ok: Prisma exposes query_raw, execute_raw and tx with these signatures
            TaskInsightsDatabase, prisma
        )
    finally:
        await prisma.disconnect()
