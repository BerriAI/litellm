"""A real Postgres connection for the rollup's SQL.

The auto-router rollup classifies each turn inside its upsert, against the row's own
stored state, so the classification only exists when a real database evaluates it.
"""

from dataclasses import dataclass

import pytest_asyncio
from prisma import Prisma


@dataclass(frozen=True)
class PrismaClientShim:
    """What the rollup writer needs from litellm's PrismaClient: a connected `db`."""

    db: Prisma


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def prisma_db():
    db = Prisma()
    await db.connect()
    try:
        yield db
    finally:
        await db.disconnect()


@pytest_asyncio.fixture(loop_scope="session")
async def rollup_client(prisma_db):
    await prisma_db.execute_raw('DELETE FROM "LiteLLM_AutoRouterSession"')
    try:
        yield PrismaClientShim(db=prisma_db)
    finally:
        await prisma_db.execute_raw('DELETE FROM "LiteLLM_AutoRouterSession"')
