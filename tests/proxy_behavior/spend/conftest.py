"""Session-scoped Prisma client for spend-rollup behavior tests against a real Postgres."""

import pytest_asyncio
from prisma import Prisma


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def db():
    client = Prisma()
    await client.connect()
    yield client
    await client.disconnect()
