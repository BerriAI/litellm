"""Session-scoped Prisma client for spend-rollup behavior tests against a real Postgres."""

import asyncio
from collections.abc import AsyncIterator
from typing import Final

import pytest_asyncio
from litellm_proxy_extras.utils import ProxyExtrasDBManager
from prisma import Prisma


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def db() -> AsyncIterator[Prisma]:
    await asyncio.to_thread(ProxyExtrasDBManager.apply_autorouter_daily_coverage)
    client: Final = Prisma()
    await client.connect()
    try:
        yield client
    finally:
        await client.disconnect()
