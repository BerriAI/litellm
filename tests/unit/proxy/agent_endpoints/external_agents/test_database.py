import asyncio
from typing import Final, Literal
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.agent_endpoints.external_agents.database import ProxyDatabase
from litellm.proxy.db.prisma_client import PrismaWrapper


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["query_raw", "execute_raw"])
async def test_cancelled_queued_statement_does_not_reach_the_database(
    method: Literal["query_raw", "execute_raw"],
) -> None:
    limiter: Final = asyncio.Semaphore(0)
    db: Final = MagicMock(spec=PrismaWrapper)
    db.query_raw = AsyncMock(return_value=[])
    db.execute_raw = AsyncMock(return_value=1)
    proxy: Final = ProxyDatabase(db, limiter)
    started: Final = asyncio.Event()

    async def queued() -> object:
        started.set()
        return await getattr(proxy, method)("SELECT 1")

    task: Final = asyncio.create_task(queued())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert db.query_raw.await_count == db.execute_raw.await_count == 0
    limiter.release()
    assert await proxy.query_raw("SELECT 1") == []
    assert await proxy.execute_raw("SELECT 1") == 1
