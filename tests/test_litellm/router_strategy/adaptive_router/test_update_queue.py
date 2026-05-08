import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.router_strategy.adaptive_router.update_queue import (
    AdaptiveRouterUpdateQueue,
)


@pytest.fixture
def queue():
    return AdaptiveRouterUpdateQueue()


@pytest.fixture
def mock_prisma():
    """Prisma client with both adaptive router models stubbed as AsyncMocks."""
    p = MagicMock()
    p.db.litellm_adaptiverouterstate.find_unique = AsyncMock(return_value=None)
    p.db.litellm_adaptiverouterstate.upsert = AsyncMock()
    p.db.litellm_adaptiveroutersession.upsert = AsyncMock()
    return p


@pytest.mark.asyncio
async def test_add_state_delta_aggregates_same_key(queue):
    await queue.add_state_delta("r1", "general", "gpt-4", 1.0, 0.0)
    await queue.add_state_delta("r1", "general", "gpt-4", 0.0, 1.0)
    sizes = await queue.queue_size()
    assert sizes["state_pending"] == 1


@pytest.mark.asyncio
async def test_add_state_delta_separate_keys(queue):
    await queue.add_state_delta("r1", "general", "gpt-4", 1.0, 0.0)
    await queue.add_state_delta("r1", "writing", "gpt-4", 1.0, 0.0)
    sizes = await queue.queue_size()
    assert sizes["state_pending"] == 2


@pytest.mark.asyncio
async def test_add_session_state_last_write_wins(queue):
    await queue.add_session_state("s1", "r1", "gpt-4", {"misalignment_count": 1})
    await queue.add_session_state("s1", "r1", "gpt-4", {"misalignment_count": 5})
    sizes = await queue.queue_size()
    assert sizes["session_pending"] == 1

    flushed = []
    p = MagicMock()

    async def upsert(**kwargs):
        flushed.append(kwargs)

    p.db.litellm_adaptiveroutersession.upsert = upsert
    await queue.flush_session_to_db(p)
    assert len(flushed) == 1
    assert flushed[0]["data"]["update"]["misalignment_count"] == 5


@pytest.mark.asyncio
async def test_flush_state_drains_aggregator(queue, mock_prisma):
    await queue.add_state_delta("r1", "general", "gpt-4", 1.0, 0.0)
    await queue.add_state_delta("r1", "writing", "gpt-4", 0.0, 1.0)
    n = await queue.flush_state_to_db(mock_prisma)
    assert n == 2
    sizes = await queue.queue_size()
    assert sizes["state_pending"] == 0


@pytest.mark.asyncio
async def test_flush_state_sums_correctly(queue, mock_prisma):
    await queue.add_state_delta("r1", "general", "gpt-4", 1.0, 0.0)
    await queue.add_state_delta("r1", "general", "gpt-4", 2.0, 1.0)
    await queue.flush_state_to_db(mock_prisma)
    # find_unique returned None (cold start), so alpha = 1+2 = 3, beta = 0+1 = 1
    call = mock_prisma.db.litellm_adaptiverouterstate.upsert.call_args
    assert call.kwargs["data"]["create"]["alpha"] == 3.0
    assert call.kwargs["data"]["create"]["beta"] == 1.0
    assert call.kwargs["data"]["create"]["total_samples"] == 2


@pytest.mark.asyncio
async def test_flush_session_drains_aggregator(queue, mock_prisma):
    await queue.add_session_state("s1", "r1", "gpt-4", {"classified_type": "general"})
    n = await queue.flush_session_to_db(mock_prisma)
    assert n == 1
    sizes = await queue.queue_size()
    assert sizes["session_pending"] == 0


@pytest.mark.asyncio
async def test_flush_empty_queue_returns_zero(queue, mock_prisma):
    assert await queue.flush_state_to_db(mock_prisma) == 0
    assert await queue.flush_session_to_db(mock_prisma) == 0


@pytest.mark.asyncio
async def test_flush_state_isolation_from_concurrent_adds(queue, mock_prisma):
    """Adds during a flush should land in the NEW aggregator, not the drained batch."""
    await queue.add_state_delta("r1", "general", "gpt-4", 1.0, 0.0)
    flush_task = asyncio.create_task(queue.flush_state_to_db(mock_prisma))
    # Yield control so the flush task can swap the aggregator before we add again.
    await asyncio.sleep(0)
    await queue.add_state_delta("r1", "general", "gpt-5", 2.0, 0.0)
    await flush_task
    sizes = await queue.queue_size()
    assert sizes["state_pending"] == 1


@pytest.mark.asyncio
async def test_max_size_observability(queue):
    await queue.add_state_delta("r1", "general", "gpt-4", 1.0, 0.0)
    await queue.add_state_delta("r1", "writing", "gpt-4", 1.0, 0.0)
    await queue.add_state_delta("r1", "code_generation", "gpt-4", 1.0, 0.0)
    sizes = await queue.queue_size()
    assert sizes["max_state_seen"] >= 3
