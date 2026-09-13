from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.db import shadow_eval_funnel
from litellm.proxy.db.shadow_eval_funnel import (
    flush_shadow_eval_funnel,
    record_shadow_eval_funnel_event,
)


@pytest.fixture(autouse=True)
def _clean_queue():
    shadow_eval_funnel._pending.clear()
    yield
    shadow_eval_funnel._pending.clear()


def _prisma() -> MagicMock:
    prisma = MagicMock()
    prisma.db.execute_raw = AsyncMock(return_value=1)
    return prisma


@pytest.mark.asyncio
async def test_increments_aggregate_per_job_and_flush_upserts_and_clears():
    record_shadow_eval_funnel_event("leg-1", "not_sampled")
    record_shadow_eval_funnel_event("leg-1", "not_sampled")
    record_shadow_eval_funnel_event("leg-1", "shed")
    record_shadow_eval_funnel_event("leg-2", "unjudgeable")
    prisma = _prisma()

    await flush_shadow_eval_funnel(prisma)

    calls = {call.args[1]: call.args[2:] for call in prisma.db.execute_raw.await_args_list}
    assert calls == {"leg-1": (2, 0, 1, 0), "leg-2": (0, 1, 0, 0)}
    sql = prisma.db.execute_raw.await_args_list[0].args[0]
    assert "ON CONFLICT (job_id) DO UPDATE" in sql
    assert '"LiteLLM_ShadowEvalFunnel".not_sampled + EXCLUDED.not_sampled' in sql
    assert shadow_eval_funnel._pending == {}


@pytest.mark.asyncio
async def test_empty_queue_touches_nothing():
    prisma = _prisma()

    await flush_shadow_eval_funnel(prisma)

    prisma.db.execute_raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_upsert_drops_only_that_legs_batch():
    record_shadow_eval_funnel_event("leg-bad", "not_sampled")
    record_shadow_eval_funnel_event("leg-good", "shed")
    prisma = _prisma()

    async def execute_raw(sql, job_id, *counts):
        if job_id == "leg-bad":
            raise RuntimeError("db down")
        return 1

    prisma.db.execute_raw = AsyncMock(side_effect=execute_raw)

    await flush_shadow_eval_funnel(prisma)

    flushed = [call.args[1] for call in prisma.db.execute_raw.await_args_list]
    assert set(flushed) == {"leg-bad", "leg-good"}
    assert shadow_eval_funnel._pending == {}


@pytest.mark.asyncio
async def test_events_recorded_during_a_flush_survive_into_the_next_batch():
    record_shadow_eval_funnel_event("leg-1", "not_sampled")
    prisma = _prisma()

    async def execute_raw(sql, job_id, *counts):
        record_shadow_eval_funnel_event("leg-2", "shed")
        return 1

    prisma.db.execute_raw = AsyncMock(side_effect=execute_raw)

    await flush_shadow_eval_funnel(prisma)

    assert shadow_eval_funnel._pending == {"leg-2": {"not_sampled": 0, "unjudgeable": 0, "shed": 1, "withheld": 0}}


def test_pending_count_feeds_the_drain_census():
    from litellm.proxy.db.shadow_eval_funnel import pending_shadow_eval_funnel_events

    assert pending_shadow_eval_funnel_events() == 0
    record_shadow_eval_funnel_event("leg-1", "not_sampled")
    record_shadow_eval_funnel_event("leg-1", "shed")
    record_shadow_eval_funnel_event("leg-2", "unjudgeable")
    assert pending_shadow_eval_funnel_events() == 3
