import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Final

import pytest
from prisma import Prisma

from litellm.integrations.shadow_eval_logger import ActiveShadowEvalJob, ShadowEvalLogger, _ShadowResponse


@pytest.mark.asyncio
async def test_shadow_failure_streak_is_durable_atomic_and_target_local():
    db: Final = Prisma()
    await db.connect()
    prisma: Final = SimpleNamespace(db=db)

    async def accounted(_key: str, _cost: float) -> None:
        return None

    logger: Final = ShadowEvalLogger(prisma_provider=lambda: prisma, job_spend_writer=accounted)
    row: Final = await db.litellm_shadowevaljob.create(data={
        "group_id": "failure-safety-integration", "target_type": "user", "target_id": "failure-safety",
        "router_name": "a", "router_names": ["a", "b"], "judge_model": "judge", "shadow_percentage": 100,
        "max_turns": 10000, "max_budget": 10, "ends_at": datetime.now(timezone.utc) + timedelta(days=1),
    })
    job: Final = ActiveShadowEvalJob.model_validate(row)

    async def record(outcome: str = "error", arm: str = "a", shadow_failed: bool = False) -> None:
        await logger._record_attempt(
            prisma, job, "req", None, router_name=arm, outcome=outcome,
            real_cost=0, real_classifier_cost=0, real_cache_hit=False,
            shadow=None if shadow_failed else _ShadowResponse("reply", "local", None, .01, .002),
            shadow_cost=.01, shadow_classifier_cost=.002, judge_cost=.003,
        )

    try:
        for _ in range(4):
            await record()
        pending: Final = await db.litellm_shadowevaljob.find_unique_or_raise(where={"id": job.id})
        assert pending.stopped_at is None
        await record("tie")
        reset: Final = await db.litellm_shadowevaljob.find_unique_or_raise(where={"id": job.id})
        assert reset.failure_counts == {"judge": 0, "shadow:a": 0}
        for _ in range(4):
            await record(arm="a", shadow_failed=True)
            await record("tie", arm="b")
        isolated: Final = await db.litellm_shadowevaljob.find_unique_or_raise(where={"id": job.id})
        assert isolated.failure_counts == {"judge": 0, "shadow:a": 4, "shadow:b": 0}
        assert isolated.stopped_at is None
        await asyncio.gather(*(record(arm="a", shadow_failed=True) for _ in range(16)))
        failed: Final = await db.litellm_shadowevaljob.find_unique_or_raise(where={"id": job.id})
        assert failed.stopped_at is not None
        assert failed.failure_counts == {"judge": 0, "shadow:a": 5, "shadow:b": 0}
        assert "shadow:a" in failed.failure_reason
        await record("tie")
        latched: Final = await db.litellm_shadowevaljob.find_unique_or_raise(where={"id": job.id})
        assert latched.failure_counts == failed.failure_counts
        assert latched.failure_reason == failed.failure_reason
        attempts: Final = await db.litellm_shadowevalattempt.find_many(where={"job_id": job.id})
        assert len(attempts) == 30
        assert sum(a.shadow_cost + a.shadow_classifier_cost + a.judge_cost for a in attempts) == pytest.approx(.45)
    finally:
        await db.litellm_shadowevalattempt.delete_many(where={"job_id": job.id})
        await db.litellm_shadowevaljob.delete(where={"id": job.id})
        await db.disconnect()
