import asyncio
from datetime import date, datetime, timedelta, timezone
from typing import Final

import pytest
from integration._support.actors import ManagementDatabase

from litellm.repositories.usage_alert_health import UsageAlertFlushRecorder, UsageAlertHealthRepository
from litellm.repositories.usage_alerts import UsageAlertsRepository
from litellm.types.proxy.usage_alerts import UsageAlertRuleInput


@pytest.mark.asyncio(loop_scope="session")
async def test_batch_checkpoints_preserve_rule_revisions_and_incomplete_status(db: ManagementDatabase) -> None:
    repository: Final = UsageAlertsRepository.from_prisma(db.prisma)
    definition: Final = UsageAlertRuleInput(
        name="Daily evaluation", scope_type="team", scope_id="team-example", metric="spend", direction="increase"
    )
    rule: Final = await repository.save_rule(definition, "qa-admin")
    assert rule is not None
    day: Final = date(2026, 11, 4)
    try:
        assert not await repository.completed_rule_ids((rule,), day)
        await repository.record_evaluations(((rule, "complete"),), day, completed=True)
        assert await repository.completed_rule_ids((rule,), day) == frozenset({rule.id})
        edited: Final = await repository.save_rule(definition.model_copy(update={"multiplier": 4}), "qa-admin", rule)
        assert edited is not None
        assert not await repository.completed_rule_ids((edited,), day)
        await repository.record_evaluations(((rule, "insufficient_coverage"),), day, completed=False)
        assert await repository.get_rule(rule.id) == edited
        await repository.record_evaluations(((edited, "insufficient_coverage"),), day, completed=False)
        incomplete: Final = await repository.get_rule(rule.id)
        assert incomplete is not None and incomplete.evaluation_status == "insufficient_coverage"
        assert incomplete.last_evaluated_day == day
        assert not await repository.completed_rule_ids((edited,), day)
    finally:
        await repository.delete_rule(rule.id)


@pytest.mark.asyncio(loop_scope="session")
async def test_freshness_timestamps_use_the_real_prisma_parameter_types(db: ManagementDatabase) -> None:
    repository: Final = UsageAlertHealthRepository.from_prisma(db.prisma)
    start: Final = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=2)
    try:
        assert await repository.record("user", start)
        assert (await repository.coverage("user", start)).hours == 1
        assert not (await repository.coverage("user", start)).after_grace
        assert await repository.record("user", start + timedelta(days=1, hours=1))
        coverage: Final = await repository.coverage("user", start)
        assert coverage.hours == 1 and coverage.after_grace
    finally:
        await db.prisma.writer_db.query_raw(
            "DELETE FROM \"LiteLLM_UsageAlertFlush\" WHERE source='user' AND hour IN ($1::timestamp,$2::timestamp) RETURNING source",
            start.replace(tzinfo=None),
            (start + timedelta(days=1, hours=1)).replace(tzinfo=None),
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_freshness_lock_cannot_stall_committed_spend(db: ManagementDatabase) -> None:
    recorder: Final = UsageAlertFlushRecorder()
    async with db.prisma.writer_db.tx(timeout=timedelta(seconds=10)) as transaction:
        await transaction.execute_raw('LOCK TABLE "LiteLLM_UsageAlertFlush" IN ACCESS EXCLUSIVE MODE')
        await asyncio.wait_for(recorder.record(db.prisma, "user"), timeout=2)
        await asyncio.wait_for(recorder.record(db.prisma, "user"), timeout=0.1)
    await UsageAlertFlushRecorder().record(db.prisma, "user")
    rows: Final = await db.prisma.writer_db.query_raw(
        "SELECT source FROM \"LiteLLM_UsageAlertFlush\" WHERE source='user' AND hour=date_trunc('hour',CURRENT_TIMESTAMP AT TIME ZONE 'UTC')"
    )
    assert rows == [{"source": "user"}]
