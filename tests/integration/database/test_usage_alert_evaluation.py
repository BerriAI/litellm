import asyncio
import json
from datetime import date, datetime, timedelta, timezone
from typing import Final, cast  # noqa: TID251  # generated Prisma transaction implements the accounting protocol
from uuid import uuid4

import pytest
from integration._support.actors import ManagementDatabase
from integration.database.test_performance_issues import _isolated

from litellm.accounting.application import apply_deliveries
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
from litellm.proxy.utils import ProxyLogging
from litellm.repositories.accounting_delivery_repository import (
    AccountingDeliveryRepository,
    DeliveryProjection,
    LedgerDatabase,
    LedgerTransaction,
)
from litellm.repositories.usage_alert_health import UsageAlertFlushRecorder, UsageAlertHealthRepository
from litellm.repositories.usage_alerts import UsageAlertsRepository
from litellm.types.proxy.usage_alerts import AlertRuleInput


@pytest.mark.asyncio(loop_scope="session")
async def test_batch_checkpoints_preserve_rule_revisions_and_incomplete_status(db: ManagementDatabase) -> None:
    repository: Final = UsageAlertsRepository.from_prisma(db.prisma)
    definition: Final = AlertRuleInput(
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
    start: Final = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=2)
    async with _isolated(db) as prisma:
        repository: Final = UsageAlertHealthRepository.from_prisma(prisma)
        assert await repository.record("user", start)
        assert (await repository.coverage("user", start)).hours == 1
        assert not (await repository.coverage("user", start)).after_grace
        assert await repository.record("user", start + timedelta(days=1, hours=1))
        coverage: Final = await repository.coverage("user", start)
        assert coverage.hours == 1 and coverage.after_grace


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


@pytest.mark.asyncio(loop_scope="session")
async def test_freshness_waits_until_accounting_has_committed_the_daily_totals(db: ManagementDatabase) -> None:
    async with _isolated(db) as prisma:
        await prisma.writer_db.execute_raw(
            'CREATE TABLE "LiteLLM_DailyModelUsage" (LIKE public."LiteLLM_DailyModelUsage" INCLUDING ALL)'
        )
        identity: Final = "coverage-" + uuid4().hex
        now: Final = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) + timedelta(days=3)
        row: Final = {
            "request_id": identity,
            "user": identity,
            "api_key": "synthetic-key",
            "startTime": (now - timedelta(hours=2)).isoformat(),
            "model": "example-model",
            "model_group": "example-model",
            "custom_llm_provider": "example-provider",
            "spend": 5,
            "metadata": {},
        }
        health: Final = UsageAlertHealthRepository.from_prisma(prisma)
        writer: Final = DBSpendUpdateWriter()
        logging: Final = ProxyLogging(UserApiKeyCache())

        async def apply(transaction: LedgerTransaction, deliveries: tuple[DeliveryProjection, ...]) -> None:
            await apply_deliveries(
                transaction, deliveries, prisma_client=prisma, writer=writer, proxy_logging_obj=logging
            )
            assert not await health.record("user", now)

        try:
            await prisma.writer_db.query_raw(
                'INSERT INTO "LiteLLM_SpendLogDelivery" (request_id,accounting_payload,created_at,is_batch) VALUES ($1,$2::jsonb,$3::timestamp,false) RETURNING delivery_id',
                identity,
                json.dumps(row),
                (now - timedelta(hours=1)).replace(tzinfo=None),
            )
            assert not await health.record("user", now)
            database: Final = cast(LedgerDatabase, prisma)  # cast-ok: generated Prisma transaction delegate
            result: Final = await AccountingDeliveryRepository(database).apply_pending(apply, batch_size=1)
            assert result.applied == 1
            assert await health.record("user", now)
            totals: Final = await prisma.writer_db.query_raw(
                'SELECT spend,api_requests FROM "LiteLLM_DailyUserSpend" WHERE user_id=$1', identity
            )
            assert totals == [{"spend": 5, "api_requests": 1}]
        finally:
            await prisma.writer_db.query_raw(
                'DELETE FROM "LiteLLM_SpendLogDelivery" WHERE request_id=$1 RETURNING delivery_id', identity
            )
            await prisma.writer_db.query_raw(
                'DELETE FROM "LiteLLM_DailyUserSpend" WHERE user_id=$1 RETURNING id', identity
            )
            await prisma.writer_db.query_raw(
                "DELETE FROM \"LiteLLM_UsageAlertFlush\" WHERE source='user' AND hour=$1::timestamp RETURNING source",
                now.replace(tzinfo=None),
            )
