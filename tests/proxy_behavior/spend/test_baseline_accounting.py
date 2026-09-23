import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Final

import pytest
from prisma import Prisma

import litellm
from litellm.llms.anthropic.prompt_cache_prediction import CountedBreakpoint, CountedPromptCachePlan
from litellm.proxy.db.autorouter_session_rollup import AutoRouterTurnTransaction
from litellm.proxy.db.baseline_accounting import (
    BaselineAccountingRecord,
    BaselineAccountingStore,
    BaselinePublication,
    DailyBaselineAttribution,
    DailyBaselineTarget,
)
from litellm.proxy.db.create_views import SupportsRawQueries
from litellm.proxy.db.daily_spend_bulk_upsert import DAILY_SPEND_TABLES, build_bulk_upsert, merge_by_conflict_key
from litellm.proxy.spend_tracking.baseline_accounting import BaselineObservation
from litellm.proxy.spend_tracking.savings import BaselineCostSnapshot
from litellm.types.utils import Usage

pytestmark = pytest.mark.asyncio(loop_scope="session")


@asynccontextmanager
async def _transaction(db: Prisma, *, before_commit: bool = False, after_commit: bool = False) -> AsyncIterator[SupportsRawQueries]:
    async with db.tx() as tx:
        yield tx
        if before_commit:
            raise RuntimeError("injected pre-commit interruption")
    if after_commit:
        raise RuntimeError("injected lost commit acknowledgement")


def _store(db: Prisma, **faults: bool) -> BaselineAccountingStore:
    def transaction():
        return _transaction(db, **faults)

    return BaselineAccountingStore(transaction)


@pytest.fixture
def record() -> Callable[..., BaselineAccountingRecord]:
    run: Final = uuid.uuid4().hex
    marker: Final = CountedBreakpoint("prefix", 3600, 6000, ("prefix",), "content", ("content",))
    usage: Final = Usage(
        prompt_tokens=6200, completion_tokens=30, total_tokens=6230,
        cache_creation_input_tokens=6000, cache_read_input_tokens=0,
        prompt_tokens_details={
            "text_tokens": 200, "cached_tokens": 0, "cache_creation_tokens": 6000,
            "cache_creation_token_details": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 6000},
        },
    )

    def create(
        label: str = "first", started: float = 10000.0, identical: bool = True, user_id: str = ""
    ) -> BaselineAccountingRecord:
        return BaselineAccountingRecord(
            scope="autorouter-baseline:v3:" + run * 2, api_key=run, session_id=run,
            router_name="test-router", baseline_model="anthropic/claude-opus-5",
            observation=BaselineObservation(
                request_id=run + label, started_at=started, available_at=started + 0.1,
                outcome="complete", baseline_equivalent=identical, usage=usage,
                plan=CountedPromptCachePlan(6200, (marker,)), minimum_cache_tokens=4096,
            ),
            pricing=BaselineCostSnapshot(
                model="claude-opus-5", provider="anthropic",
                prices=litellm.get_model_info("claude-opus-5", custom_llm_provider="anthropic"),
                actual_spend=0.17, actual_token_cost=0.17,
            ),
            turn=AutoRouterTurnTransaction(
                api_key=run, session_id=run, router_name="test-router", router_type="heuristic",
                model="claude-opus-5", turn_at=datetime.fromtimestamp(started, timezone.utc),
                total_tokens=6230, spend=0.17, saved_spend=0.0, classifier_cost=0.0,
                covered=True, cache_hit=False, cache_ttl_seconds=3600, cache_touched=True,
                baseline_model="anthropic/claude-opus-5",
                user_id=user_id,
            ),
            daily=DailyBaselineAttribution(
                date="2026-09-15", api_key=run, model="claude-opus-5", custom_llm_provider="anthropic",
                targets=tuple(DailyBaselineTarget(entity=entity, entity_id=run) for entity in ("user", "team", "org", "end_user", "agent", "tag")),
            ),
        )

    return create


async def _log(db: Prisma, record: BaselineAccountingRecord) -> None:
    await db.execute_raw(
        'INSERT INTO "LiteLLM_SpendLogs" (request_id,call_type,api_key,spend,"startTime","endTime") '
        "VALUES ($1, 'anthropic_messages', $2, 0.17, to_timestamp($3::float8), to_timestamp($3::float8))",
        record.observation.request_id, record.api_key, record.observation.started_at,
    )


async def _session(db: Prisma, record: BaselineAccountingRecord):
    rows: Final = await db.query_raw('SELECT * FROM "LiteLLM_AutoRouterSession" WHERE api_key=$1', record.api_key)
    return rows[0]


async def _user_sessions(db: Prisma, record: BaselineAccountingRecord) -> dict[str, dict[str, object]]:
    rows: Final = await db.query_raw('SELECT * FROM "LiteLLM_AutoRouterUserSession" WHERE api_key=$1', record.api_key)
    return {str(row["user_id"]): row for row in rows}


async def _daily_user(db: Prisma, record: BaselineAccountingRecord) -> dict[str, object]:
    rows: Final = await db.query_raw('SELECT * FROM "LiteLLM_DailyUserSpend" WHERE api_key=$1', record.api_key)
    return rows[0]


async def test_late_replay_updates_all_projections_without_rebilling(db: Prisma, record: Callable[..., BaselineAccountingRecord]) -> None:
    store: Final = _store(db)
    late: Final = record("late", 10001.0, user_id="late-user")
    early: Final = record("early", identical=False, user_id="early-user")
    await _log(db, late)
    assert await store.append(late) == "recorded"
    assert await store.project(late.scope) == "published"
    before: Final = await _session(db, late)
    assert before["savings_estimated_actual_spend"] == before["spend"] == 0.17
    assert before["saved_spend"] == 0.0
    before_daily: Final = await _daily_user(db, late)
    assert before_daily["autorouter_estimated_requests"] == 1
    assert before_daily["autorouter_estimated_actual_spend"] == before["spend"]
    assert before_daily["autorouter_savings_spend"] == 0.0
    before_users: Final = await _user_sessions(db, late)
    assert set(before_users) == {"late-user"}
    assert before_users["late-user"]["savings_estimated_turns"] == 1
    assert before_users["late-user"]["savings_estimated_baseline_models"] == {late.baseline_model: 1}
    await _log(db, early)
    assert await store.append(early) == "recorded"
    pending: Final = await _session(db, late)
    assert pending["spend"] == 0.34 and pending["savings_estimated_turns"] == 0
    assert pending["saved_spend"] == pending["savings_estimated_actual_spend"] == 0.0
    pending_daily: Final = await _daily_user(db, late)
    assert pending_daily["autorouter_estimated_requests"] == pending_daily["autorouter_estimated_actual_spend"] == 0
    pending_users: Final = await _user_sessions(db, late)
    assert set(pending_users) == {"late-user", "early-user"}
    for user in pending_users.values():
        assert user["turns"] == 1 and user["spend"] == 0.17
        assert user["savings_estimated_turns"] == user["savings_estimated_actual_spend"] == user["saved_spend"] == 0
        assert user["savings_estimated_baseline_models"] == {}
    waiting: Final = await db.query_raw('SELECT metadata FROM "LiteLLM_SpendLogs" WHERE request_id=$1', late.observation.request_id)
    assert waiting[0]["metadata"]["autorouter_savings"] is None
    assert waiting[0]["metadata"]["autorouter_savings_estimate"]["reason"] == "pending_projection"
    assert await store.project(early.scope) == "published"
    after: Final = await _session(db, late)
    assert after["spend"] == 0.34 and after["turns"] == 2
    assert after["savings_estimated_actual_spend"] == 0.17 and after["savings_estimated_turns"] == 1
    logs: Final = await db.query_raw('SELECT spend, metadata FROM "LiteLLM_SpendLogs" WHERE request_id=$1', late.observation.request_id)
    assert logs[0]["spend"] == 0.17
    assert logs[0]["metadata"]["autorouter_savings_estimate"]["provenance"] == "modeled"
    assert after["saved_spend"] == pytest.approx(logs[0]["metadata"]["autorouter_savings"])
    after_daily: Final = await _daily_user(db, late)
    assert after_daily["autorouter_estimated_requests"] == after["savings_estimated_turns"]
    assert after_daily["autorouter_estimated_actual_spend"] == after["savings_estimated_actual_spend"]
    for field in (
        "api_requests", "autorouter_accounted_requests", "autorouter_requests", "autorouter_llm_spend",
        "autorouter_classifier_cost", "autorouter_classifier_cost_recorded_requests",
    ):
        assert after_daily[field] == 0
    after_users: Final = await _user_sessions(db, late)
    assert after_users["early-user"] == pending_users["early-user"]
    for field in (
        "saved_spend", "savings_estimated_turns", "savings_estimated_actual_spend",
        "savings_estimated_saved_spend", "savings_estimated_baseline_models",
    ):
        assert after_users["late-user"][field] == after[field]
    assert after_users["late-user"]["turns"] == 1 and after_users["late-user"]["spend"] == 0.17
    for table in ("DailyUserSpend", "DailyTeamSpend", "DailyOrganizationSpend", "DailyEndUserSpend", "DailyAgentSpend", "DailyTagSpend"):
        rows: Final = await db.query_raw(f'SELECT spend,api_requests,autorouter_savings_spend FROM "LiteLLM_{table}" WHERE api_key=$1', late.api_key)
        assert rows[0]["spend"] == rows[0]["api_requests"] == 0
        assert rows[0]["autorouter_savings_spend"] == pytest.approx(after["saved_spend"])


@pytest.mark.parametrize("attributed", [True, False])
async def test_commit_ack_loss_and_concurrent_duplicate_delivery_are_idempotent(
    db: Prisma, record: Callable[..., BaselineAccountingRecord], attributed: bool
) -> None:
    event: Final = record(user_id="first-user" if attributed else "")
    other: Final = record("other", 10001.0, user_id="second-user" if attributed else "")
    await _log(db, event)
    assert await _store(db, after_commit=True).append(event) == "unavailable"
    store: Final = _store(db)
    assert set(await asyncio.gather(*(store.append(event) for _ in range(4)))) == {"recorded"}
    await _log(db, other)
    assert await store.append(other) == "recorded"
    if not attributed:
        await db.execute_raw(
            'UPDATE "LiteLLM_AutoRouterBaselineObservation" SET data=(data::jsonb #- \'{turn,user_id}\')::text WHERE scope=$1',
            event.scope,
        )
    assert await store.project(event.scope) == "published"
    assert await store.project(event.scope) == "unchanged"
    session: Final = await _session(db, event)
    assert session["turns"] == session["savings_estimated_turns"] == 2
    assert session["spend"] == session["savings_estimated_actual_spend"] == 0.34
    daily: Final = await _daily_user(db, event)
    assert daily["autorouter_estimated_requests"] == session["savings_estimated_turns"]
    assert daily["autorouter_estimated_actual_spend"] == session["savings_estimated_actual_spend"]
    assert daily["autorouter_accounted_requests"] == daily["autorouter_requests"] == 0
    users: Final = await _user_sessions(db, event)
    assert set(users) == ({"first-user", "second-user"} if attributed else set())
    for user in users.values():
        assert user["turns"] == user["savings_estimated_turns"] == 1
        assert user["spend"] == user["savings_estimated_actual_spend"] == 0.17
        assert user["savings_estimated_baseline_models"] == {event.baseline_model: 1}


async def test_publication_rollback_keeps_dirty_revision_for_retry(db: Prisma, record: Callable[..., BaselineAccountingRecord]) -> None:
    event: Final = record(user_id="rollback-user")
    await _log(db, event)
    store: Final = _store(db)
    assert await store.append(event) == "recorded"
    assert await _store(db, before_commit=True).project(event.scope) == "unavailable"
    session: Final = await _session(db, event)
    assert session["spend"] == 0.17 and session["savings_estimated_turns"] == 0
    before_users: Final = await _user_sessions(db, event)
    assert before_users["rollback-user"]["spend"] == 0.17
    assert before_users["rollback-user"]["savings_estimated_turns"] == 0
    assert before_users["rollback-user"]["savings_estimated_baseline_models"] == {}
    revisions: Final = await db.query_raw('SELECT revision,published_revision FROM "LiteLLM_AutoRouterBaselineComparison" WHERE scope=$1', event.scope)
    assert revisions[0]["revision"] > revisions[0]["published_revision"]
    assert await store.project(event.scope) == "published"
    assert (await _session(db, event))["savings_estimated_turns"] == 1
    after_users: Final = await _user_sessions(db, event)
    assert after_users["rollback-user"]["turns"] == after_users["rollback-user"]["savings_estimated_turns"] == 1
    assert after_users["rollback-user"]["spend"] == after_users["rollback-user"]["savings_estimated_actual_spend"] == 0.17


async def test_conflicting_duplicate_cannot_restore_an_observed_estimate(db: Prisma, record: Callable[..., BaselineAccountingRecord]) -> None:
    event: Final = record()
    await _log(db, event)
    store: Final = _store(db)
    assert await store.append(event) == "recorded"
    assert await store.project(event.scope) == "published"
    conflict: Final = event.model_copy(update={"observation": event.observation.model_copy(update={"baseline_equivalent": False, "started_at": 20000.0, "available_at": 20001.0})})
    assert await store.append(conflict) == "recorded"
    assert (await _session(db, event))["savings_estimated_turns"] == 0
    assert await store.append(event) == "recorded"
    assert await store.project(event.scope) == "published"
    session: Final = await _session(db, event)
    assert session["turns"] == 1 and session["savings_estimated_turns"] == 0
    rows: Final = await db.query_raw('SELECT publication FROM "LiteLLM_AutoRouterBaselineObservation" WHERE request_id=$1', event.observation.request_id)
    assert json.loads(rows[0]["publication"])["reason"] == "conflicting_observation"


async def test_retired_history_never_recreates_an_initial_zero(db: Prisma, record: Callable[..., BaselineAccountingRecord]) -> None:
    original: Final = record()
    await _log(db, original)
    store: Final = _store(db)
    assert await store.append(original) == "recorded"
    assert await store.project(original.scope) == "published"
    await db.execute_raw('UPDATE "LiteLLM_AutoRouterBaselineComparison" SET updated_at=to_timestamp(0) WHERE scope=$1', original.scope)
    await store.retire_before(datetime(2000, 1, 1, tzinfo=timezone.utc), 1000, 1000)
    next_turn: Final = record("after-retention", 20000.0)
    await _log(db, next_turn)
    assert await store.append(next_turn) == "retired"
    assert await store.project(original.scope) == "unchanged"
    after: Final = await _session(db, original)
    assert after["turns"] == 2 and after["spend"] == 0.34
    assert after["savings_estimated_turns"] == 1 and after["savings_estimated_actual_spend"] == 0.17


async def test_native_observation_enters_spend_pipeline_once_with_shared_daily_attribution(
    db: Prisma, record: Callable[..., BaselineAccountingRecord], monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
    from litellm.proxy.hooks.autorouter_baseline_cache import CapturedBaselineObservation
    from litellm.proxy.utils import PrismaClient, ProxyLogging

    event: Final = record("routed", identical=False)
    capture: Final = CapturedBaselineObservation(
        scope=event.scope, api_key=event.api_key, session_id=event.session_id,
        router_name=event.router_name, baseline_model=event.baseline_model,
        model=event.pricing.model, prices=event.pricing.prices, observation=event.observation,
    )
    metadata: Final = {
        "routing_decision": {"router_model_name": event.router_name, "savings_baseline_model": event.baseline_model},
        "usage_object": event.observation.usage.model_dump(),
        "cost_breakdown": {"input_cost": 0.16, "output_cost": 0.01},
        "autorouter_savings": None, "autorouter_savings_estimate": {"version": 3, "status": "unknown", "reason": "pending_projection"},
        "autorouter_baseline_observation": capture.model_dump_json(),
    }
    payload: Final = {
        "request_id": event.observation.request_id, "api_key": event.api_key, "session_id": event.session_id,
        "startTime": datetime.fromtimestamp(event.observation.started_at, timezone.utc).isoformat(),
        "endTime": datetime.fromtimestamp(event.observation.available_at, timezone.utc).isoformat(),
        "spend": 0.17, "prompt_tokens": 6200, "completion_tokens": 30, "model": event.pricing.model,
        "model_group": event.router_name, "model_id": "baseline", "custom_llm_provider": "anthropic",
        "call_type": "anthropic_messages", "status": "success", "metadata": json.dumps(metadata),
        "user": None, "team_id": "", "organization_id": "org", "agent_id": None,
        "end_user": "", "request_tags": '["tag","tag"]',
    }
    monkeypatch.delenv("DATABASE_URL_READ_REPLICA", raising=False)
    client: Final = PrismaClient(os.environ["DATABASE_URL"], ProxyLogging(UserApiKeyCache()))
    writer: Final = DBSpendUpdateWriter()
    try:
        await client.db.connect()
        await _log(db, event)
        await writer._enqueue_autorouter_turn_transaction(payload, client)
        assert len(client.baseline_accounting_transactions) == 1
        queued: Final = client.baseline_accounting_transactions[0]
        assert queued.daily is not None
        assert [(target.entity, target.entity_id) for target in queued.daily.targets] == [
            ("user", None), ("team", ""), ("org", "org"), ("tag", "tag"),
        ]
        await writer.add_spend_log_transaction_to_daily_tag_transaction(payload, client)
        actual_tags: Final = await writer.daily_tag_spend_update_queue.flush_and_get_aggregated_daily_spend_update_transactions()
        assert len(actual_tags) == 1
        assert next(iter(actual_tags.values()))["spend"] == 0.17
        durable: Final = BaselineAccountingStore.for_client(client)
        anchor: Final = record("anchor", 9999.0)
        await _log(db, anchor)
        assert await durable.append(anchor) == "recorded"
        assert await durable.append(queued) == "recorded"
        assert await durable.append(queued) == "recorded"
        assert await durable.project(queued.scope) == "published"
        session: Final = await _session(db, queued)
        assert session["turns"] == session["savings_estimated_turns"] == 2
        assert session["spend"] == session["savings_estimated_actual_spend"] == 0.34
        tag_rows: Final = await db.query_raw(
            'SELECT spend, api_requests, autorouter_savings_spend FROM "LiteLLM_DailyTagSpend" WHERE api_key=$1 AND tag=$2',
            queued.api_key, "tag",
        )
        assert session["saved_spend"] < 0
        assert len(tag_rows) == 1
        assert tag_rows[0]["autorouter_savings_spend"] == pytest.approx(session["saved_spend"])
        assert tag_rows[0]["spend"] == tag_rows[0]["api_requests"] == 0
    finally:
        await client.db.disconnect()


async def test_old_publisher_transitions_keep_daily_comparison_complete_after_retention(
    db: Prisma, record: Callable[..., BaselineAccountingRecord]
) -> None:
    event: Final = record("old-publisher")
    assert event.daily is not None
    assert await _store(db).append(event) == "recorded"
    existing_actual: Final = 5.0
    other_savings: Final = 3.0
    actual: Final = event.pricing.actual_spend
    row: Final = {
        **event.daily.model_dump(exclude={"targets"}),
        "user_id": event.api_key,
        "spend": existing_actual + actual,
        "api_requests": 2,
        "successful_requests": 2,
        "autorouter_accounted_requests": 2,
        "autorouter_requests": 2,
        "autorouter_llm_spend": existing_actual + actual,
        "autorouter_classifier_cost_recorded_requests": 2,
        "autorouter_estimated_requests": 1,
        "autorouter_estimated_actual_spend": existing_actual,
        "autorouter_savings_spend": other_savings,
    }
    table: Final = DAILY_SPEND_TABLES["user"]
    statement, values = build_bulk_upsert(table, merge_by_conflict_key(table, (row,)))
    await db.execute_raw(statement, *values)

    for status, baseline, savings_delta, expected_savings, count, covered_actual in (
        ("estimated", actual + 0.5, 0.5, other_savings + 0.5, 2, existing_actual + actual),
        ("estimated", actual, -0.5, other_savings, 2, existing_actual + actual),
        ("unknown", None, 0.0, other_savings, 1, existing_actual),
        ("estimated", actual + 0.2, 0.2, other_savings + 0.2, 2, existing_actual + actual),
        ("estimated", actual + 0.2, 0.0, other_savings + 0.2, 2, existing_actual + actual),
    ):
        publication: Final = BaselinePublication(
            comparison_id=event.scope,
            comparison_started_at=event.observation.started_at,
            status=status,
            reason="legacy-publisher-transition",
            actual_spend=actual if baseline is not None else None,
            baseline_spend=baseline,
        )
        async with db.tx() as tx:
            if savings_delta:
                await tx.execute_raw(
                    'UPDATE "LiteLLM_DailyUserSpend" SET autorouter_savings_spend=autorouter_savings_spend+$1::float8 '
                    "WHERE api_key=$2 AND user_id=$3",
                    savings_delta,
                    event.api_key,
                    event.api_key,
                )
            await tx.execute_raw(
                'UPDATE "LiteLLM_AutoRouterBaselineObservation" SET publication=$1 WHERE request_id=$2',
                publication.model_dump_json(),
                event.observation.request_id,
            )
        daily: Final = await _daily_user(db, event)
        assert daily["autorouter_estimated_requests"] == count
        assert daily["autorouter_estimated_actual_spend"] == pytest.approx(covered_actual)
        assert daily["autorouter_savings_spend"] == pytest.approx(expected_savings)
        assert daily["api_requests"] == daily["autorouter_accounted_requests"] == daily["autorouter_requests"] == 2
        assert daily["spend"] == daily["autorouter_llm_spend"] == pytest.approx(existing_actual + actual)
        assert daily["autorouter_classifier_cost"] == 0
        assert daily["autorouter_classifier_cost_recorded_requests"] == 2
        shadow: Final = await db.query_raw(
            "SELECT publication::jsonb = daily_costs_publication::jsonb AS accounted "
            'FROM "LiteLLM_AutoRouterBaselineObservation" WHERE request_id=$1',
            event.observation.request_id,
        )
        assert shadow == [{"accounted": True}]

    retained: Final = await _daily_user(db, event)
    await db.execute_raw(
        'DELETE FROM "LiteLLM_AutoRouterBaselineObservation" WHERE request_id=$1', event.observation.request_id
    )
    assert await _daily_user(db, event) == retained


@pytest.mark.parametrize("targets", ("missing", "duplicate", "null-and-empty"))
async def test_legacy_zero_publication_repairs_shadow_once_for_normalized_user_targets(
    db: Prisma, record: Callable[..., BaselineAccountingRecord], targets: str
) -> None:
    source: Final = record("legacy-zero")
    assert source.daily is not None
    identities: Final = (
        () if targets == "missing" else ((None, "") if targets == "null-and-empty" else (source.api_key,) * 2)
    )
    event: Final = source.model_copy(
        update={
            "daily": source.daily.model_copy(
                update={
                    "targets": tuple(DailyBaselineTarget(entity="user", entity_id=identity) for identity in identities),
                }
            )
        }
    )
    publication: Final = BaselinePublication(
        comparison_id=event.scope,
        comparison_started_at=event.observation.started_at,
        status="estimated",
        reason="legacy-zero",
        actual_spend=event.pricing.actual_spend,
        baseline_spend=event.pricing.actual_spend,
    ).model_dump_json()
    await db.execute_raw(
        'INSERT INTO "LiteLLM_AutoRouterBaselineObservation" (request_id,scope,started_at,revision,data,publication) '
        "VALUES ($1,$2,$3::float8,1,$4,$5)",
        event.observation.request_id,
        event.scope,
        event.observation.started_at,
        event.model_dump_json(),
        publication,
    )
    before: Final = await db.query_raw(
        'SELECT daily_costs_publication FROM "LiteLLM_AutoRouterBaselineObservation" WHERE request_id=$1',
        event.observation.request_id,
    )
    assert before == [{"daily_costs_publication": None}]
    for _ in range(2):
        await db.execute_raw(
            'UPDATE "LiteLLM_AutoRouterBaselineObservation" SET publication=$1 WHERE request_id=$2',
            publication,
            event.observation.request_id,
        )
        rows: Final = await db.query_raw(
            "SELECT user_id,autorouter_estimated_requests,autorouter_estimated_actual_spend,autorouter_savings_spend, "
            'api_requests,autorouter_requests,spend FROM "LiteLLM_DailyUserSpend" WHERE api_key=$1',
            event.api_key,
        )
        assert rows == (
            [
                {
                    "user_id": "" if targets == "null-and-empty" else event.api_key,
                    "autorouter_estimated_requests": 1,
                    "autorouter_estimated_actual_spend": event.pricing.actual_spend,
                    "autorouter_savings_spend": 0,
                    "api_requests": 0,
                    "autorouter_requests": 0,
                    "spend": 0,
                }
            ]
            if identities
            else []
        )
    shadow: Final = await db.query_raw(
        "SELECT publication::jsonb = daily_costs_publication::jsonb AS accounted "
        'FROM "LiteLLM_AutoRouterBaselineObservation" WHERE request_id=$1',
        event.observation.request_id,
    )
    assert shadow == [{"accounted": True}]
