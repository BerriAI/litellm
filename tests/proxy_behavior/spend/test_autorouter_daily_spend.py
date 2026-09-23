import json
import os
import time
import uuid
from collections.abc import Mapping
from datetime import timedelta
from typing import Final
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pytest
from prisma import Prisma
from prisma.errors import RawQueryError

from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.db.autorouter_daily_spend import AUTOROUTER_DAILY_COSTS_SQL, AutoRouterDailyCosts
from litellm.proxy.db.autorouter_historical_spend import AUTOROUTER_HISTORICAL_COSTS_SQL, recover_daily_router_costs
from litellm.proxy.db.daily_spend_bulk_upsert import DAILY_SPEND_TABLES, build_bulk_upsert, merge_by_conflict_key
from litellm.proxy.route_llm_request import ROUTE_ENDPOINT_MAPPING
from litellm.proxy.utils import PrismaClient, ProxyLogging

pytestmark = pytest.mark.asyncio(loop_scope="session")


def _daily(day: str, key: str, user: str, llm: float, classifier: float, saved: float) -> Mapping[str, object]:
    return {
        "date": day,
        "api_key": key,
        "user_id": user,
        "model": "model",
        "custom_llm_provider": "provider",
        "spend": llm,
        "api_requests": 1,
        "successful_requests": 1,
        "autorouter_accounted_requests": 1,
        "autorouter_requests": 1,
        "autorouter_llm_spend": llm,
        "autorouter_classifier_cost": classifier,
        "autorouter_classifier_cost_recorded_requests": 1,
        "autorouter_estimated_requests": 1,
        "autorouter_estimated_actual_spend": llm + classifier,
        "autorouter_savings_spend": saved,
    }


async def test_daily_costs_use_request_dates_and_filters_instead_of_session_boundaries(db: Prisma) -> None:
    async with db.tx() as tx:
        await tx.execute_raw(
            'CREATE TEMP TABLE "LiteLLM_DailyUserSpend" (LIKE public."LiteLLM_DailyUserSpend" INCLUDING ALL) ON COMMIT DROP'
        )
        rows: Final = (
            _daily("2026-09-21", "a", "owner-a", 8.0, 0.0, 4.0),
            _daily("2026-09-22", "a", "owner-a", 1.8, 0.2, 1.0),
            _daily("2026-09-22", "a", "owner-a", 3.0, 0.0, 2.0),
            _daily("2026-09-22", "b", "owner-b", 7.0, 0.0, 5.0),
        )
        table: Final = DAILY_SPEND_TABLES["user"]
        sql, values = build_bulk_upsert(table, merge_by_conflict_key(table, rows))
        await tx.execute_raw(sql, *values)
        for key, user, expected_spend, saved, requests in (
            (None, None, 12.0, 8.0, 3),
            ("a", None, 5.0, 3.0, 2),
            (None, "owner-a", 5.0, 3.0, 2),
            ("a", "owner-b", 0.0, 0.0, 0),
        ):
            result: Final = await tx.query_raw(AUTOROUTER_DAILY_COSTS_SQL, "2026-09-22", "2026-09-22", key, user)
            costs: Final = AutoRouterDailyCosts.model_validate(result[0])
            assert costs.complete and costs.requests == requests
            assert costs.recorded_spend == pytest.approx(expected_spend)
            assert costs.saved_spend == saved
            assert costs.baseline_spend(saved) == pytest.approx(expected_spend + saved)


async def test_an_old_writer_cannot_certify_partial_daily_costs_as_complete(db: Prisma) -> None:
    async with db.tx() as tx:
        await tx.execute_raw(
            'CREATE TEMP TABLE "LiteLLM_DailyUserSpend" (LIKE public."LiteLLM_DailyUserSpend" INCLUDING ALL) ON COMMIT DROP'
        )
        table: Final = DAILY_SPEND_TABLES["user"]
        row: Final = _daily("2026-09-22", "a", "owner", 2.0, 0.1, 3.0)
        sql, values = build_bulk_upsert(table, merge_by_conflict_key(table, (row,)))
        await tx.execute_raw(sql, *values)
        await tx.execute_raw(
            'UPDATE "LiteLLM_DailyUserSpend" SET api_requests=api_requests+1, successful_requests=successful_requests+1, '
            "spend=spend+8, autorouter_savings_spend=autorouter_savings_spend+4"
        )
        result: Final = await tx.query_raw(AUTOROUTER_DAILY_COSTS_SQL, "2026-09-22", "2026-09-22", None, None)
        costs: Final = AutoRouterDailyCosts.model_validate(result[0])
        assert costs.coverage == "partial"
        assert costs.recorded_spend == pytest.approx(2.1)
        assert costs.saved_spend == 7.0
        assert costs.baseline_spend(7.0) is None


async def test_retained_historical_costs_reconcile_through_prisma_and_reject_a_missing_free_request(db: Prisma) -> None:
    async with db.tx() as tx:
        await tx.execute_raw(
            'CREATE TEMP TABLE "LiteLLM_DailyUserSpend" (LIKE public."LiteLLM_DailyUserSpend" INCLUDING ALL) ON COMMIT DROP'
        )
        await tx.execute_raw(
            'CREATE TEMP TABLE "LiteLLM_SpendLogs" (LIKE public."LiteLLM_SpendLogs" INCLUDING ALL) ON COMMIT DROP'
        )
        cases: Final = (
            ("2026-09-21", "a", "owner-a", 90.0, 1.0, 9.0),
            ("2026-09-22", "a", "owner-a", 2.0, 0.2, 3.0),
            ("2026-09-22", "b", "owner-b", 7.0, 0.5, 5.0),
        )
        rows: Final = tuple(
            {
                "date": day,
                "api_key": key,
                "user_id": user,
                "model": "removed-historical-model",
                "custom_llm_provider": "removed-provider",
                "endpoint": "/responses",
                "spend": llm + classifier,
                "api_requests": 2,
                "successful_requests": 2,
                "prompt_tokens": 25,
                "completion_tokens": 10,
                "autorouter_savings_spend": saved,
            }
            for day, key, user, llm, classifier, saved in cases
        )
        table: Final = DAILY_SPEND_TABLES["user"]
        sql, values = build_bulk_upsert(table, merge_by_conflict_key(table, rows))
        await tx.execute_raw(sql, *values)
        logs: Final = tuple(
            {
                "request_id": f"{day}-{key}-{suffix}",
                "started_at": f"{day}T12:00:00",
                "api_key": key,
                "user_id": user,
                "spend": spend,
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "metadata": metadata,
            }
            for day, key, user, llm, classifier, saved in cases
            for suffix, spend, prompt, completion, metadata in (
                (
                    "routed",
                    llm,
                    20,
                    10,
                    {
                        "routing_decision": {"router_model_name": "retired-router", "classifier_cost": classifier},
                        "autorouter_savings": saved,
                    },
                ),
                ("free", 0.0, 0, 0, {}),
                (
                    "classifier",
                    classifier,
                    5,
                    0,
                    {
                        "internal_call_origin": "autorouter_classifier",
                        "routing_decision": {"router_model_name": "retired-router", "classifier_cost": classifier},
                    },
                ),
            )
        )
        await tx.execute_raw(
            """INSERT INTO "LiteLLM_SpendLogs"
            (request_id, "startTime", "endTime", call_type, api_key, "user", model,
             custom_llm_provider, spend, prompt_tokens, completion_tokens, status, metadata)
            SELECT request_id, started_at::timestamp, started_at::timestamp, 'aresponses', api_key, user_id,
                   'removed-historical-model', 'removed-provider', spend, prompt_tokens, completion_tokens,
                   'success', metadata
            FROM jsonb_to_recordset($1::jsonb) AS rows(
                request_id text, started_at text, api_key text, user_id text,
                spend float8, prompt_tokens int, completion_tokens int, metadata jsonb
            )""",
            json.dumps(logs),
        )
        for key, user, actual, saved, requests in (
            (None, None, 9.7, 8.0, 2),
            ("a", None, 2.2, 3.0, 1),
            (None, "owner-a", 2.2, 3.0, 1),
            ("a", "owner-a", 2.2, 3.0, 1),
            ("a", "owner-b", 0.0, 0.0, 0),
        ):
            result: Final = await tx.query_raw(
                AUTOROUTER_HISTORICAL_COSTS_SQL,
                "2026-09-22",
                "2026-09-22",
                key,
                user,
                json.dumps(ROUTE_ENDPOINT_MAPPING),
            )
            recovered: Final = AutoRouterDailyCosts.model_validate(result[0])
            assert recovered.complete and recovered.comparison_complete
            assert recovered.requests == requests
            assert recovered.recorded_spend == pytest.approx(actual)
            assert recovered.saved_spend == saved
            assert recovered.baseline_spend(saved) == pytest.approx(actual + saved)

        await tx.execute_raw('DELETE FROM "LiteLLM_SpendLogs" WHERE request_id = $1', "2026-09-22-a-free")
        incomplete: Final = await tx.query_raw(
            AUTOROUTER_HISTORICAL_COSTS_SQL,
            "2026-09-22",
            "2026-09-22",
            "a",
            None,
            json.dumps(ROUTE_ENDPOINT_MAPPING),
        )
        costs: Final = AutoRouterDailyCosts.model_validate(incomplete[0])
        assert not costs.complete
        assert costs.saved_spend == 3.0
        assert costs.recorded_spend is None
        assert costs.baseline_spend(3.0) is None


async def test_blocked_historical_recovery_times_out_without_changing_costs_or_read_connection(
    db: Prisma, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity: Final = uuid.uuid4().hex
    database_url: Final = os.environ["DATABASE_URL"]
    parsed: Final = urlsplit(database_url)
    query: Final = urlencode({**dict(parse_qsl(parsed.query)), "connection_limit": "1"})
    monkeypatch.setenv("DATABASE_URL_READ_REPLICA", urlunsplit(parsed._replace(query=query)))
    client: Final = PrismaClient(database_url, ProxyLogging(UserApiKeyCache()))
    row: Final = {
        **_daily("2026-09-22", identity, identity, 2.0, 0.1, 7.0),
        "api_requests": 2,
        "successful_requests": 2,
        "spend": 10.0,
    }
    table: Final = DAILY_SPEND_TABLES["user"]
    statement, values = build_bulk_upsert(table, merge_by_conflict_key(table, (row,)))
    await db.execute_raw(statement, *values)
    try:
        await client.db.connect()
        await client.read_db.execute_raw("SET statement_timeout = 4500")
        connection_query: Final = (
            "SELECT pg_backend_pid() AS pid, current_setting('statement_timeout') AS timeout, "
            "current_setting('transaction_read_only') AS read_only"
        )
        connection_before: Final = await client.read_db.query_raw(connection_query)
        daily_before: Final = await client.read_db.query_raw(
            AUTOROUTER_DAILY_COSTS_SQL,
            "2026-09-22",
            "2026-09-22",
            identity,
            identity,
        )
        recorded: Final = AutoRouterDailyCosts.model_validate(daily_before[0])
        assert recorded.recorded_spend == pytest.approx(2.1) and recorded.saved_spend == 7.0
        assert recorded.coverage == "partial"
        async with db.tx(timeout=timedelta(seconds=8)) as blocker:
            await blocker.execute_raw('LOCK TABLE "LiteLLM_SpendLogs" IN ACCESS EXCLUSIVE MODE')
            started: Final = time.monotonic()
            with pytest.raises(RawQueryError, match="canceling statement due to statement timeout"):
                await recover_daily_router_costs(client, "2026-09-22", "2026-09-22", identity, identity)
            assert time.monotonic() - started < 3.5
        assert await client.read_db.query_raw(connection_query) == connection_before
        assert (
            await client.read_db.query_raw(
                AUTOROUTER_DAILY_COSTS_SQL,
                "2026-09-22",
                "2026-09-22",
                identity,
                identity,
            )
            == daily_before
        )
        recovered: Final = await recover_daily_router_costs(client, "2026-09-22", "2026-09-22", identity, identity)
        assert recovered is not None
        assert recovered.recorded_spend == recorded.recorded_spend and recovered.saved_spend == recorded.saved_spend
        assert await client.read_db.query_raw(connection_query) == connection_before
    finally:
        await client.db.disconnect()
        await db.execute_raw('DELETE FROM "LiteLLM_DailyUserSpend" WHERE api_key=$1', identity)
