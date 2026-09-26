import json
from collections.abc import Mapping
from typing import Final

import pytest
from prisma import Prisma

from litellm.proxy.db.autorouter_daily_spend import AUTOROUTER_DAILY_COSTS_SQL, AutoRouterDailyCosts
from litellm.proxy.db.autorouter_historical_spend import AUTOROUTER_HISTORICAL_COSTS_SQL
from litellm.proxy.route_llm_request import ROUTE_ENDPOINT_MAPPING
from litellm.proxy.db.daily_spend_bulk_upsert import DAILY_SPEND_TABLES, build_bulk_upsert, merge_by_conflict_key

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


@pytest.mark.parametrize("missing", (None, "routed", "free", "classifier"))
async def test_historical_costs_require_all_logs_and_count_classifier_charges_once(db: Prisma, missing: str | None) -> None:
    async with db.tx() as tx:
        for table in ("LiteLLM_DailyUserSpend", "LiteLLM_SpendLogs"):
            await tx.execute_raw(f'CREATE TEMP TABLE "{table}" (LIKE public."{table}" INCLUDING ALL) ON COMMIT DROP')
        await tx.execute_raw("""INSERT INTO "LiteLLM_DailyUserSpend"
            (id,date,user_id,api_key,model,custom_llm_provider,mcp_namespaced_tool_name,endpoint,
             api_requests,successful_requests,prompt_tokens,completion_tokens,spend,autorouter_savings_spend,updated_at)
            VALUES ('history','2026-09-22','owner','key','model','provider','','/chat/completions',2,2,25,10,2.1,4,NOW())
        """)
        decision: Final = {"router_model_name": "router", "classifier_cost": 0.1}
        for request_id, spend, prompt, completion, metadata in (
            ("routed", 2.0, 20, 10, {"routing_decision": decision, "autorouter_savings": 4}),
            ("free", 0.0, 0, 0, {}),
            ("classifier", 0.1, 5, 0, {"internal_call_origin": "autorouter_classifier"}),
        ):
            await tx.execute_raw("""INSERT INTO "LiteLLM_SpendLogs"
                (request_id,"startTime","endTime","user",api_key,model,custom_llm_provider,
                 call_type,spend,prompt_tokens,completion_tokens,status,metadata)
                VALUES ($1,'2026-09-22 12:00:00','2026-09-22 12:00:00','owner','key','model','provider',
                        'acompletion',$2::float8,$3::integer,$4::integer,'success',$5::jsonb)
            """, request_id, spend, prompt, completion, json.dumps(metadata))
        if missing is not None:
            await tx.execute_raw('DELETE FROM "LiteLLM_SpendLogs" WHERE request_id=$1', missing)
        rows: Final = await tx.query_raw(
            AUTOROUTER_HISTORICAL_COSTS_SQL, "2026-09-22", "2026-09-22", "key", "owner",
            json.dumps(ROUTE_ENDPOINT_MAPPING),
        )
        costs: Final = AutoRouterDailyCosts.model_validate(rows[0])
        assert costs.complete is (missing is None)
        assert costs.saved_spend == 4
        assert costs.recorded_spend == (pytest.approx(2.1) if missing is None else None)
        assert costs.baseline_spend(4) == (pytest.approx(6.1) if missing is None else None)
