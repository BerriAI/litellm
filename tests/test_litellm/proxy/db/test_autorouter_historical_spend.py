import json
import re
from collections.abc import Mapping
from typing import Final

import psycopg
import pytest
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from pytest_postgresql import factories

from litellm.proxy.db.autorouter_daily_spend import AutoRouterDailyCosts
from litellm.proxy.db.autorouter_historical_spend import AUTOROUTER_HISTORICAL_COSTS_SQL
from litellm.proxy.route_llm_request import ROUTE_ENDPOINT_MAPPING
from litellm.proxy.spend_tracking.savings import classifier_cost_from_decision, known_autorouter_savings

_historical_postgresql_proc: Final = factories.postgresql_proc()
_historical_postgresql: Final = factories.postgresql("_historical_postgresql_proc")
_DDL: Final = """
CREATE TABLE "LiteLLM_DailyUserSpend" (
    date TEXT, user_id TEXT, api_key TEXT, model TEXT, custom_llm_provider TEXT,
    mcp_namespaced_tool_name TEXT, endpoint TEXT,
    api_requests BIGINT DEFAULT 0, successful_requests BIGINT DEFAULT 0, failed_requests BIGINT DEFAULT 0,
    prompt_tokens BIGINT DEFAULT 0, completion_tokens BIGINT DEFAULT 0, spend FLOAT8 DEFAULT 0,
    autorouter_requests BIGINT DEFAULT 0, autorouter_llm_spend FLOAT8 DEFAULT 0,
    autorouter_classifier_cost FLOAT8 DEFAULT 0, autorouter_classifier_cost_recorded_requests BIGINT DEFAULT 0,
    autorouter_estimated_requests BIGINT DEFAULT 0, autorouter_estimated_actual_spend FLOAT8 DEFAULT 0,
    autorouter_savings_spend FLOAT8 DEFAULT 0, autorouter_accounted_requests BIGINT DEFAULT 0
);
CREATE TABLE "LiteLLM_SpendLogs" (
    request_id TEXT PRIMARY KEY, "startTime" TIMESTAMP, "user" TEXT, api_key TEXT,
    model TEXT, custom_llm_provider TEXT, mcp_namespaced_tool_name TEXT, call_type TEXT,
    spend FLOAT8, prompt_tokens BIGINT, completion_tokens BIGINT, status TEXT, metadata JSONB
)
"""


def _seed(conn: psycopg.Connection) -> None:
    conn.execute(_DDL)
    conn.execute(
        """INSERT INTO "LiteLLM_DailyUserSpend"
        (date, user_id, api_key, model, custom_llm_provider, mcp_namespaced_tool_name, endpoint,
         api_requests, successful_requests, prompt_tokens, completion_tokens, spend, autorouter_savings_spend)
        VALUES ('2026-09-22', 'owner', 'key', 'model', NULL, '', '/chat/completions', 2, 2, 55, 10, 5.1, 4)"""
    )
    decision: Final = {"router_model_name": "router", "classifier_cost": 0.1}
    rows: Final = (
        ("routed", 2.0, 20, 10, {"routing_decision": decision, "autorouter_savings": 4}),
        ("ordinary", 3.0, 30, 0, {}),
        ("classifier", 0.1, 5, 0, {"internal_call_origin": "autorouter_classifier", "routing_decision": decision}),
    )
    for request_id, spend, prompt, completion, metadata in rows:
        conn.execute(
            """INSERT INTO "LiteLLM_SpendLogs"
            (request_id, "startTime", "user", api_key, model, custom_llm_provider, mcp_namespaced_tool_name,
             call_type, spend, prompt_tokens, completion_tokens, status, metadata)
            VALUES (%s, '2026-09-22 12:00:00', 'owner', 'key', 'model', '', NULL,
                    'acompletion', %s, %s, %s, 'success', %s)""",
            (request_id, spend, prompt, completion, Jsonb(metadata)),
        )


def _read(conn: psycopg.Connection, *, key: str | None = None, user: str | None = None) -> AutoRouterDailyCosts:
    query: Final = re.sub(r"\$(\d+)", r"%(p\1)s", AUTOROUTER_HISTORICAL_COSTS_SQL)
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            query,  # pyright: ignore[reportArgumentType]  # fixed production SQL uses Prisma parameter syntax
            {"p1": "2026-09-22", "p2": "2026-09-22", "p3": key, "p4": user, "p5": json.dumps(ROUTE_ENDPOINT_MAPPING)},
        )
        return AutoRouterDailyCosts.model_validate(cursor.fetchone())


def test_historical_costs_include_no_session_turns_and_classifier_charge_once(
    _historical_postgresql: psycopg.Connection,
) -> None:
    _seed(_historical_postgresql)

    costs: Final = _read(_historical_postgresql, key="key", user="owner")

    assert costs == AutoRouterDailyCosts(
        requests=1,
        llm_spend=2,
        classifier_cost=0.1,
        classifier_requests=1,
        estimated_requests=1,
        estimated_actual_spend=2.1,
        saved_spend=4,
    )
    assert costs.recorded_spend == pytest.approx(2.1)
    assert costs.baseline_spend(4) == pytest.approx(6.1)
    assert _read(_historical_postgresql, key="other").requests == 0
    assert _read(_historical_postgresql, user="other").requests == 0


@pytest.mark.parametrize("missing_request", ("ordinary", "classifier", "routed"))
def test_missing_logs_cannot_certify_costs_even_when_saved_savings_match(
    _historical_postgresql: psycopg.Connection, missing_request: str
) -> None:
    _seed(_historical_postgresql)
    _historical_postgresql.execute('DELETE FROM "LiteLLM_SpendLogs" WHERE request_id = %s', (missing_request,))

    costs: Final = _read(_historical_postgresql)

    assert not costs.complete
    assert costs.saved_spend == 4
    assert costs.recorded_spend is None
    assert costs.baseline_spend(4) is None


def test_missing_zero_cost_request_invalidates_matching_spend_and_token_totals(
    _historical_postgresql: psycopg.Connection,
) -> None:
    _seed(_historical_postgresql)
    _historical_postgresql.execute(
        """UPDATE "LiteLLM_DailyUserSpend" SET spend = 2.1, prompt_tokens = 25;
        DELETE FROM "LiteLLM_SpendLogs" WHERE request_id = 'ordinary'"""
    )

    costs: Final = _read(_historical_postgresql)

    assert not costs.complete
    assert costs.recorded_spend is None
    assert costs.saved_spend == 4


@pytest.mark.parametrize("changed", ("model", "custom_llm_provider", "mcp_namespaced_tool_name", "call_type"))
def test_equal_totals_in_different_daily_identity_do_not_certify_recovery(
    _historical_postgresql: psycopg.Connection, changed: str
) -> None:
    _seed(_historical_postgresql)
    _historical_postgresql.execute(
        psycopg.sql.SQL('UPDATE "LiteLLM_SpendLogs" SET {} = %s WHERE request_id = %s').format(
            psycopg.sql.Identifier(changed)
        ),
        ("different", "ordinary"),
    )

    costs: Final = _read(_historical_postgresql)

    assert not costs.complete
    assert costs.recorded_spend is None


@pytest.mark.parametrize(
    "metadata,saved,estimated,comparison",
    (
        ({"autorouter_savings": 0}, 0, 1, True),
        ({"autorouter_savings": -1}, -1, 1, True),
        ({"autorouter_savings": 0, "autorouter_savings_estimate": {"version": 3, "status": "unknown"}}, 0, 0, True),
        ({"autorouter_savings": 0, "autorouter_savings_estimate": {"version": 3, "status": "estimated"}}, 0, 1, True),
        ({}, 4, 0, False),
        ({"autorouter_savings": True}, 4, 0, False),
        ({"autorouter_savings": "4"}, 4, 0, False),
        ({"autorouter_savings": 4, "autorouter_savings_estimate": {}}, 4, 0, False),
        (
            {"autorouter_savings": 4, "autorouter_savings_estimate": {"version": 1.0, "status": "estimated"}},
            4,
            0,
            False,
        ),
        (
            {"autorouter_savings": 4, "autorouter_savings_estimate": {"version": True, "status": "estimated"}},
            4,
            0,
            False,
        ),
        ({"autorouter_savings": 4, "autorouter_savings_estimate": {"version": 99, "status": "estimated"}}, 4, 0, False),
    ),
)
def test_recorded_estimate_provenance_gates_baseline_without_hiding_actual_cost(
    _historical_postgresql: psycopg.Connection,
    metadata: Mapping[str, object],
    saved: float,
    estimated: int,
    comparison: bool,
) -> None:
    _seed(_historical_postgresql)
    _historical_postgresql.execute(
        'UPDATE "LiteLLM_SpendLogs" SET metadata = %s WHERE request_id = %s',
        (Jsonb({"routing_decision": {"router_model_name": "router", "classifier_cost": 0.1}, **metadata}), "routed"),
    )
    _historical_postgresql.execute('UPDATE "LiteLLM_DailyUserSpend" SET autorouter_savings_spend = %s', (saved,))

    costs: Final = _read(_historical_postgresql)
    estimate: Final = metadata.get("autorouter_savings_estimate")
    recorded: Final = known_autorouter_savings(
        model=None,
        custom_llm_provider=None,
        routing_decision=None,
        usage_object=None,
        recorded_autorouter_savings=metadata.get("autorouter_savings"),
        recorded_autorouter_savings_estimate=estimate if isinstance(estimate, Mapping) else None,
    )

    assert costs.complete and costs.recorded_spend == pytest.approx(2.1)
    assert costs.estimated_requests == estimated == int(recorded is not None)
    assert costs.comparison_complete is comparison
    assert costs.saved_spend == saved
    assert costs.baseline_spend(saved) == (pytest.approx(2.1 + saved) if estimated and comparison else None)


def test_unknown_classifier_preserves_recorded_subtotal_with_partial_coverage(
    _historical_postgresql: psycopg.Connection,
) -> None:
    _seed(_historical_postgresql)
    _historical_postgresql.execute(
        """UPDATE "LiteLLM_SpendLogs" SET metadata = metadata #- '{routing_decision,classifier_cost}'
        WHERE request_id = 'routed'"""
    )

    costs: Final = _read(_historical_postgresql)

    assert costs.complete and costs.recorded_llm_spend == 2
    assert costs.recorded_classifier_cost is None
    assert costs.recorded_spend == 2
    assert costs.coverage == "partial"
    assert costs.baseline_spend(4) is None


@pytest.mark.parametrize("classifier,expected", ((-0.1, -0.1), (0, 0), (True, None), ("0.1", None)))
def test_classifier_numeric_contract_matches_daily_owner(
    _historical_postgresql: psycopg.Connection, classifier: object, expected: float | None
) -> None:
    _seed(_historical_postgresql)
    _historical_postgresql.execute(
        """UPDATE "LiteLLM_SpendLogs" SET metadata = jsonb_set(metadata, '{routing_decision,classifier_cost}', %s)
        WHERE request_id = 'routed'""",
        (Jsonb(classifier),),
    )

    costs: Final = _read(_historical_postgresql)

    assert costs.recorded_classifier_cost == expected == classifier_cost_from_decision({"classifier_cost": classifier})
    assert costs.classifier_requests == int(expected is not None)
    assert costs.recorded_spend == pytest.approx(2 + (expected or 0))


def test_unknown_routed_request_is_counted_in_actual_but_excluded_from_baseline(
    _historical_postgresql: psycopg.Connection,
) -> None:
    _seed(_historical_postgresql)
    _historical_postgresql.execute(
        'UPDATE "LiteLLM_SpendLogs" SET metadata = %s WHERE request_id = %s',
        (
            Jsonb(
                {
                    "routing_decision": {"router_model_name": "router", "classifier_cost": 0},
                    "autorouter_savings_estimate": {"version": 3, "status": "unknown"},
                }
            ),
            "ordinary",
        ),
    )

    costs: Final = _read(_historical_postgresql)

    assert costs.complete and costs.comparison_complete
    assert costs.requests == 2 and costs.estimated_requests == 1
    assert costs.recorded_spend == pytest.approx(5.1)
    assert costs.baseline_spend(4) == pytest.approx(6.1)


def test_historical_logs_replace_partial_bucket_and_retain_durable_other_days(
    _historical_postgresql: psycopg.Connection,
) -> None:
    _seed(_historical_postgresql)
    _historical_postgresql.execute(
        """UPDATE "LiteLLM_DailyUserSpend" SET autorouter_accounted_requests = 1,
        autorouter_requests = 1, autorouter_llm_spend = 1, autorouter_classifier_cost = 0.05,
        autorouter_classifier_cost_recorded_requests = 1, autorouter_estimated_requests = 1,
        autorouter_estimated_actual_spend = 1.05;
        INSERT INTO "LiteLLM_DailyUserSpend"
        (date, user_id, api_key, model, endpoint, api_requests, successful_requests,
         autorouter_accounted_requests, autorouter_requests, autorouter_llm_spend,
         autorouter_classifier_cost, autorouter_classifier_cost_recorded_requests,
         autorouter_estimated_requests, autorouter_estimated_actual_spend, autorouter_savings_spend)
        VALUES ('2026-09-22', 'owner', 'key', 'durable-model', '/chat/completions', 1, 1, 1, 1, 3, 0.2, 1, 1, 3.2, 2),
               ('2026-09-21', 'owner', 'key', 'durable-model', '/chat/completions', 1, 1, 1, 1, 90, 1, 1, 1, 91, 9)"""
    )

    costs: Final = _read(_historical_postgresql)

    assert costs.complete and costs.requests == 2
    assert costs.recorded_spend == pytest.approx(5.3)
    assert costs.saved_spend == 6
    assert costs.baseline_spend(6) == pytest.approx(11.3)
