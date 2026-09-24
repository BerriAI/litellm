import json
import os
import re
from datetime import datetime, timezone
from typing import Final

import psycopg
import pytest
from psycopg.rows import dict_row

from litellm.proxy.spend_tracking.routing_usage import ROUTING_USAGE_SQL, RoutingUsageRow


def test_routing_ledger_keeps_history_scoped_and_separates_overhead() -> None:
    params: Final = {
        "p1": datetime(2025, 1, 1, tzinfo=timezone.utc),
        "p2": datetime(2025, 1, 2, tzinfo=timezone.utc),
        "p3": True,
        "p4": "owner",
        "p5": [],
        "p6": None,
        "p7": None,
        "p8": None,
        "p9": None,
        "p10": None,
        "p11": ["health"],
    }
    query: Final = re.sub(r"\$(\d+)", r"%(p\1)s", ROUTING_USAGE_SQL)
    routed: Final = {
        "routing_origin": {"kind": "router", "router_name": "deleted-router"},
        "routing_decision": {"router_model_name": "deleted-router", "classifier_cost": 0.25},
    }
    direct: Final = {"routing_origin": {"kind": "direct", "router_name": None}}
    records: Final = (
        ("alias-a", "a", "model-a", "alias-a", "success", "false", 1, direct, "owner", "team", "key", "2025-01-01"),
        ("alias-b", "b", "model-a", "alias-b", "success", "true", 0, direct, "owner", "team", "key", "2025-01-01"),
        ("routed", "c", "model-a", "router", "success", "false", 2, routed, "owner", "team", "key", "2025-01-01"),
        ("retry", "c", "model-a", "router", "failure", "false", 0.5, routed, "owner", "team", "key", "2025-01-01"),
        (
            "classifier",
            "d",
            "classifier-model",
            "router",
            "success",
            "false",
            0.25,
            {**routed, "internal_call_origin": "autorouter_classifier"},
            "owner",
            "team",
            "key",
            "2025-01-01",
        ),
        (
            "judge",
            "e",
            "model-a",
            "router",
            "success",
            "false",
            9,
            {**routed, "internal_call_origin": "shadow_eval_judge"},
            "owner",
            "team",
            "key",
            "2025-01-01",
        ),
        ("old", "f", "model-a", "alias", "success", "false", 3, {}, "owner", "team", "key", "2025-01-01"),
        (
            "decision",
            "g",
            "model-a",
            "router-two",
            "success",
            "false",
            4,
            {"routing_decision": {"router_model_name": "router-two", "classifier_cost": "unknown"}},
            "owner",
            "team",
            "key",
            "2025-01-01",
        ),
        (
            "synthetic",
            "h",
            "model-b",
            "synthetic",
            "success",
            "false",
            5,
            {**direct, "router_metadata": {"requested_model": "synthetic"}},
            "owner",
            "team",
            "key",
            "2025-01-01",
        ),
        (
            "fallback",
            "i",
            "model-b",
            "plain",
            "success",
            "false",
            6,
            {"routing_origin": {"kind": "router", "router_name": "deleted-router"}},
            "owner",
            "team",
            "key",
            "2025-01-01",
        ),
        (
            "unknown-model",
            "j",
            "",
            "router",
            "failure",
            "false",
            0,
            {"routing_origin": {"kind": "router", "router_name": "deleted-router"}},
            "owner",
            "team",
            "key",
            "2025-01-01",
        ),
        (
            "foreign",
            "k",
            "private-model",
            "alias",
            "success",
            "false",
            100,
            direct,
            "other",
            "other-team",
            "other-key",
            "2025-01-01",
        ),
        ("end-bound", "l", "model-a", "alias", "success", "false", 1000, direct, "owner", "team", "key", "2025-01-02"),
        (
            "before-bound",
            "m",
            "model-a",
            "alias",
            "success",
            "false",
            1000,
            direct,
            "owner",
            "team",
            "key",
            "2024-12-31 23:59:59.999999",
        ),
    )
    with psycopg.connect(os.environ["DATABASE_URL"], row_factory=dict_row) as conn:
        conn.execute(
            'CREATE TEMP TABLE "LiteLLM_SpendLogs" (request_id text, litellm_call_id text, model text, '
            "model_group text, status text, cache_hit text, spend double precision, metadata jsonb, "
            '"user" text, team_id text, api_key text, "startTime" timestamp, '
            "custom_llm_provider text DEFAULT 'provider', call_type text DEFAULT 'acompletion')"
        )
        for record in records:
            conn.execute(
                'INSERT INTO "LiteLLM_SpendLogs" (request_id, litellm_call_id, model, model_group, status, '
                'cache_hit, spend, metadata, "user", team_id, api_key, "startTime") '
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (*record[:7], json.dumps(record[7]), *record[8:]),
            )
        conn.execute("SET TIME ZONE 'America/Los_Angeles'")
        rows: Final = tuple(RoutingUsageRow.model_validate(row) for row in conn.execute(query, params))
        assert sum(row.inference_spend for row in rows) == pytest.approx(121.5)
        assert sum(row.requests for row in rows) == 8
        assert sum(row.failed_attempts for row in rows) == 2
        assert sum(row.classifier_spend for row in rows) == pytest.approx(0.25)
        conn.execute(
            "UPDATE \"LiteLLM_SpendLogs\" SET metadata = jsonb_set(metadata, '{routing_decision,decision_id}', to_jsonb(request_id)) WHERE request_id IN ('routed', 'retry')"
        )
        reclassified: Final = tuple(RoutingUsageRow.model_validate(row) for row in conn.execute(query, params))
        assert sum(row.classifier_spend for row in reclassified) == pytest.approx(0.5)

        assert {(row.attribution, row.router_name, row.model, row.requests, row.cache_hits) for row in rows} == {
            ("direct", None, "model-a", 2, 1),
            ("direct", None, "private-model", 1, 0),
            ("router", "deleted-router", "model-a", 1, 0),
            ("router", "deleted-router", "model-b", 1, 0),
            ("router", "deleted-router", "Unknown model", 0, 0),
            ("router", "router-two", "model-a", 1, 0),
            ("router", "synthetic", "model-b", 1, 0),
            ("unattributed", None, "model-a", 1, 0),
        }
        owned: Final = tuple(
            RoutingUsageRow.model_validate(row) for row in conn.execute(query, {**params, "p3": False})
        )
        assert sum(row.inference_spend for row in owned) == pytest.approx(21.5)
        assert list(conn.execute(query, {**params, "p3": False, "p7": "other"})) == []
        assert list(conn.execute(query, {**params, "p3": False, "p8": "other-key"})) == []
        team_view: Final = tuple(conn.execute(query, {**params, "p3": False, "p4": "viewer", "p5": ["team"]}))
        assert len(team_view) == len(owned)
        filtered: Final = tuple(
            RoutingUsageRow.model_validate(row)
            for row in conn.execute(query, {**params, "p9": "deleted-router", "p10": "model-a"})
        )
        assert len(filtered) == 1
        assert filtered[0].inference_spend == pytest.approx(2.5)
        assert filtered[0].classifier_cost_known_requests == 1
        assert list(conn.execute(query, {**params, "p9": "' OR true --"})) == []
