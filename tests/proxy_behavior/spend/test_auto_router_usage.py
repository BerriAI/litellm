import json
from datetime import date
from typing import Final
from uuid import uuid4

import pytest

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.management_endpoints.auto_router_usage import get_auto_router_usage

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_actual_model_spend_sources_tiers_and_scope(db):
    key: Final = f"routing-usage-{uuid4()}"
    records: Final = (
        ("fast", 1.0, "a", "SIMPLE", None, "owner", "2026-01-01T00:00:00"),
        ("strong", 5.0, "a", "SIMPLE", None, "owner", "2026-01-01T12:00:00"),
        ("fast", 0.5, "a", None, None, "owner", "2026-01-01T12:00:00"),
        ("fast", 2.0, "b", "COMPLEX", None, "owner", "2026-01-01T12:00:00"),
        ("fast", 3.0, None, None, None, "owner", "2026-01-01T12:00:00"),
        ("fast", 90.0, "a", "SIMPLE", "shadow_eval_router", "owner", "2026-01-01T12:00:00"),
        ("fast", 80.0, None, None, "autorouter_classifier", "owner", "2026-01-01T12:00:00"),
        ("fast", 70.0, "a", "SIMPLE", None, "other-user", "2026-01-01T12:00:00"),
        ("fast", 60.0, "a", "SIMPLE", None, "owner", "2026-01-02T00:00:00"),
    )
    try:
        for index, (model, spend, router, tier, origin, user, at) in enumerate(records):
            metadata: Final = json.dumps(
                {
                    "routing_decision": {"router_model_name": router, "router_type": "complexity", "tier": tier}
                    if router
                    else None,
                    "internal_call_origin": origin,
                }
            )
            await db.execute_raw(
                'INSERT INTO "LiteLLM_SpendLogs" (request_id, call_type, api_key, model, model_group, spend, '
                '"user", "startTime", "endTime", metadata) '
                "VALUES ($1, 'completion', $2, $3, 'renamed-alias', $4, $5, $6::timestamp, $6::timestamp, $7::jsonb)",
                f"{key}-{index}",
                key,
                model,
                spend,
                user,
                at,
                metadata,
            )
        caller: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, user_id="owner")
        model_usage: Final = await get_auto_router_usage(
            date(2026, 1, 1),
            date(2026, 1, 1),
            caller,
            db,
            destination_model="fast",
            api_key=key,
        )
        assert {(row.router_name, row.tier, row.requests, row.spend) for row in model_usage} == {
            (None, None, 1, 3.0),
            ("b", "COMPLEX", 1, 2.0),
            ("a", "SIMPLE", 1, 1.0),
            ("a", None, 1, 0.5),
        }
        router_usage: Final = await get_auto_router_usage(
            date(2026, 1, 1),
            date(2026, 1, 1),
            caller,
            db,
            router_name="a",
            router_type="complexity",
            api_key=key,
        )
        assert {(row.model, row.tier, row.requests, row.spend) for row in router_usage} == {
            ("fast", "SIMPLE", 1, 1.0),
            ("strong", "SIMPLE", 1, 5.0),
            ("fast", None, 1, 0.5),
        }
        assert (
            await get_auto_router_usage(
                date(2026, 1, 1),
                date(2026, 1, 1),
                caller,
                db,
                destination_model="fast",
                api_key=f"{key}-other",
            )
            == ()
        )
    finally:
        await db.execute_raw('DELETE FROM "LiteLLM_SpendLogs" WHERE api_key = $1', key)
