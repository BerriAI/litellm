import json
from collections.abc import Mapping
from typing import Final
from uuid import uuid4

import pytest
from httpx import AsyncClient

from litellm.proxy.utils import PrismaClient

from .actors import Actor, World
from .conftest import MASTER_KEY

pytestmark: Final = pytest.mark.asyncio(loop_scope="session")


async def test_actual_model_spend_sources_tiers_and_scope(
    proxy_client: AsyncClient, prisma: PrismaClient, world: World
) -> None:
    db: Final = prisma.db
    caller: Final = world.keys[Actor.INTERNAL_USER]
    owner: Final = caller.user_id
    key: Final = f"routing-usage-{uuid4()}"
    records: Final = (
        ("fast", 1.0, "a", "SIMPLE", None, owner, "2026-01-01T00:00:00"),
        ("strong", 5.0, "a", "SIMPLE", None, owner, "2026-01-01T12:00:00"),
        ("fast", 0.5, "a", None, None, owner, "2026-01-01T12:00:00"),
        ("fast", 2.0, "b", "COMPLEX", None, owner, "2026-01-01T12:00:00"),
        ("fast", 3.0, None, None, None, owner, "2026-01-01T12:00:00"),
        ("fast", 90.0, "a", "SIMPLE", "shadow_eval_router", owner, "2026-01-01T12:00:00"),
        ("fast", 80.0, None, None, "autorouter_classifier", owner, "2026-01-01T12:00:00"),
        ("fast", 70.0, "a", "SIMPLE", None, "other-user", "2026-01-01T12:00:00"),
        ("fast", 60.0, "a", "SIMPLE", None, owner, "2026-01-02T00:00:00"),
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
        query: Final = {"start_date": "2026-01-01", "end_date": "2026-01-01", "api_key": key}
        headers: Final = {"Authorization": f"Bearer {caller.cleartext}"}
        model_usage: Final = await proxy_client.get(
            "/auto_router/usage", params={**query, "destination_model": "fast"}, headers=headers
        )
        assert model_usage.status_code == 200, model_usage.text
        assert model_usage.json() == [
            {"model": "fast", "router_name": router, "router_type": router_type, "tier": tier, "requests": 1, "spend": spend}
            for router, router_type, tier, spend in (
                (None, None, None, 3.0),
                ("b", "complexity", "COMPLEX", 2.0),
                ("a", "complexity", "SIMPLE", 1.0),
                ("a", "complexity", None, 0.5),
            )
        ]
        router_usage: Final = await proxy_client.get(
            "/auto_router/usage", params={**query, "router_name": "a", "router_type": "complexity"}, headers=headers
        )
        assert router_usage.status_code == 200, router_usage.text
        assert router_usage.json() == [
            {"model": model, "router_name": "a", "router_type": "complexity", "tier": tier, "requests": 1, "spend": spend}
            for model, tier, spend in (("strong", "SIMPLE", 5.0), ("fast", "SIMPLE", 1.0), ("fast", None, 0.5))
        ]
        empty: Final = await proxy_client.get(
            "/auto_router/usage",
            params={**query, "destination_model": "fast", "api_key": f"{key}-other"},
            headers=headers,
        )
        assert empty.status_code == 200, empty.text
        assert empty.json() == []

    finally:
        await db.execute_raw('DELETE FROM "LiteLLM_SpendLogs" WHERE api_key = $1', key)


@pytest.mark.parametrize(
    "params,status",
    (
        ({}, 400),
        ({"destination_model": "fast", "router_name": "a"}, 400),
        ({"destination_model": "fast", "router_type": "complexity"}, 400),
        ({"destination_model": "fast", "end_date": "2025-12-31"}, 400),
        ({"destination_model": "fast", "end_date": "9999-12-31"}, 400),
        ({"destination_model": "fast", "end_date": "2026-04-03"}, 200),
        ({"destination_model": "fast", "end_date": "2026-04-04"}, 400),
        ({"destination_model": "fast", "start_date": "0001-01-01"}, 400),
        ({"destination_model": ""}, 422),
        ({"destination_model": "fast", "start_date": "not-a-date"}, 422),
    ),
)
async def test_routing_usage_rejects_invalid_filters_and_bounds_inclusive_days(
    proxy_client: AsyncClient, params: Mapping[str, str], status: int
) -> None:
    response: Final = await proxy_client.get(
        "/auto_router/usage",
        params={"start_date": "2026-01-01", "end_date": "2026-01-01", **params},
        headers={"Authorization": f"Bearer {MASTER_KEY}"},
    )
    assert response.status_code == status, response.text


async def test_routing_usage_cannot_query_another_user(proxy_client: AsyncClient, world: World) -> None:
    caller: Final = world.keys[Actor.INTERNAL_USER]
    response: Final = await proxy_client.get(
        "/auto_router/usage",
        params={"start_date": "2026-01-01", "end_date": "2026-01-01", "destination_model": "fast", "user_id": "other"},
        headers={"Authorization": f"Bearer {caller.cleartext}"},
    )
    assert response.status_code == 403, response.text
