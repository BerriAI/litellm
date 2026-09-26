import json
from typing import Final
from uuid import uuid4

import pytest
from httpx import AsyncClient

from litellm.proxy.utils import PrismaClient

from .actors import Actor, World

pytestmark: Final = pytest.mark.asyncio(loop_scope="session")


async def test_routing_usage_groups_spend_and_keeps_other_users_and_internal_calls_out(
    proxy_client: AsyncClient, prisma: PrismaClient, world: World
) -> None:
    db: Final = prisma.db
    caller: Final = world.keys[Actor.INTERNAL_USER]
    owner: Final = caller.user_id
    key: Final = f"routing-usage-{uuid4()}"
    records: Final = (
        ("fast", 1.0, "a", "SIMPLE", None, owner, "2026-01-01T00:00:00"),
        ("fast", 1.5, "a", "SIMPLE", None, owner, "2026-01-01T12:00:00"),
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
            {
                "model": "fast", "router_name": router, "router_type": router_type,
                "tier": tier, "requests": requests, "spend": spend,
            }
            for router, router_type, tier, requests, spend in (
                (None, None, None, 1, 3.0),
                ("a", "complexity", "SIMPLE", 2, 2.5),
                ("b", "complexity", "COMPLEX", 1, 2.0),
                ("a", "complexity", None, 1, 0.5),
            )
        ]
        router_usage: Final = await proxy_client.get(
            "/auto_router/usage", params={**query, "router_name": "a", "router_type": "complexity"}, headers=headers
        )
        assert router_usage.status_code == 200, router_usage.text
        assert router_usage.json() == [
            {
                "model": model, "router_name": "a", "router_type": "complexity",
                "tier": tier, "requests": requests, "spend": spend,
            }
            for model, tier, requests, spend in (
                ("strong", "SIMPLE", 1, 5.0), ("fast", "SIMPLE", 2, 2.5), ("fast", None, 1, 0.5)
            )
        ]
        empty: Final = await proxy_client.get(
            "/auto_router/usage",
            params={**query, "destination_model": "fast", "api_key": f"{key}-other"},
            headers=headers,
        )
        assert empty.status_code == 200, empty.text
        assert empty.json() == []
        forbidden: Final = await proxy_client.get(
            "/auto_router/usage",
            params={**query, "destination_model": "fast", "user_id": "other-user"},
            headers=headers,
        )
        assert forbidden.status_code == 403, forbidden.text
    finally:
        await db.execute_raw('DELETE FROM "LiteLLM_SpendLogs" WHERE api_key = $1', key)
