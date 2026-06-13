"""
Regression test for the budget-reservation settle path under Redis trouble.

Holds (in-flight reservations) live separately from the committed spend
counter, and the committed counter is only ever incremented by actual cost.
So even if removing this request's holds fails (a Redis blip during settle),
``increment_spend_counters`` must still write the real cost to the committed
counter — the two operations are independent and a hold-removal failure cannot
suppress the spend increment or corrupt the shared counter.
"""

import pytest

from litellm.caching import DualCache
from litellm.proxy import proxy_server


@pytest.mark.asyncio
async def test_committed_spend_increments_even_when_hold_removal_fails(monkeypatch):
    hashed_token = "hashed_test_token"
    counter_key = f"spend:key:{hashed_token}"
    reserved_cost = 0.5
    response_cost = 1.0

    counter_cache = DualCache()
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", DualCache())
    monkeypatch.setattr(proxy_server, "spend_counter_cache", counter_cache)

    async def fail_remove(*args, **kwargs):
        raise Exception("Redis timeout")

    monkeypatch.setattr(proxy_server.budget_hold_store, "remove", fail_remove)

    budget_reservation = {
        "hold_id": "hold-redis-failure",
        "finalized": False,
        "reserved_cost": reserved_cost,
        "entries": [
            {
                "counter_key": counter_key,
                "entity_type": "Key",
                "entity_id": hashed_token,
                "reserved_cost": reserved_cost,
            }
        ],
    }

    await proxy_server.increment_spend_counters(
        token=hashed_token,
        team_id=None,
        user_id=None,
        response_cost=response_cost,
        budget_reservation=budget_reservation,
    )

    enforced_spend = counter_cache.in_memory_cache.get_cache(key=counter_key)
    assert enforced_spend == pytest.approx(response_cost)
    assert budget_reservation["finalized"] is True
