"""
Regression test for enforced-spend underreporting when Redis fails during the
budget-reservation reconcile step of ``increment_spend_counters``.

Production failure mode: a managed Redis returns an intermittent timeout on the
reconcile increment. Reconcile deletes (invalidates) the shared counter and
gives up, but ``increment_spend_counters`` still treats the counter as
"already reconciled" and skips the direct increment. The actual call cost never
lands in the enforced counter, so budgets stop gating until the next cold
reseed pulls a lagging value from the DB.

The reconcile adjustment and the direct increment now leave in one pipeline, so
a failure either writes the actual cost or drops the counter (and surfaces the
error) for the next read to reseed from the DB; it never leaves the reserved
estimate in place as if it were reconciled.
"""

import pytest

from litellm.caching import DualCache
from litellm.proxy import proxy_server
from litellm.rust_bridge._native import NativeSpendCounterError
from tests.unit.redis.stores.spend_counters_double import FakeSpendCounters


@pytest.mark.asyncio
async def test_direct_increment_runs_when_reservation_reconcile_hits_redis_failure(
    monkeypatch,
):
    hashed_token = "hashed_test_token"
    counter_key = f"spend:key:{hashed_token}"
    reserved_cost = 0.5
    response_cost = 1.0

    spend = FakeSpendCounters(
        redis={counter_key: reserved_cost}, memory={counter_key: reserved_cost}, fail_pipelines=True
    )

    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", DualCache())
    monkeypatch.setattr(proxy_server, "spend_counter_store", spend.store())

    budget_reservation = {
        "reserved_cost": reserved_cost,
        "finalized": False,
        "entries": [
            {
                "counter_key": counter_key,
                "entity_type": "Key",
                "entity_id": hashed_token,
                "reserved_cost": reserved_cost,
                "applied_adjustment": 0.0,
            }
        ],
    }

    with pytest.raises(NativeSpendCounterError, match="transport"):
        await proxy_server.increment_spend_counters(
            token=hashed_token,
            team_id=None,
            user_id=None,
            response_cost=response_cost,
            budget_reservation=budget_reservation,
        )

    assert spend.redis == {}
    assert spend.memory == {}
