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
from litellm.proxy._types import UserAPIKeyAuth
from litellm.redis.stores.connection import RedisStores, connect
from litellm.redis.stores.spend_counters import SpendCounterStore
from litellm.rust_bridge._native import NativeRedisStoreError
from tests.unit.proxy.spend_tracking.native_budgets_replica import native_reservation, refused_redis_client


@pytest.mark.asyncio
async def test_direct_increment_runs_when_reservation_reconcile_hits_redis_failure(
    monkeypatch,
    native_budgets_on,
):
    hashed_token = "hashed_test_token"
    counter_key = f"spend:key:{hashed_token}"
    reserved_cost = 0.5
    response_cost = 1.0

    spend = SpendCounterStore(connect(None).spend_counters())
    spend.set_local(counter_key, 0.0)

    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", DualCache())
    monkeypatch.setattr(proxy_server, "spend_counter_store", spend)

    budget_reservation = await native_reservation(
        spend,
        UserAPIKeyAuth(token=hashed_token),
        [{"counter_key": counter_key, "entity_type": "Key", "entity_id": hashed_token}],
        reserved_cost,
    )
    assert spend.get_local(counter_key) == pytest.approx(reserved_cost)
    spend.attach_redis(RedisStores(refused_redis_client()))

    with pytest.raises(NativeRedisStoreError, match="transport"):
        await proxy_server.increment_spend_counters(
            token=hashed_token,
            team_id=None,
            user_id=None,
            response_cost=response_cost,
            budget_reservation=budget_reservation,
        )

    assert spend.get_local(counter_key) is None
