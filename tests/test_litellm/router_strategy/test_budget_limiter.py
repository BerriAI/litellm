import asyncio

import pytest

from litellm.caching.caching import DualCache
from litellm.caching.redis_cache import RedisPipelineIncrementOperation
from litellm.router_strategy.budget_limiter import RouterBudgetLimiting
from litellm.types.utils import BudgetConfig

SPEND_KEY = "provider_spend:openai:1d"


class _SlowFakeRedisCache:
    """
    Redis stand-in whose pipeline yields to the event loop before applying
    increments. This makes fire-and-forget scheduling observably stale: a read
    that happens before the pipeline is awaited to completion still sees the old
    value.
    """

    def __init__(self, store: dict[str, float]):
        self._store = store

    async def async_increment_pipeline(self, increment_list, **kwargs):
        await asyncio.sleep(0.05)
        for op in increment_list:
            self._store[op["key"]] = self._store.get(op["key"], 0.0) + op["increment_value"]
        return [self._store[op["key"]] for op in increment_list]

    async def async_batch_get_cache(self, key_list, **kwargs):
        return {key: self._store.get(key) for key in key_list}


@pytest.fixture
def budget_limiter(monkeypatch):
    monkeypatch.setattr(asyncio, "create_task", lambda coro: coro.close())
    limiter = RouterBudgetLimiting(dual_cache=DualCache(), provider_budget_config={})
    limiter.provider_budget_config = {"openai": BudgetConfig(time_period="1d", budget_limit=100)}
    return limiter


@pytest.mark.asyncio
async def test_push_in_memory_increments_to_redis_completes_before_returning(budget_limiter):
    store = {SPEND_KEY: 100.0}
    budget_limiter.dual_cache.redis_cache = _SlowFakeRedisCache(store)
    budget_limiter.redis_increment_operation_queue = [
        RedisPipelineIncrementOperation(key=SPEND_KEY, increment_value=60.0, ttl=86400),
    ]

    await budget_limiter._push_in_memory_increments_to_redis()

    assert store[SPEND_KEY] == 160.0
    assert budget_limiter.redis_increment_operation_queue == []


@pytest.mark.asyncio
async def test_sync_does_not_overwrite_memory_with_stale_redis(budget_limiter):
    await budget_limiter.dual_cache.in_memory_cache.async_set_cache(key=SPEND_KEY, value=160.0)
    store = {SPEND_KEY: 100.0}
    budget_limiter.dual_cache.redis_cache = _SlowFakeRedisCache(store)
    budget_limiter.redis_increment_operation_queue = [
        RedisPipelineIncrementOperation(key=SPEND_KEY, increment_value=60.0, ttl=86400),
    ]

    await budget_limiter._sync_in_memory_spend_with_redis()

    in_memory_spend = await budget_limiter.dual_cache.in_memory_cache.async_get_cache(SPEND_KEY)
    assert float(in_memory_spend) == 160.0
    assert store[SPEND_KEY] == 160.0
