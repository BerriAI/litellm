import asyncio
from types import SimpleNamespace

import pytest

from litellm.router_strategy.budget_limiter import RouterBudgetLimiting
from litellm.types.utils import BudgetConfig


class _MockRedisCache:
    def __init__(self, initial_values):
        self.values = initial_values
        self.events = []

    async def async_increment_pipeline(self, increment_list, **kwargs):
        self.events.append("increment_pipeline:start")
        await asyncio.sleep(0.05)
        for op in increment_list:
            key = op["key"]
            current = float(self.values.get(key, 0.0) or 0.0)
            self.values[key] = current + float(op["increment_value"])
        self.events.append("increment_pipeline:done")

    async def async_batch_get_cache(self, key_list, **kwargs):
        self.events.append("batch_get")
        return {key: self.values.get(key) for key in key_list}


class _MockInMemoryCache:
    def __init__(self, initial_values):
        self.values = initial_values

    async def async_set_cache(self, key, value, **kwargs):
        self.values[key] = float(value)


@pytest.mark.asyncio
async def test_should_await_redis_pipeline_before_sync_reads():
    spend_key = "provider_spend:openai:1d"
    redis_cache = _MockRedisCache(initial_values={spend_key: 100.0})
    in_memory_cache = _MockInMemoryCache(initial_values={spend_key: 160.0})

    budget_limiter = RouterBudgetLimiting.__new__(RouterBudgetLimiting)
    budget_limiter.dual_cache = SimpleNamespace(
        redis_cache=redis_cache,
        in_memory_cache=in_memory_cache,
    )
    budget_limiter.provider_budget_config = {
        "openai": BudgetConfig(time_period="1d", budget_limit=500.0)
    }
    budget_limiter.deployment_budget_config = None
    budget_limiter.tag_budget_config = None
    budget_limiter.redis_increment_operation_queue = [
        {
            "key": spend_key,
            "increment_value": 60.0,
            "ttl": 86400,
        }
    ]

    await budget_limiter._sync_in_memory_spend_with_redis()

    assert redis_cache.values[spend_key] == 160.0
    assert in_memory_cache.values[spend_key] == 160.0
    assert budget_limiter.redis_increment_operation_queue == []
    assert redis_cache.events == [
        "increment_pipeline:start",
        "increment_pipeline:done",
        "batch_get",
    ]


@pytest.mark.asyncio
async def test_should_requeue_increments_when_redis_pipeline_fails():
    spend_key = "provider_spend:openai:1d"

    class _FailingRedisCache:
        async def async_increment_pipeline(self, increment_list, **kwargs):
            raise RuntimeError("redis down")

    budget_limiter = RouterBudgetLimiting.__new__(RouterBudgetLimiting)
    budget_limiter.dual_cache = SimpleNamespace(
        redis_cache=_FailingRedisCache(),
        in_memory_cache=SimpleNamespace(),
    )
    budget_limiter.redis_increment_operation_queue = [
        {"key": spend_key, "increment_value": 10.0, "ttl": 86400}
    ]

    await budget_limiter._push_in_memory_increments_to_redis()

    assert budget_limiter.redis_increment_operation_queue == [
        {"key": spend_key, "increment_value": 10.0, "ttl": 86400}
    ]
