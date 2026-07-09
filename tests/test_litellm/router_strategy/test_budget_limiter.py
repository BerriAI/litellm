import asyncio

import pytest

from litellm.caching.caching import DualCache
from litellm.caching.redis_cache import RedisPipelineIncrementOperation
from litellm.router_strategy.budget_limiter import RouterBudgetLimiting
from litellm.types.utils import BudgetConfig


class FakeRedisCache:
    """Minimal async Redis stand-in whose increment pipeline yields control before
    applying, so fire-and-forget scheduling would race a subsequent read."""

    def __init__(self, store):
        self.store = dict(store)

    async def async_get_cache(self, key, **kwargs):
        return self.store.get(key)

    async def async_set_cache(self, key, value, **kwargs):
        self.store[key] = value

    async def async_batch_get_cache(self, key_list, **kwargs):
        return {key: self.store.get(key) for key in key_list}

    async def async_increment_pipeline(self, increment_list, **kwargs):
        await asyncio.sleep(0.05)
        results = []
        for op in increment_list:
            self.store[op["key"]] = self.store.get(op["key"], 0.0) + op["increment_value"]
            results.append(self.store[op["key"]])
        return results


class FlakyRedisCache(FakeRedisCache):
    """Fails the increment pipeline while `failures` > 0, then behaves like FakeRedisCache."""

    def __init__(self, store, failures=0):
        super().__init__(store)
        self.failures = failures

    async def async_increment_pipeline(self, increment_list, **kwargs):
        if self.failures > 0:
            self.failures -= 1
            raise ConnectionError("redis unreachable")
        return await super().async_increment_pipeline(increment_list, **kwargs)


@pytest.fixture
def disable_budget_sync(monkeypatch):
    async def noop(*args, **kwargs):
        return None

    monkeypatch.setattr(
        "litellm.router_strategy.budget_limiter.RouterBudgetLimiting.periodic_sync_in_memory_spend_with_redis",
        noop,
    )


async def _make_limiter(fake_redis, provider):
    limiter = RouterBudgetLimiting(
        dual_cache=DualCache(redis_cache=fake_redis),
        provider_budget_config={provider: BudgetConfig(time_period="1d", budget_limit=1000)},
    )
    # let the background _init_provider_budget_in_cache tasks settle
    await asyncio.sleep(0.1)
    return limiter


@pytest.mark.asyncio
async def test_sync_awaits_pipeline_before_reading_redis(disable_budget_sync):
    """Regression for #32614: sync must flush queued increments to Redis before
    reading them back, otherwise in-memory spend is clobbered with stale Redis spend."""
    provider = "openai"
    spend_key = f"provider_spend:{provider}:1d"

    fake_redis = FakeRedisCache({spend_key: 100.0})
    limiter = await _make_limiter(fake_redis, provider)

    # in-memory already reflects the incremented spend (160), the delta (+60) is still queued for Redis
    await limiter.dual_cache.in_memory_cache.async_set_cache(key=spend_key, value=160.0)
    fake_redis.store[spend_key] = 100.0
    limiter.redis_increment_operation_queue = [
        RedisPipelineIncrementOperation(key=spend_key, increment_value=60.0, ttl=86400)
    ]

    await limiter._sync_in_memory_spend_with_redis()

    in_memory_spend = await limiter.dual_cache.in_memory_cache.async_get_cache(spend_key)
    assert float(in_memory_spend) == 160.0
    assert fake_redis.store[spend_key] == 160.0


@pytest.mark.asyncio
async def test_sync_pulls_in_other_instance_spend(disable_budget_sync):
    """Redis ahead of memory (another instance spent) should win."""
    provider = "anthropic"
    spend_key = f"provider_spend:{provider}:1d"

    fake_redis = FakeRedisCache({spend_key: 100.0})
    limiter = await _make_limiter(fake_redis, provider)

    await limiter.dual_cache.in_memory_cache.async_set_cache(key=spend_key, value=100.0)
    fake_redis.store[spend_key] = 250.0
    limiter.redis_increment_operation_queue = []

    await limiter._sync_in_memory_spend_with_redis()

    in_memory_spend = await limiter.dual_cache.in_memory_cache.async_get_cache(spend_key)
    assert float(in_memory_spend) == 250.0


@pytest.mark.asyncio
async def test_push_returns_awaited_redis_values(disable_budget_sync):
    """_push_in_memory_increments_to_redis must await the pipeline and return the
    resulting per-key Redis values (not schedule a background task)."""
    provider = "openai"
    spend_key = f"provider_spend:{provider}:1d"

    fake_redis = FakeRedisCache({spend_key: 10.0})
    limiter = await _make_limiter(fake_redis, provider)

    limiter.redis_increment_operation_queue = [
        RedisPipelineIncrementOperation(key=spend_key, increment_value=5.0, ttl=86400)
    ]

    result = await limiter._push_in_memory_increments_to_redis()

    assert result == {spend_key: 15.0}
    assert fake_redis.store[spend_key] == 15.0
    assert limiter.redis_increment_operation_queue == []


@pytest.mark.asyncio
async def test_push_requeues_increments_when_pipeline_fails(disable_budget_sync):
    """A failed Redis pipeline must restore the snapshot to the queue so the increments
    are retried on the next push instead of being silently lost."""
    provider = "openai"
    spend_key = f"provider_spend:{provider}:1d"

    fake_redis = FlakyRedisCache({spend_key: 10.0})
    limiter = await _make_limiter(fake_redis, provider)

    op = RedisPipelineIncrementOperation(key=spend_key, increment_value=5.0, ttl=86400)
    limiter.redis_increment_operation_queue = [op]
    fake_redis.failures = 1

    assert await limiter._push_in_memory_increments_to_redis() is None
    assert limiter.redis_increment_operation_queue == [op]
    assert fake_redis.store[spend_key] == 10.0

    assert await limiter._push_in_memory_increments_to_redis() == {spend_key: 15.0}
    assert limiter.redis_increment_operation_queue == []
    assert fake_redis.store[spend_key] == 15.0
