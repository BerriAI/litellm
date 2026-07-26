import asyncio
from typing import Any, Dict, List, Optional, Sequence, Union

import pytest

from litellm.caching.caching import DualCache
from litellm.caching.redis_cache import RedisPipelineIncrementOperation
from litellm.router_strategy.budget_limiter import RouterBudgetLimiting
from litellm.types.utils import BudgetConfig


@pytest.fixture
def disable_budget_sync(monkeypatch):
    async def noop(*args, **kwargs):
        return None

    monkeypatch.setattr(
        "litellm.router_strategy.budget_limiter.RouterBudgetLimiting.periodic_sync_in_memory_spend_with_redis",
        noop,
    )


class FakeAtomicRedisCache:
    """
    Stands in for RedisCache, emulating the atomicity guarantee the budget
    window Lua script relies on: the whole claim-or-increment runs without
    yielding to other coroutines.
    """

    def __init__(self) -> None:
        self.store: Dict[str, str] = {}
        self.registered_scripts: List[str] = []
        self.script_calls: List[Dict[str, Any]] = []
        self.increment_pipeline_calls: List[List[RedisPipelineIncrementOperation]] = []

    def async_register_script(self, script: str):
        self.registered_scripts.append(script)

        async def run_script(
            keys: Sequence[str], args: Sequence[Union[str, int, float]], client: Any = None
        ) -> List[str]:
            self.script_calls.append({"keys": list(keys), "args": list(args)})
            start_time_key, spend_key = keys
            current_time, response_cost, ttl = str(args[0]), str(args[1]), float(args[2])

            budget_start = self.store.get(start_time_key)
            if budget_start is None or (float(current_time) - float(budget_start)) > ttl:
                self.store[start_time_key] = current_time
                self.store[spend_key] = response_cost
                return [current_time, "1"]

            self.store[spend_key] = str(float(self.store.get(spend_key, "0")) + float(response_cost))
            return [budget_start, "0"]

        return run_script

    async def async_increment_pipeline(self, increment_list: List[RedisPipelineIncrementOperation], **kwargs):
        await asyncio.sleep(0)
        self.increment_pipeline_calls.append(list(increment_list))
        for op in increment_list:
            self.store[op["key"]] = str(float(self.store.get(op["key"], "0")) + float(op["increment_value"]))


class YieldingDualCache(DualCache):
    """
    DualCache whose reads and writes yield to the event loop, the way a real
    network-backed cache does, so concurrent callers actually interleave.
    """

    async def async_get_cache(self, *args, **kwargs):
        await asyncio.sleep(0)
        return await super().async_get_cache(*args, **kwargs)

    async def async_set_cache(self, *args, **kwargs):
        await asyncio.sleep(0)
        return await super().async_set_cache(*args, **kwargs)


def _budget_limiter(
    redis_cache: Optional[FakeAtomicRedisCache] = None,
    dual_cache: Optional[DualCache] = None,
) -> RouterBudgetLimiting:
    dual_cache = dual_cache or DualCache()
    if redis_cache is not None:
        dual_cache.redis_cache = redis_cache  # type: ignore[assignment]
    return RouterBudgetLimiting(dual_cache=dual_cache, provider_budget_config={})


@pytest.mark.asyncio
async def test_concurrent_new_budget_windows_do_not_overwrite_each_other(disable_budget_sync):
    """
    Two responses crossing the same window boundary both enter the reset path.
    The window may only be reset once; the loser must add its cost on top.
    """
    limiter = _budget_limiter()
    spend_key = "provider_spend:synthetic:1h"
    start_time_key = "provider_budget_start_time:synthetic"

    await asyncio.gather(
        limiter._handle_new_budget_window(
            spend_key=spend_key,
            start_time_key=start_time_key,
            current_time=1000.0,
            response_cost=0.05,
            ttl_seconds=3600,
        ),
        limiter._handle_new_budget_window(
            spend_key=spend_key,
            start_time_key=start_time_key,
            current_time=1000.0,
            response_cost=0.07,
            ttl_seconds=3600,
        ),
    )

    spend = await limiter.dual_cache.async_get_cache(spend_key)
    assert float(spend) == pytest.approx(0.12)
    assert float(await limiter.dual_cache.async_get_cache(start_time_key)) == 1000.0


@pytest.mark.asyncio
async def test_concurrent_spend_increments_across_expired_window_keep_full_total(disable_budget_sync):
    """
    Full path through `_increment_spend_for_key` with a window that expired long
    ago: every concurrent cost must land in the freshly opened window.
    """
    limiter = _budget_limiter(dual_cache=YieldingDualCache())
    budget_config = BudgetConfig(time_period="1h", budget_limit=100)
    spend_key = "provider_spend:synthetic:1h"
    start_time_key = "provider_budget_start_time:synthetic"

    await limiter.dual_cache.async_set_cache(key=start_time_key, value=1000.0, ttl=3600)
    await limiter.dual_cache.async_set_cache(key=spend_key, value=42.0, ttl=3600)

    costs = [0.05, 0.07, 0.11]
    await asyncio.gather(
        *[
            limiter._increment_spend_for_key(
                budget_config=budget_config,
                spend_key=spend_key,
                start_time_key=start_time_key,
                response_cost=cost,
            )
            for cost in costs
        ]
    )

    spend = await limiter.dual_cache.async_get_cache(spend_key)
    assert float(spend) == pytest.approx(sum(costs))


@pytest.mark.asyncio
async def test_new_budget_window_claims_through_redis_when_available(disable_budget_sync):
    """
    With Redis wired the reset must go through the atomic claim, and the local
    cache must reflect whatever Redis decided.
    """
    redis_cache = FakeAtomicRedisCache()
    limiter = _budget_limiter(redis_cache=redis_cache)
    spend_key = "provider_spend:synthetic:1h"
    start_time_key = "provider_budget_start_time:synthetic"

    claimed_start = await limiter._handle_new_budget_window(
        spend_key=spend_key,
        start_time_key=start_time_key,
        current_time=1000.0,
        response_cost=0.05,
        ttl_seconds=3600,
    )
    losing_start = await limiter._handle_new_budget_window(
        spend_key=spend_key,
        start_time_key=start_time_key,
        current_time=1000.0,
        response_cost=0.07,
        ttl_seconds=3600,
    )

    assert claimed_start == 1000.0
    assert losing_start == 1000.0
    assert float(redis_cache.store[spend_key]) == pytest.approx(0.12)
    assert float(limiter.dual_cache.in_memory_cache.get_cache(spend_key)) == pytest.approx(0.12)
    assert [call["keys"] for call in redis_cache.script_calls] == [[start_time_key, spend_key]] * 2


@pytest.mark.asyncio
async def test_new_budget_window_falls_back_to_local_reset_when_redis_script_fails(disable_budget_sync):
    class BrokenRedisCache(FakeAtomicRedisCache):
        def async_register_script(self, script: str):
            async def run_script(keys, args, client=None):
                raise ConnectionError("redis is down")

            return run_script

    limiter = _budget_limiter(redis_cache=BrokenRedisCache())
    spend_key = "provider_spend:synthetic:1h"

    start_time = await limiter._handle_new_budget_window(
        spend_key=spend_key,
        start_time_key="provider_budget_start_time:synthetic",
        current_time=1000.0,
        response_cost=0.05,
        ttl_seconds=3600,
    )

    assert start_time == 1000.0
    assert float(await limiter.dual_cache.async_get_cache(spend_key)) == pytest.approx(0.05)


@pytest.mark.asyncio
async def test_push_in_memory_increments_waits_for_redis_and_keeps_new_increments(disable_budget_sync):
    """
    The queue may only be drained once Redis has actually acknowledged the
    increments, and increments queued while that write is in flight must survive.
    """
    redis_cache = FakeAtomicRedisCache()
    limiter = _budget_limiter(redis_cache=redis_cache)

    pipeline_started = asyncio.Event()
    release_pipeline = asyncio.Event()
    original_increment_pipeline = redis_cache.async_increment_pipeline

    async def blocking_increment_pipeline(increment_list, **kwargs):
        pipeline_started.set()
        await release_pipeline.wait()
        return await original_increment_pipeline(increment_list, **kwargs)

    redis_cache.async_increment_pipeline = blocking_increment_pipeline  # type: ignore[method-assign]

    await limiter._increment_spend_in_current_window(
        spend_key="provider_spend:synthetic:1h", response_cost=0.05, ttl=3600
    )
    push_task = asyncio.create_task(limiter._push_in_memory_increments_to_redis())

    await pipeline_started.wait()
    await limiter._increment_spend_in_current_window(
        spend_key="provider_spend:synthetic:1h", response_cost=0.07, ttl=3600
    )
    release_pipeline.set()
    await push_task

    assert [op["increment_value"] for op in redis_cache.increment_pipeline_calls[0]] == [0.05]
    assert [op["increment_value"] for op in limiter.redis_increment_operation_queue] == [0.07]


@pytest.mark.asyncio
async def test_push_in_memory_increments_retains_queue_when_redis_write_fails(disable_budget_sync):
    redis_cache = FakeAtomicRedisCache()
    limiter = _budget_limiter(redis_cache=redis_cache)

    async def failing_increment_pipeline(increment_list, **kwargs):
        raise ConnectionError("redis is down")

    redis_cache.async_increment_pipeline = failing_increment_pipeline  # type: ignore[method-assign]

    await limiter._increment_spend_in_current_window(
        spend_key="provider_spend:synthetic:1h", response_cost=0.05, ttl=3600
    )
    await limiter._push_in_memory_increments_to_redis()

    assert [op["increment_value"] for op in limiter.redis_increment_operation_queue] == [0.05]
