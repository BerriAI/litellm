import asyncio
from typing import Any, Dict, List, Optional, Sequence

import pytest

from litellm.caching.caching import DualCache
from litellm.router_strategy.budget_limiter import RouterBudgetLimiting
from litellm.types.utils import BudgetConfig

TTL_SECONDS = 86400


class YieldingDualCache(DualCache):
    """DualCache that suspends on every read/write, so concurrent callers interleave."""

    async def async_get_cache(self, key, parent_otel_span=None, local_only: bool = False, **kwargs):
        await asyncio.sleep(0)
        return await super().async_get_cache(key, parent_otel_span, local_only, **kwargs)

    async def async_set_cache(self, key, value, local_only: bool = False, **kwargs):
        await asyncio.sleep(0)
        return await super().async_set_cache(key, value, local_only, **kwargs)


class FakeRedisCacheWithAtomicScripts:
    """
    Stand-in for RedisCache that runs registered scripts atomically over a local dict.

    Mirrors what Redis guarantees for a Lua script: the body observes and mutates the
    store without another caller interleaving, while callers still race to enter it.
    """

    def __init__(self) -> None:
        self.store: Dict[str, str] = {}
        self.registered_scripts: List[str] = []

    def async_register_script(self, script: str):
        self.registered_scripts.append(script)

        async def run_script(keys: Sequence[str], args: Sequence[Any], client: Optional[Any] = None) -> List[bytes]:
            await asyncio.sleep(0)
            start_time_key, spend_key = keys
            current_time, response_cost, ttl = str(args[0]), str(args[1]), float(args[2])

            window_start = self.store.get(start_time_key)
            if window_start is None or (float(current_time) - float(window_start)) > ttl:
                self.store[start_time_key] = current_time
                self.store[spend_key] = response_cost
                return [current_time.encode(), response_cost.encode()]

            new_spend = str(float(self.store.get(spend_key, "0")) + float(response_cost))
            self.store[spend_key] = new_spend
            return [window_start.encode(), new_spend.encode()]

        return run_script


@pytest.fixture
def disable_budget_sync(monkeypatch):
    async def noop(*args, **kwargs):
        return None

    monkeypatch.setattr(
        "litellm.router_strategy.budget_limiter.RouterBudgetLimiting.periodic_sync_in_memory_spend_with_redis",
        noop,
    )


@pytest.mark.asyncio
async def test_concurrent_expired_window_resets_keep_every_response_cost(disable_budget_sync):
    """Every response crossing an expired window boundary must be counted, not just the last one."""
    budget_limiter = RouterBudgetLimiting(
        dual_cache=YieldingDualCache(),
        provider_budget_config={"openai": BudgetConfig(budget_duration="1d", max_budget=100)},
    )
    spend_key = "provider_spend:openai:1d"
    start_time_key = "provider_budget_start_time:openai"
    now = 1_000_000.0

    await budget_limiter.dual_cache.async_set_cache(
        key=start_time_key, value=now - (2 * TTL_SECONDS), ttl=10 * TTL_SECONDS
    )
    await budget_limiter.dual_cache.async_set_cache(key=spend_key, value=7.0, ttl=10 * TTL_SECONDS)

    costs = (0.5, 0.25, 0.125)
    await asyncio.gather(
        *[
            budget_limiter._handle_new_budget_window(
                spend_key=spend_key,
                start_time_key=start_time_key,
                current_time=now,
                response_cost=cost,
                ttl_seconds=TTL_SECONDS,
            )
            for cost in costs
        ]
    )

    spend = await budget_limiter.dual_cache.async_get_cache(spend_key)
    assert float(spend) == pytest.approx(sum(costs))

    window_start = await budget_limiter.dual_cache.async_get_cache(start_time_key)
    assert float(window_start) == now


@pytest.mark.asyncio
async def test_concurrent_expired_window_resets_keep_every_response_cost_with_redis(disable_budget_sync):
    """With Redis the window reset and the increment are delegated to one atomic script."""
    fake_redis = FakeRedisCacheWithAtomicScripts()
    fake_redis.store["provider_budget_start_time:openai"] = str(1_000_000.0 - (2 * TTL_SECONDS))
    fake_redis.store["provider_spend:openai:1d"] = "7.0"

    budget_limiter = RouterBudgetLimiting(
        dual_cache=DualCache(redis_cache=fake_redis),
        provider_budget_config={"openai": BudgetConfig(budget_duration="1d", max_budget=100)},
    )
    spend_key = "provider_spend:openai:1d"
    start_time_key = "provider_budget_start_time:openai"
    now = 1_000_000.0

    costs = (0.5, 0.25, 0.125)
    window_starts = await asyncio.gather(
        *[
            budget_limiter._handle_new_budget_window(
                spend_key=spend_key,
                start_time_key=start_time_key,
                current_time=now,
                response_cost=cost,
                ttl_seconds=TTL_SECONDS,
            )
            for cost in costs
        ]
    )

    assert float(fake_redis.store[spend_key]) == pytest.approx(sum(costs))
    assert float(fake_redis.store[start_time_key]) == now
    assert window_starts == [now, now, now]

    in_memory_spend = await budget_limiter.dual_cache.in_memory_cache.async_get_cache(spend_key)
    assert float(in_memory_spend) == pytest.approx(sum(costs))


@pytest.mark.asyncio
async def test_expired_window_reset_drops_previous_window_spend(disable_budget_sync):
    """A single response crossing the boundary still starts the new window from its own cost."""
    budget_limiter = RouterBudgetLimiting(
        dual_cache=DualCache(),
        provider_budget_config={"openai": BudgetConfig(budget_duration="1d", max_budget=100)},
    )
    spend_key = "provider_spend:openai:1d"
    start_time_key = "provider_budget_start_time:openai"
    now = 1_000_000.0

    await budget_limiter.dual_cache.async_set_cache(
        key=start_time_key, value=now - (2 * TTL_SECONDS), ttl=10 * TTL_SECONDS
    )
    await budget_limiter.dual_cache.async_set_cache(key=spend_key, value=7.0, ttl=10 * TTL_SECONDS)

    window_start = await budget_limiter._handle_new_budget_window(
        spend_key=spend_key,
        start_time_key=start_time_key,
        current_time=now,
        response_cost=0.5,
        ttl_seconds=TTL_SECONDS,
    )

    assert window_start == now
    spend = await budget_limiter.dual_cache.async_get_cache(spend_key)
    assert float(spend) == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_concurrent_success_events_across_window_boundary_keep_every_response_cost(disable_budget_sync):
    """End-to-end through _increment_spend_for_key: expired window, concurrent responses."""
    budget_limiter = RouterBudgetLimiting(
        dual_cache=YieldingDualCache(),
        provider_budget_config={"openai": BudgetConfig(budget_duration="1d", max_budget=100)},
    )
    budget_config = BudgetConfig(budget_duration="1d", max_budget=100)
    spend_key = "provider_spend:openai:1d"
    start_time_key = "provider_budget_start_time:openai"

    await budget_limiter.dual_cache.async_set_cache(key=start_time_key, value=0.0, ttl=10 * TTL_SECONDS)

    costs = (0.5, 0.25)
    await asyncio.gather(
        *[
            budget_limiter._increment_spend_for_key(
                budget_config=budget_config,
                spend_key=spend_key,
                start_time_key=start_time_key,
                response_cost=cost,
            )
            for cost in costs
        ]
    )

    spend = await budget_limiter.dual_cache.async_get_cache(spend_key)
    assert float(spend) == pytest.approx(sum(costs))
