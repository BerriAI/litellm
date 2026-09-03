from datetime import datetime, timezone

import pytest

from litellm.caching.caching import DualCache
from litellm.proxy.hooks.dynamic_rate_limiter import (
    DynamicRateLimiterCache,
    _PROXY_DynamicRateLimitHandler,
)
from litellm.utils import utc_epoch_minute


@pytest.mark.asyncio
async def test_sadd_and_get_share_injected_clock_window():
    dual_cache = DualCache()
    window_time = datetime(2024, 1, 1, 10, 30, 0, tzinfo=timezone.utc)
    cache = DynamicRateLimiterCache(
        cache=dual_cache,
        time_fn=lambda: window_time,
    )
    await cache.async_set_cache_sadd(model="my-fake-model", value=["p1", "p2", "p3"])
    assert await cache.async_get_cache(model="my-fake-model") == 3
    key = f"v2:{utc_epoch_minute(window_time)}:my-fake-model"
    assert await dual_cache.async_get_cache(key=key) is not None


@pytest.mark.asyncio
async def test_minute_rollover_between_sadd_and_get_reads_empty_window():
    ticks = iter(
        (
            datetime(2024, 1, 1, 10, 30, 59, 999999, tzinfo=timezone.utc),
            datetime(2024, 1, 1, 10, 31, 0, 0, tzinfo=timezone.utc),
        )
    )
    cache = DynamicRateLimiterCache(cache=DualCache(), time_fn=lambda: next(ticks))
    await cache.async_set_cache_sadd(model="my-fake-model", value=["p1"])
    assert await cache.async_get_cache(model="my-fake-model") is None


@pytest.mark.asyncio
async def test_handler_threads_time_fn_to_internal_cache():
    handler = _PROXY_DynamicRateLimitHandler(
        internal_usage_cache=DualCache(),
        time_fn=lambda: datetime(2024, 1, 1, 10, 30, 0, tzinfo=timezone.utc),
    )
    await handler.internal_usage_cache.async_set_cache_sadd(model="my-fake-model", value=["p1", "p2"])
    assert await handler.internal_usage_cache.async_get_cache(model="my-fake-model") == 2
