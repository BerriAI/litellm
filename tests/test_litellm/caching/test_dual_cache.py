import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.caching.dual_cache import DualCache
from litellm.caching.redis_cache import RedisCache


@pytest.mark.asyncio
async def test_dual_cache_async_batch_get_cache_coalesces_concurrent_redis_reads():
    dual_cache = DualCache(
        redis_cache=MagicMock(spec=RedisCache), default_redis_batch_cache_expiry=10
    )
    keys = ["shared_a", "shared_b"]
    start_gate = asyncio.Event()

    async def _mock_async_batch_get_cache(key_list, parent_otel_span=None):
        await asyncio.sleep(0.05)
        return {k: None for k in key_list}

    with patch.object(
        dual_cache.redis_cache,
        "async_batch_get_cache",
        new=AsyncMock(side_effect=_mock_async_batch_get_cache),
    ) as mock_async_batch_get_cache:

        async def worker():
            await start_gate.wait()
            return await dual_cache.async_batch_get_cache(keys=keys)

        tasks = [asyncio.create_task(worker()) for _ in range(50)]
        start_gate.set()
        await asyncio.gather(*tasks)

        assert mock_async_batch_get_cache.call_count == 1


@pytest.mark.asyncio
async def test_dual_cache_async_batch_get_cache_rolls_back_redis_reservation_on_error():
    dual_cache = DualCache(
        redis_cache=MagicMock(spec=RedisCache), default_redis_batch_cache_expiry=10
    )
    keys = ["shared_a", "shared_b"]

    with patch.object(
        dual_cache.redis_cache,
        "async_batch_get_cache",
        new=AsyncMock(side_effect=RuntimeError("redis unavailable")),
    ) as mock_async_batch_get_cache:
        first_result = await dual_cache.async_batch_get_cache(keys=keys)
        second_result = await dual_cache.async_batch_get_cache(keys=keys)

        assert first_result is None
        assert second_result is None
        assert mock_async_batch_get_cache.call_count == 2
        assert "shared_a" not in dual_cache.last_redis_batch_access_time
        assert "shared_b" not in dual_cache.last_redis_batch_access_time
