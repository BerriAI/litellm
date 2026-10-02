import os
import uuid
from typing import Final

import pytest
from redis import Redis

from litellm.caching.caching import DualCache
from litellm.caching.redis_cache import RedisCache
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    _PROXY_MaxParallelRequestsHandler_v3 as _PROXY_MaxParallelRequestsHandler,
)
from litellm.proxy.utils import InternalUsageCache
from litellm.types.caching import RedisPipelineIncrementOperation


@pytest.mark.asyncio
async def test_async_increment_tokens_with_ttl_preservation() -> None:
    redis_host: Final = os.environ["REDIS_HOST"]
    redis_port: Final = int(os.environ["REDIS_PORT"])
    redis_cache: Final = RedisCache(host=redis_host, port=redis_port)
    handler: Final = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache(redis_cache=redis_cache))
    )
    assert handler.token_increment_script is not None

    suffix: Final = uuid.uuid4().hex[:8]
    key_with_ttl: Final = f"{{test_ttl}}:with_ttl:{suffix}"
    key_without_ttl: Final = f"{{test_ttl}}:without_ttl:{suffix}"

    try:
        await redis_cache.async_delete_cache(key_with_ttl)
        await redis_cache.async_delete_cache(key_without_ttl)

        await handler.async_increment_tokens_with_ttl_preservation(
            pipeline_operations=[
                RedisPipelineIncrementOperation(key=key_with_ttl, increment_value=10.0, ttl=60),
                RedisPipelineIncrementOperation(key=key_without_ttl, increment_value=5.0, ttl=None),
            ]
        )

        assert await redis_cache.async_get_cache(key_with_ttl) == 10.0
        assert await redis_cache.async_get_cache(key_without_ttl) == 5.0
        first_ttl: Final = await redis_cache.async_get_ttl(key_with_ttl)
        assert first_ttl is not None and 0 < first_ttl <= 60
        assert await redis_cache.async_get_ttl(key_without_ttl) is None

        with Redis(host=redis_host, port=redis_port) as raw:
            assert raw.expire(key_with_ttl, 30, xx=True) == 1

        await handler.async_increment_tokens_with_ttl_preservation(
            pipeline_operations=[
                RedisPipelineIncrementOperation(key=key_with_ttl, increment_value=15.0, ttl=60),
                RedisPipelineIncrementOperation(key=key_without_ttl, increment_value=7.0, ttl=None),
            ]
        )

        assert await redis_cache.async_get_cache(key_with_ttl) == 25.0
        assert await redis_cache.async_get_cache(key_without_ttl) == 12.0
        second_ttl: Final = await redis_cache.async_get_ttl(key_with_ttl)
        assert second_ttl is not None and 0 < second_ttl <= 30
        assert await redis_cache.async_get_ttl(key_without_ttl) is None
    finally:
        await redis_cache.async_delete_cache(key_with_ttl)
        await redis_cache.async_delete_cache(key_without_ttl)
