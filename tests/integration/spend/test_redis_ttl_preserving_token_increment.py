import os
import uuid
from typing import Final

import pytest
from redis import Redis

from litellm.caching.caching import DualCache
from litellm.caching.redis_cache import RedisCache
from litellm.proxy.hooks.parallel_request_limiter_v3 import OpenWindowIncrementOperation
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    PROXY_MaxParallelRequestsHandler_v3 as _PROXY_MaxParallelRequestsHandler,
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


@pytest.mark.asyncio
async def test_open_window_token_increment_charges_only_scopes_whose_window_is_open() -> None:
    redis_host: Final = os.environ["REDIS_HOST"]
    redis_port: Final = int(os.environ["REDIS_PORT"])
    redis_cache: Final = RedisCache(host=redis_host, port=redis_port)
    handler: Final = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache(redis_cache=redis_cache))
    )
    suffix: Final = uuid.uuid4().hex[:8]
    limited: Final = f"{{user:limited-{suffix}}}"
    unlimited: Final = f"{{user:unlimited-{suffix}}}"
    ungated: Final = f"{{team:ungated-{suffix}}}"

    with Redis(host=redis_host, port=redis_port) as raw:
        try:
            assert raw.set(f"{limited}:window", "1", ex=60) is True
            assert raw.set(f"{limited}:tokens", "5", ex=60) is True
            await handler.async_increment_tokens_with_ttl_preservation(
                pipeline_operations=[
                    RedisPipelineIncrementOperation(key=f"{ungated}:tokens", increment_value=20, ttl=60),
                    *(
                        OpenWindowIncrementOperation(
                            key=f"{scope}:tokens", increment_value=20, ttl=60, open_window_key=f"{scope}:window"
                        )
                        for scope in (limited, unlimited)
                    ),
                ]
            )

            assert raw.mget([f"{limited}:tokens", f"{unlimited}:tokens", f"{unlimited}:window"]) == [b"25", None, None]
            assert raw.get(f"{ungated}:tokens") == b"20"
            assert 0 < raw.ttl(f"{limited}:tokens") <= 60
            assert 0 < raw.ttl(f"{ungated}:tokens") <= 60
        finally:
            raw.delete(
                *(f"{scope}:{suffix_key}" for scope in (limited, unlimited) for suffix_key in ("window", "tokens")),
                f"{ungated}:tokens",
            )
