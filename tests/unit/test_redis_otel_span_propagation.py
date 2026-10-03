"""`async_rpush` / `async_lpop` must parent their service spans to the caller's.

Both accept and document `parent_otel_span`, but did not forward it to the
service logger, so the Redis span was orphaned from the request trace.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.caching.redis_cache import RedisCache, RedisCircuitBreaker

SENTINEL_SPAN = object()


def _cache(redis_client) -> RedisCache:
    cache = RedisCache.__new__(RedisCache)
    cache.namespace = None
    cache.redis_version = "7.0.0"
    cache.service_logger_obj = MagicMock()
    cache.service_logger_obj.async_service_success_hook = AsyncMock()
    cache.service_logger_obj.async_service_failure_hook = AsyncMock()
    cache._async_commands = MagicMock(return_value=redis_client)  # type: ignore[method-assign]
    cache._circuit_breaker = RedisCircuitBreaker(failure_threshold=5, recovery_timeout=60)
    return cache


async def _drain() -> None:
    """Hooks are fired via asyncio.create_task."""
    await asyncio.sleep(0)
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_async_rpush_success_forwards_parent_span() -> None:
    client = MagicMock()
    client.rpush = AsyncMock(return_value=1)
    cache = _cache(client)

    await cache.async_rpush("k", ["v"], parent_otel_span=SENTINEL_SPAN)
    await _drain()

    hook = cache.service_logger_obj.async_service_success_hook
    hook.assert_awaited_once()
    assert hook.await_args.kwargs["parent_otel_span"] is SENTINEL_SPAN


@pytest.mark.asyncio
async def test_async_rpush_failure_forwards_parent_span() -> None:
    client = MagicMock()
    client.rpush = AsyncMock(side_effect=RuntimeError("redis down"))
    cache = _cache(client)

    with pytest.raises(RuntimeError):
        await cache.async_rpush("k", ["v"], parent_otel_span=SENTINEL_SPAN)
    await _drain()

    hook = cache.service_logger_obj.async_service_failure_hook
    hook.assert_awaited_once()
    assert hook.await_args.kwargs["parent_otel_span"] is SENTINEL_SPAN


@pytest.mark.asyncio
async def test_async_lpop_success_forwards_parent_span() -> None:
    client = MagicMock()
    client.lpop = AsyncMock(return_value=b"v")
    cache = _cache(client)

    await cache.async_lpop("k", parent_otel_span=SENTINEL_SPAN)
    await _drain()

    hook = cache.service_logger_obj.async_service_success_hook
    hook.assert_awaited_once()
    assert hook.await_args.kwargs["parent_otel_span"] is SENTINEL_SPAN


@pytest.mark.asyncio
async def test_async_lpop_failure_forwards_parent_span() -> None:
    client = MagicMock()
    client.lpop = AsyncMock(side_effect=RuntimeError("redis down"))
    cache = _cache(client)

    with pytest.raises(RuntimeError):
        await cache.async_lpop("k", parent_otel_span=SENTINEL_SPAN)
    await _drain()

    hook = cache.service_logger_obj.async_service_failure_hook
    hook.assert_awaited_once()
    assert hook.await_args.kwargs["parent_otel_span"] is SENTINEL_SPAN
