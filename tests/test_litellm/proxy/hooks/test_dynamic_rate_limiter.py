import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.hooks.dynamic_rate_limiter import (
    DynamicRateLimiterCache,
    _PROXY_DynamicRateLimitHandler,
)


@pytest.mark.asyncio
async def test_sadd_and_get_share_injected_clock_window():
    dual_cache = DualCache()
    cache = DynamicRateLimiterCache(
        cache=dual_cache,
        time_fn=lambda: datetime(2024, 1, 1, 10, 30, 0, tzinfo=timezone.utc),
    )
    await cache.async_set_cache_sadd(model="my-fake-model", value=["p1", "p2", "p3"])
    assert await cache.async_get_cache(model="my-fake-model") == 3
    assert await dual_cache.async_get_cache(key="10-30:my-fake-model") is not None


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


def fake_handler(*, cache_error=None, tpm=100, rpm=10):
    usage_cache = MagicMock()
    usage_cache.async_get_cache = AsyncMock(side_effect=cache_error, return_value=None)
    handler = _PROXY_DynamicRateLimitHandler(internal_usage_cache=usage_cache)
    router = MagicMock()
    router.get_model_group_info.return_value = MagicMock(tpm=tpm, rpm=rpm)
    router.get_model_group_usage = AsyncMock(return_value=(0, 0))
    handler.update_variables(router)
    return handler, usage_cache, router


@pytest.mark.asyncio
async def test_cache_lookup_failure_rejects_before_router_call():
    handler, _, router = fake_handler(cache_error=ConnectionError("cache offline"))
    with pytest.raises(HTTPException) as exc:
        await handler.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(api_key="sk-test", metadata={}),
            cache=DualCache(),
            data={"model": "test-model"},
            call_type="completion",
        )
    assert exc.value.status_code == 503
    assert "cache offline" not in str(exc.value.detail)
    router.get_model_group_usage.assert_not_awaited()


@pytest.mark.asyncio
async def test_zero_quota_still_raises_rate_limit():
    handler, _, router = fake_handler(tpm=0)
    with pytest.raises(ProxyRateLimitError) as exc:
        await handler.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(api_key="sk-test", metadata={}),
            cache=DualCache(),
            data={"model": "gpt-4o-mini"},
            call_type="completion",
        )
    assert exc.value.status_code == 429
    router.get_model_group_usage.assert_awaited_once()


@pytest.mark.asyncio
async def test_unlimited_capacity_still_admits_request():
    handler, _, router = fake_handler(tpm=None, rpm=None)
    user = UserAPIKeyAuth(api_key="sk-test", metadata={})
    with patch("litellm.proxy.hooks.dynamic_rate_limiter.asyncio.create_task") as create_task:
        assert await handler.async_pre_call_hook(user, DualCache(), {"model": "test-model"}, "completion") is None
        create_task.assert_not_called()
    router.get_model_group_usage.assert_awaited_once()


@pytest.mark.asyncio
async def test_successful_lookup_keeps_active_project_accounting():
    handler, _, _ = fake_handler()
    user = UserAPIKeyAuth(api_key="sk-test", token="token-123", metadata={})
    handler.internal_usage_cache.async_set_cache_sadd = AsyncMock()  # type: ignore[method-assign]
    await handler.async_pre_call_hook(user, DualCache(), {"model": "test-model"}, "completion")
    await asyncio.sleep(0)
    handler.internal_usage_cache.async_set_cache_sadd.assert_awaited_once_with(model="test-model", value=[user.token])
