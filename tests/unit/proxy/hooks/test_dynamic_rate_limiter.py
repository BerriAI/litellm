from datetime import datetime, timezone
from typing import Final

import pytest

from litellm import Router
from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.dynamic_rate_limiter import (
    DynamicRateLimiterCache,
    _PROXY_DynamicRateLimitHandler,
)
from litellm.types.utils import HiddenParams, ModelResponse


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


@pytest.mark.asyncio
async def test_success_hook_updates_existing_hidden_params_storage() -> None:
    model_id: Final = "rate-limit-deployment"
    router: Final = Router(
        model_list=[
            {
                "model_name": "my-fake-model",
                "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "test-key", "tpm": 100, "rpm": 10},
                "model_info": {"id": model_id},
            }
        ]
    )
    handler: Final = _PROXY_DynamicRateLimitHandler(internal_usage_cache=DualCache())
    handler.update_variables(llm_router=router)
    response: Final = ModelResponse()
    hidden_params: Final = HiddenParams(model_id=model_id)
    response._hidden_params = hidden_params

    result: Final = await handler.async_post_call_success_hook(
        data={},
        user_api_key_dict=UserAPIKeyAuth(metadata={}),
        response=response,
    )

    assert result is response
    assert response._hidden_params is hidden_params
    assert response.hidden_params["additional_headers"]["x-litellm-model_group"] == "my-fake-model"
