"""
Unit Tests for the max parallel request limiter v3 for the proxy
"""
import asyncio
import os
import sys
from datetime import datetime

import pytest
from fastapi import HTTPException

import litellm
from litellm import Router
from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    _PROXY_MaxParallelRequestsHandler_v3 as _PROXY_MaxParallelRequestsHandler,
)
from litellm.proxy.utils import InternalUsageCache, ProxyLogging, hash_token


@pytest.mark.flaky(reruns=3)
@pytest.mark.asyncio
async def test_normal_router_call_v3(monkeypatch):
    """
    Test normal router call with parallel request limiter v3
    """
    model_list = [
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "azure/gpt-turbo",
                "api_key": "os.environ/AZURE_FRANCE_API_KEY",
                "api_base": "https://openai-france-1234.openai.azure.com",
                "rpm": 1440,
            },
            "model_info": {"id": 1},
        },
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "azure/gpt-35-turbo",
                "api_key": "os.environ/AZURE_EUROPE_API_KEY",
                "api_base": "https://my-endpoint-europe-berri-992.openai.azure.com",
                "rpm": 6,
            },
            "model_info": {"id": 2},
        },
    ]
    router = Router(
        model_list=model_list,
        set_verbose=False,
        num_retries=3,
    )  # type: ignore

    _api_key = "sk-12345"
    _api_key = hash_token(_api_key)
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key, rpm_limit=2, window_size=2)
    local_cache = DualCache()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )
    monkeypatch.setattr(litellm, "callbacks", [parallel_request_handler])

    # First request should succeed
    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )

    # Second request should succeed
    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )

    # Third request should fail
    with pytest.raises(HTTPException) as exc_info:
        await parallel_request_handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={},
            call_type="",
        )
    assert exc_info.value.status_code == 429
    assert "Rate limit exceeded" in str(exc_info.value.detail)

    # normal call
    response = await router.acompletion(
        model="azure-model",
        messages=[{"role": "user", "content": "Hey, how's it going?"}],
        metadata={"user_api_key": _api_key},
        mock_response="hello",
    )
    await asyncio.sleep(1)  # success is done in a separate thread


@pytest.mark.flaky(reruns=3)
@pytest.mark.asyncio
async def test_streaming_router_call_v3(monkeypatch):
    """
    Test streaming router call with parallel request limiter v3
    """
    model_list = [
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "azure/gpt-turbo",
                "api_key": "os.environ/AZURE_FRANCE_API_KEY",
                "api_base": "https://openai-france-1234.openai.azure.com",
                "rpm": 1440,
            },
            "model_info": {"id": 1},
        },
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "azure/gpt-35-turbo",
                "api_key": "os.environ/AZURE_EUROPE_API_KEY",
                "api_base": "https://my-endpoint-europe-berri-992.openai.azure.com",
                "rpm": 6,
            },
            "model_info": {"id": 2},
        },
    ]
    router = Router(
        model_list=model_list,
        set_verbose=False,
        num_retries=3,
    )  # type: ignore

    _api_key = "sk-12345"
    _api_key = hash_token(_api_key)
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key, rpm_limit=2, window_size=2)
    local_cache = DualCache()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )
    monkeypatch.setattr(litellm, "callbacks", [parallel_request_handler])

    # First request should succeed
    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )

    # streaming call
    response = await router.acompletion(
        model="azure-model",
        messages=[{"role": "user", "content": "Hey, how's it going?"}],
        stream=True,
        metadata={"user_api_key": _api_key},
        mock_response="hello",
    )
    async for chunk in response:
        continue
    await asyncio.sleep(1)  # success is done in a separate thread

    # Second request should succeed
    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )

    # Third request should fail
    with pytest.raises(HTTPException) as exc_info:
        await parallel_request_handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={},
            call_type="",
        )
    assert exc_info.value.status_code == 429
    assert "Rate limit exceeded" in str(exc_info.value.detail)


@pytest.mark.flaky(reruns=3)
@pytest.mark.asyncio
async def test_sliding_window_rate_limit_v3(monkeypatch):
    """
    Test the sliding window rate limiting functionality
    """
    monkeypatch.setenv("LITELLM_RATE_LIMIT_WINDOW_SIZE", "2")
    _api_key = "sk-12345"
    _api_key = hash_token(_api_key)
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key, rpm_limit=2)
    local_cache = DualCache()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )

    # First request should succeed
    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )

    # Second request should succeed
    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )

    # Third request should fail
    with pytest.raises(HTTPException) as exc_info:
        await parallel_request_handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={},
            call_type="",
        )
    assert exc_info.value.status_code == 429
    assert "Rate limit exceeded" in str(exc_info.value.detail)

    # Wait for window to expire (2 seconds)
    await asyncio.sleep(3)

    print("WAITED 3 seconds")

    print(f"local_cache: {local_cache.in_memory_cache.cache_dict}")

    # After window expires, should be able to make requests again
    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )


@pytest.mark.flaky(reruns=3)
@pytest.mark.asyncio
async def test_rate_limit_headers_v3():
    """
    Test that rate limit headers are properly set in the response
    """
    _api_key = "sk-12345"
    _api_key = hash_token(_api_key)
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key, rpm_limit=2, window_size=2)
    local_cache = DualCache()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )

    # Make a request
    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )

    # Create a mock response
    class MockResponse:
        def __init__(self):
            self._hidden_params = {"additional_headers": {}}

    response = MockResponse()

    # Call post-call hook
    await parallel_request_handler.async_post_call_success_hook(
        data={}, user_api_key_dict=user_api_key_dict, response=response
    )

    # Verify headers are set
    assert hasattr(response, "_hidden_params")
    assert "additional_headers" in response._hidden_params
    headers = response._hidden_params["additional_headers"]
    assert "x-ratelimit-api_key-remaining-requests" in headers
    assert "x-ratelimit-api_key-limit-requests" in headers
    assert headers["x-ratelimit-api_key-limit-requests"] == 2
    assert headers["x-ratelimit-api_key-remaining-requests"] == 1


@pytest.mark.asyncio
async def test_rate_limiter_script_return_values_v3(monkeypatch):
    """
    Test that the rate limiter script returns both counter and window values correctly
    """
    monkeypatch.setenv("LITELLM_RATE_LIMIT_WINDOW_SIZE", "2")
    _api_key = "sk-12345"
    _api_key = hash_token(_api_key)
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key, rpm_limit=2)
    local_cache = DualCache()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )

    # Make first request
    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )

    # Verify both counter and window values are stored in cache
    window_key = f"{{api_key:{_api_key}}}:window"
    counter_key = f"{{api_key:{_api_key}}}:requests"

    window_value = await local_cache.async_get_cache(key=window_key)
    counter_value = await local_cache.async_get_cache(key=counter_key)

    assert window_value is not None, "Window value should be stored in cache"
    assert counter_value is not None, "Counter value should be stored in cache"
    assert counter_value == 1, "Counter should be 1 after first request"

    # Make second request
    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )

    # Verify counter increased but window stayed same
    new_window_value = await local_cache.async_get_cache(key=window_key)
    new_counter_value = await local_cache.async_get_cache(key=counter_key)

    assert (
        new_window_value == window_value
    ), "Window value should not change within window"
    assert new_counter_value == 2, "Counter should be 2 after second request"

    # Wait for window to expire
    await asyncio.sleep(3)

    # Make request after window expiry
    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )

    # Verify new window and reset counter
    final_window_value = await local_cache.async_get_cache(key=window_key)
    final_counter_value = await local_cache.async_get_cache(key=counter_key)

    assert final_window_value != window_value, "Window value should change after expiry"
    assert final_counter_value == 1, "Counter should reset to 1 after window expiry"
