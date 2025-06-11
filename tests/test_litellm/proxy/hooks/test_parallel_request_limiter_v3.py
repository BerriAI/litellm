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
async def test_sliding_window_rate_limit_v3(monkeypatch):
    """
    Test the sliding window rate limiting functionality
    """
    monkeypatch.setenv("LITELLM_RATE_LIMIT_WINDOW_SIZE", "2")
    _api_key = "sk-12345"
    _api_key = hash_token(_api_key)
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key, rpm_limit=3)
    local_cache = DualCache()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )

    # Mock the batch_rate_limiter_script to simulate window expiry and use correct key construction
    window_starts = {}

    async def mock_batch_rate_limiter(*args, **kwargs):
        keys = kwargs.get("keys") if "keys" in kwargs else args[0]
        now = kwargs.get("args")[0] if "args" in kwargs else args[1][0]
        window_size = kwargs.get("args")[1] if "args" in kwargs else args[1][1]
        results = []
        for i in range(0, len(keys), 3):
            window_key = keys[i]
            counter_key = keys[i + 1]
            # Simulate window expiry
            prev_window = window_starts.get(window_key)
            prev_counter = await local_cache.async_get_cache(key=counter_key) or 0
            if prev_window is None or (now - prev_window) >= window_size:
                # Window expired, reset
                window_starts[window_key] = now
                new_counter = 1
                await local_cache.async_set_cache(
                    key=window_key, value=now, ttl=window_size
                )
                await local_cache.async_set_cache(
                    key=counter_key, value=new_counter, ttl=window_size
                )
            else:
                new_counter = prev_counter + 1
                await local_cache.async_set_cache(
                    key=counter_key, value=new_counter, ttl=window_size
                )
            results.append(now)
            results.append(new_counter)
        return results

    parallel_request_handler.batch_rate_limiter_script = mock_batch_rate_limiter

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


@pytest.mark.asyncio
async def test_rate_limiter_script_return_values_v3(monkeypatch):
    """
    Test that the rate limiter script returns both counter and window values correctly
    """
    monkeypatch.setenv("LITELLM_RATE_LIMIT_WINDOW_SIZE", "2")
    _api_key = "sk-12345"
    _api_key = hash_token(_api_key)
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key, rpm_limit=3)
    local_cache = DualCache()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )

    # Mock the batch_rate_limiter_script to simulate window expiry and use correct key construction
    window_starts = {}

    async def mock_batch_rate_limiter(*args, **kwargs):
        keys = kwargs.get("keys") if "keys" in kwargs else args[0]
        now = kwargs.get("args")[0] if "args" in kwargs else args[1][0]
        window_size = kwargs.get("args")[1] if "args" in kwargs else args[1][1]
        results = []
        for i in range(0, len(keys), 3):
            window_key = keys[i]
            counter_key = keys[i + 1]
            # Simulate window expiry
            prev_window = window_starts.get(window_key)
            prev_counter = await local_cache.async_get_cache(key=counter_key) or 0
            if prev_window is None or (now - prev_window) >= window_size:
                # Window expired, reset
                window_starts[window_key] = now
                new_counter = 1
                await local_cache.async_set_cache(
                    key=window_key, value=now, ttl=window_size
                )
                await local_cache.async_set_cache(
                    key=counter_key, value=new_counter, ttl=window_size
                )
            else:
                new_counter = prev_counter + 1
                await local_cache.async_set_cache(
                    key=counter_key, value=new_counter, ttl=window_size
                )
            results.append(now)
            results.append(new_counter)
        return results

    parallel_request_handler.batch_rate_limiter_script = mock_batch_rate_limiter

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


@pytest.mark.parametrize(
    "rate_limit_object",
    [
        "api_key",
        "model_per_key",
        "user",
        "end_user",
        "team",
    ],
)
@pytest.mark.flaky(reruns=3)
@pytest.mark.asyncio
async def test_normal_router_call_tpm_v3(monkeypatch, rate_limit_object):
    """
    Test normal router call with parallel request limiter v3 for TPM rate limiting
    """
    monkeypatch.setenv("LITELLM_RATE_LIMIT_WINDOW_SIZE", "2")
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
    if rate_limit_object == "api_key":
        user_api_key_dict = UserAPIKeyAuth(api_key=_api_key, tpm_limit=10)
    elif rate_limit_object == "user":
        user_api_key_dict = UserAPIKeyAuth(user_id="12345", user_tpm_limit=10)
    elif rate_limit_object == "team":
        user_api_key_dict = UserAPIKeyAuth(team_id="12345", team_tpm_limit=10)
    elif rate_limit_object == "end_user":
        user_api_key_dict = UserAPIKeyAuth(end_user_id="12345", end_user_tpm_limit=10)
    elif rate_limit_object == "model_per_key":
        user_api_key_dict = UserAPIKeyAuth(
            api_key=_api_key,
            metadata={"model_tpm_limit": {"azure-model": 10}},
        )
    local_cache = DualCache()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )

    # Mock the batch_rate_limiter_script to simulate window expiry and use correct key construction
    window_starts = {}

    async def mock_batch_rate_limiter(*args, **kwargs):
        keys = kwargs.get("keys") if "keys" in kwargs else args[0]
        now = kwargs.get("args")[0] if "args" in kwargs else args[1][0]
        window_size = kwargs.get("args")[1] if "args" in kwargs else args[1][1]
        results = []
        for i in range(0, len(keys), 3):
            window_key = keys[i]
            counter_key = keys[i + 1]
            # Simulate window expiry
            prev_window = window_starts.get(window_key)
            prev_counter = await local_cache.async_get_cache(key=counter_key) or 0
            if prev_window is None or (now - prev_window) >= window_size:
                # Window expired, reset
                window_starts[window_key] = now
                new_counter = 1
                await local_cache.async_set_cache(
                    key=window_key, value=now, ttl=window_size
                )
                await local_cache.async_set_cache(
                    key=counter_key, value=new_counter, ttl=window_size
                )
            else:
                new_counter = prev_counter + 1
                await local_cache.async_set_cache(
                    key=counter_key, value=new_counter, ttl=window_size
                )
            results.append(now)
            results.append(new_counter)
        return results

    parallel_request_handler.batch_rate_limiter_script = mock_batch_rate_limiter
    monkeypatch.setattr(litellm, "callbacks", [parallel_request_handler])

    # Helper to get the correct value for key construction
    def get_value_for_key(rate_limit_object, user_api_key_dict, model_name):
        if rate_limit_object == "api_key":
            return user_api_key_dict.api_key
        elif rate_limit_object == "user":
            return user_api_key_dict.user_id
        elif rate_limit_object == "team":
            return user_api_key_dict.team_id
        elif rate_limit_object == "end_user":
            return user_api_key_dict.end_user_id
        elif rate_limit_object == "model_per_key":
            return f"{user_api_key_dict.api_key}:{model_name}"
        return None

    value = get_value_for_key(rate_limit_object, user_api_key_dict, "azure-model")
    counter_key = parallel_request_handler.create_rate_limit_keys(
        rate_limit_object, value, "tokens"
    )

    # First request should succeed
    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data={"model": "azure-model"},
        call_type="",
    )

    # normal call
    response = await router.acompletion(
        model="azure-model",
        messages=[{"role": "user", "content": "Hey, how's it going?"}],
        metadata={
            "user_api_key": _api_key,
            "user_api_key_user_id": user_api_key_dict.user_id,
            "user_api_key_team_id": user_api_key_dict.team_id,
            "user_api_key_end_user_id": user_api_key_dict.end_user_id,
        },
        mock_response="hello",
    )
    await asyncio.sleep(1)  # success is done in a separate thread

    # Verify the token count is tracked
    counter_value = await local_cache.async_get_cache(key=counter_key)
    print(f"local_cache: {local_cache.in_memory_cache.cache_dict}")

    assert (
        counter_value is not None
    ), f"Counter value should be stored in cache for {counter_key}"

    # Make another request to test rate limiting
    with pytest.raises(HTTPException) as exc_info:
        await parallel_request_handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={"model": "azure-model"},
            call_type="",
        )

    # Wait for window to expire
    await asyncio.sleep(3)

    # Make request after window expiry
    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data={"model": "azure-model"},
        call_type="",
    )

    # Verify new window and reset counter
    final_counter_value = await local_cache.async_get_cache(key=counter_key)

    assert final_counter_value == 1, "Counter should reset to 1 after window expiry"
