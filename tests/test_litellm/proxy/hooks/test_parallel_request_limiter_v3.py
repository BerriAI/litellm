"""
Unit Tests for the max parallel request limiter v3 for the proxy
"""
import asyncio
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

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
from litellm.types.utils import ModelResponse, Usage


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
    window_starts: Dict[str, int] = {}

    async def mock_batch_rate_limiter(*args, **kwargs):
        keys = kwargs.get("keys") if kwargs else args[0]
        args_list = kwargs.get("args") if kwargs else args[1]
        now = args_list[0]
        window_size = args_list[1]
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
    window_starts: Dict[str, int] = {}

    async def mock_batch_rate_limiter(*args, **kwargs):
        keys = kwargs.get("keys") if kwargs else args[0]
        args_list = kwargs.get("args") if kwargs else args[1]
        now = args_list[0]
        window_size = args_list[1]
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
    window_starts: Dict[str, int] = {}

    async def mock_batch_rate_limiter(*args, **kwargs):
        print(f"args: {args}, kwargs: {kwargs}")
        keys = kwargs.get("keys") if kwargs else args[0]
        args_list = kwargs.get("args") if kwargs else args[1]
        now = args_list[0]
        window_size = args_list[1]
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


@pytest.mark.parametrize(
    "token_rate_limit_type",
    ["input", "output", "total"],
)
@pytest.mark.asyncio
async def test_token_rate_limit_type_respected_v3(monkeypatch, token_rate_limit_type):
    """
    Test that the token_rate_limit_type setting is respected when incrementing usage
    """
    # Set up environment and mock general_settings
    monkeypatch.setenv("LITELLM_RATE_LIMIT_WINDOW_SIZE", "60")

    _api_key = "sk-12345"
    _api_key = hash_token(_api_key)
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key, tpm_limit=100)
    local_cache = DualCache()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )

    # Mock the get_rate_limit_type method directly since it imports general_settings internally
    def mock_get_rate_limit_type():
        return token_rate_limit_type

    monkeypatch.setattr(
        parallel_request_handler, "get_rate_limit_type", mock_get_rate_limit_type
    )

    # Create a mock response with different token counts
    mock_usage = Usage(prompt_tokens=20, completion_tokens=30, total_tokens=50)
    mock_response = ModelResponse(
        id="mock-response",
        object="chat.completion",
        created=int(datetime.now().timestamp()),
        model="gpt-3.5-turbo",
        usage=mock_usage,
        choices=[],
    )

    # Create mock kwargs for the success event
    mock_kwargs = {
        "litellm_params": {
            "metadata": {
                "user_api_key": _api_key,
                "user_api_key_user_id": None,
                "user_api_key_team_id": None,
                "user_api_key_end_user_id": None,
            }
        },
        "model": "gpt-3.5-turbo",
    }

    # Mock the pipeline increment method to capture the operations
    captured_operations = []

    async def mock_increment_pipeline(increment_list, **kwargs):
        captured_operations.extend(increment_list)
        return True

    monkeypatch.setattr(
        parallel_request_handler.internal_usage_cache.dual_cache,
        "async_increment_cache_pipeline",
        mock_increment_pipeline,
    )

    # Call the success event handler
    await parallel_request_handler.async_log_success_event(
        kwargs=mock_kwargs,
        response_obj=mock_response,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    # Verify that the correct token count was used based on the rate limit type
    assert (
        len(captured_operations) == 2
    ), "Should have 2 operations: max_parallel_requests decrement and TPM increment"

    # Find the TPM increment operation (not the max_parallel_requests decrement)
    tpm_operation = None
    for op in captured_operations:
        if op["key"].endswith(":tokens"):
            tpm_operation = op
            break

    assert tpm_operation is not None, "Should have a TPM increment operation"

    # Check that the correct token count was used
    expected_tokens = {
        "input": mock_usage.prompt_tokens,  # 20
        "output": mock_usage.completion_tokens,  # 50 (Note: implementation uses total_tokens for output, which might be a bug)
        "total": mock_usage.total_tokens,  # 50
    }

    assert (
        tpm_operation["increment_value"] == expected_tokens[token_rate_limit_type]
    ), f"Expected {expected_tokens[token_rate_limit_type]} tokens for type '{token_rate_limit_type}', got {tpm_operation['increment_value']}"


@pytest.mark.asyncio
async def test_async_log_failure_event_v3():
    """
    Simple test for async_log_failure_event - should decrement max_parallel_requests by 1
    """
    _api_key = "sk-12345"
    _api_key = hash_token(_api_key)
    local_cache = DualCache()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )

    # Mock kwargs with user_api_key
    mock_kwargs = {"litellm_params": {"metadata": {"user_api_key": _api_key}}}

    # Capture pipeline operations
    captured_ops = []

    async def mock_pipeline(increment_list, **kwargs):
        captured_ops.extend(increment_list)

    parallel_request_handler.internal_usage_cache.dual_cache.async_increment_cache_pipeline = (
        mock_pipeline
    )

    # Call async_log_failure_event
    await parallel_request_handler.async_log_failure_event(
        kwargs=mock_kwargs, response_obj=None, start_time=None, end_time=None
    )

    # Verify correct operation was created
    assert len(captured_ops) == 1
    op = captured_ops[0]
    assert op["key"] == f"{{api_key:{_api_key}}}:max_parallel_requests"
    assert op["increment_value"] == -1
    assert op["ttl"] == 60  # default window size
