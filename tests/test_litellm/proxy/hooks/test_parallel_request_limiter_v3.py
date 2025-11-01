"""
Unit Tests for the max parallel request limiter v3 for the proxy
"""

import asyncio
import os
import sys
import time
from datetime import datetime, timedelta
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

class TimeController:
    def __init__(self):
        self._current = datetime.utcnow()

    def now(self) -> datetime:
        return self._current

    def advance(self, seconds: float) -> None:
        self._current += timedelta(seconds=seconds)


@pytest.fixture
def time_controller(monkeypatch):
    controller = TimeController()
    monkeypatch.setattr(time, "time", lambda: controller.now().timestamp())
    return controller


@pytest.mark.flaky(reruns=3)
@pytest.mark.asyncio
async def test_sliding_window_rate_limit_v3(monkeypatch, time_controller):
    """
    Test the sliding window rate limiting functionality
    """
    monkeypatch.setenv("LITELLM_RATE_LIMIT_WINDOW_SIZE", "2")
    _api_key = "sk-12345"
    _api_key = hash_token(_api_key)
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key, rpm_limit=3)
    local_cache = DualCache()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache),
        time_provider=time_controller.now,
    )

    # Mock the batch_rate_limiter_script to simulate window expiry and use correct key construction
    window_starts: Dict[str, int] = {}
    request_counts: Dict[str, int] = {}

    async def mock_batch_rate_limiter(*args, **kwargs):
        keys = kwargs.get("keys") if kwargs else args[0]
        args_list = kwargs.get("args") if kwargs else args[1]
        now = args_list[0]
        window_size = args_list[1]
        results = []
        for i in range(0, len(keys), 2):  # Fixed: should be 2, not 3
            window_key = keys[i]
            counter_key = keys[i + 1]
            # Simulate window expiry
            prev_window = window_starts.get(window_key)
            prev_counter = request_counts.get(counter_key, 0)
            if prev_window is None or (now - prev_window) >= window_size:
                # Window expired, reset
                window_starts[window_key] = now
                new_counter = 1
                request_counts[counter_key] = new_counter
                await local_cache.async_set_cache(
                    key=window_key, value=now, ttl=window_size
                )
                await local_cache.async_set_cache(
                    key=counter_key, value=new_counter, ttl=window_size
                )
            else:
                new_counter = prev_counter + 1
                request_counts[counter_key] = new_counter
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

    # Third request should succeed (counter is 3, limit is 3, so 3 <= 3)
    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )

    # Fourth request should fail (counter would be 4, limit is 3, so 4 > 3)
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
    time_controller.advance(3)

    print("WAITED 3 seconds")

    print(f"local_cache: {local_cache.in_memory_cache.cache_dict}")

    # After window expires, should be able to make requests again
    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )


@pytest.mark.asyncio
async def test_rate_limiter_script_return_values_v3(monkeypatch, time_controller):
    """
    Test that the rate limiter script returns both counter and window values correctly
    """
    monkeypatch.setenv("LITELLM_RATE_LIMIT_WINDOW_SIZE", "2")
    _api_key = "sk-12345"
    _api_key = hash_token(_api_key)
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key, rpm_limit=3)
    local_cache = DualCache()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache),
        time_provider=time_controller.now,
    )

    # Mock the batch_rate_limiter_script to simulate window expiry and use correct key construction
    window_starts: Dict[str, int] = {}
    request_counts: Dict[str, int] = {}

    async def mock_batch_rate_limiter(*args, **kwargs):
        keys = kwargs.get("keys") if kwargs else args[0]
        args_list = kwargs.get("args") if kwargs else args[1]
        now = args_list[0]
        window_size = args_list[1]
        results = []
        for i in range(0, len(keys), 2):  # Fixed: should be 2, not 3
            window_key = keys[i]
            counter_key = keys[i + 1]
            # Simulate window expiry
            prev_window = window_starts.get(window_key)
            prev_counter = request_counts.get(counter_key, 0)
            if prev_window is None or (now - prev_window) >= window_size:
                # Window expired, reset
                window_starts[window_key] = now
                new_counter = 1
                request_counts[counter_key] = new_counter
                await local_cache.async_set_cache(
                    key=window_key, value=now, ttl=window_size
                )
                await local_cache.async_set_cache(
                    key=counter_key, value=new_counter, ttl=window_size
                )
            else:
                new_counter = prev_counter + 1
                request_counts[counter_key] = new_counter
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
    time_controller.advance(3)

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
async def test_normal_router_call_tpm_v3(monkeypatch, rate_limit_object, time_controller):
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
        internal_usage_cache=InternalUsageCache(local_cache),
        time_provider=time_controller.now,
    )

    # Mock the batch_rate_limiter_script to simulate window expiry and use correct key construction
    window_starts: Dict[str, int] = {}
    request_counts: Dict[str, int] = {}

    async def mock_batch_rate_limiter(*args, **kwargs):
        print(f"args: {args}, kwargs: {kwargs}")
        keys = kwargs.get("keys") if kwargs else args[0]
        args_list = kwargs.get("args") if kwargs else args[1]
        now = args_list[0]
        window_size = args_list[1]
        results = []
        for i in range(0, len(keys), 2):  # Fixed: should be 2, not 3
            window_key = keys[i]
            counter_key = keys[i + 1]
            # Simulate window expiry
            prev_window = window_starts.get(window_key)
            prev_counter = request_counts.get(counter_key, 0)
            if prev_window is None or (now - prev_window) >= window_size:
                # Window expired, reset
                window_starts[window_key] = now
                new_counter = 1
                request_counts[counter_key] = new_counter
                await local_cache.async_set_cache(
                    key=window_key, value=now, ttl=window_size
                )
                await local_cache.async_set_cache(
                    key=counter_key, value=new_counter, ttl=window_size
                )
            else:
                new_counter = prev_counter + 1
                request_counts[counter_key] = new_counter
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
    await asyncio.sleep(0)
    time_controller.advance(1)

    # Verify the token count is tracked
    counter_value = await local_cache.async_get_cache(key=counter_key)
    print(f"local_cache: {local_cache.in_memory_cache.cache_dict}")

    assert (
        counter_value is not None
    ), f"Counter value should be stored in cache for {counter_key}"

    # Manually increment the token counter to simulate token usage from previous call
    # This simulates what would happen after a successful call
    await local_cache.async_increment_cache(key=counter_key, value=15, ttl=2)  # Use up most of our 10 token limit
    
    # Make another request to test rate limiting - this should fail as we've consumed tokens
    with pytest.raises(HTTPException) as exc_info:
        await parallel_request_handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={"model": "azure-model"},
            call_type="",
        )

    # Wait for window to expire
    time_controller.advance(3)

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


@pytest.mark.asyncio
async def test_should_rate_limit_only_called_when_limits_exist_v3():
    """
    Test that should_rate_limit is only called when actual rate limits are configured.
    This verifies the optimization that avoids unnecessary rate limit checks.
    """
    _api_key = "sk-12345"
    _api_key = hash_token(_api_key)
    local_cache = DualCache()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )

    # Mock should_rate_limit to track if it's called
    should_rate_limit_called = False

    async def mock_should_rate_limit(*args, **kwargs):
        nonlocal should_rate_limit_called
        should_rate_limit_called = True
        return {"overall_code": "OK", "statuses": []}

    parallel_request_handler.should_rate_limit = mock_should_rate_limit

    # Test 1: No rate limits configured - should_rate_limit should NOT be called
    should_rate_limit_called = False
    user_api_key_dict_no_limits = UserAPIKeyAuth(
        api_key=_api_key,
        user_id="test_user",
        team_id="test_team",
        end_user_id="test_end_user",
        # No rpm_limit, tpm_limit, max_parallel_requests, etc.
    )

    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict_no_limits,
        cache=local_cache,
        data={"model": "gpt-3.5-turbo"},
        call_type="",
    )

    assert (
        not should_rate_limit_called
    ), "should_rate_limit should not be called when no rate limits are configured"

    # Test 2: API key rate limits configured - should_rate_limit SHOULD be called
    should_rate_limit_called = False
    user_api_key_dict_with_api_limits = UserAPIKeyAuth(
        api_key=_api_key,
        rpm_limit=100,  # Rate limit configured
    )

    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict_with_api_limits,
        cache=local_cache,
        data={"model": "gpt-3.5-turbo"},
        call_type="",
    )

    assert (
        should_rate_limit_called
    ), "should_rate_limit should be called when API key rate limits are configured"

    # Test 3: User rate limits configured - should_rate_limit SHOULD be called
    should_rate_limit_called = False
    user_api_key_dict_with_user_limits = UserAPIKeyAuth(
        api_key=_api_key,
        user_id="test_user",
        user_tpm_limit=1000,  # User rate limit configured
    )

    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict_with_user_limits,
        cache=local_cache,
        data={"model": "gpt-3.5-turbo"},
        call_type="",
    )

    assert (
        should_rate_limit_called
    ), "should_rate_limit should be called when user rate limits are configured"

    # Test 4: Team rate limits configured - should_rate_limit SHOULD be called
    should_rate_limit_called = False
    user_api_key_dict_with_team_limits = UserAPIKeyAuth(
        api_key=_api_key,
        team_id="test_team",
        team_rpm_limit=500,  # Team rate limit configured
    )

    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict_with_team_limits,
        cache=local_cache,
        data={"model": "gpt-3.5-turbo"},
        call_type="",
    )

    assert (
        should_rate_limit_called
    ), "should_rate_limit should be called when team rate limits are configured"

    # Test 5: End user rate limits configured - should_rate_limit SHOULD be called
    should_rate_limit_called = False
    user_api_key_dict_with_end_user_limits = UserAPIKeyAuth(
        api_key=_api_key,
        end_user_id="test_end_user",
        end_user_rpm_limit=200,  # End user rate limit configured
    )

    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict_with_end_user_limits,
        cache=local_cache,
        data={"model": "gpt-3.5-turbo"},
        call_type="",
    )

    assert (
        should_rate_limit_called
    ), "should_rate_limit should be called when end user rate limits are configured"

    # Test 6: Max parallel requests configured - should_rate_limit SHOULD be called
    should_rate_limit_called = False
    user_api_key_dict_with_parallel_limits = UserAPIKeyAuth(
        api_key=_api_key,
        max_parallel_requests=5,  # Max parallel requests configured
    )

    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict_with_parallel_limits,
        cache=local_cache,
        data={"model": "gpt-3.5-turbo"},
        call_type="",
    )

    assert (
        should_rate_limit_called
    ), "should_rate_limit should be called when max parallel requests are configured"


@pytest.mark.asyncio
async def test_model_specific_rate_limits_only_called_when_configured_v3():
    """
    Test that model-specific rate limits only trigger should_rate_limit when actually configured for the requested model.
    """
    from litellm.proxy.auth.auth_utils import (
        get_key_model_rpm_limit,
        get_key_model_tpm_limit,
    )

    _api_key = "sk-12345"
    _api_key = hash_token(_api_key)
    local_cache = DualCache()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )

    # Mock should_rate_limit to track if it's called
    should_rate_limit_called = False

    async def mock_should_rate_limit(*args, **kwargs):
        nonlocal should_rate_limit_called
        should_rate_limit_called = True
        return {"overall_code": "OK", "statuses": []}

    parallel_request_handler.should_rate_limit = mock_should_rate_limit

    # Test 1: Model-specific rate limits configured but for different model - should NOT be called
    should_rate_limit_called = False
    user_api_key_dict_with_model_limits = UserAPIKeyAuth(
        api_key=_api_key,
        metadata={
            "model_tpm_limit": {"gpt-4": 1000}
        },  # Rate limit for gpt-4, not gpt-3.5-turbo
    )

    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict_with_model_limits,
        cache=local_cache,
        data={"model": "gpt-3.5-turbo"},  # Requesting different model
        call_type="",
    )

    assert (
        not should_rate_limit_called
    ), "should_rate_limit should not be called when model-specific limits don't match requested model"

    # Test 2: Model-specific rate limits configured for requested model - SHOULD be called
    should_rate_limit_called = False
    user_api_key_dict_with_matching_model_limits = UserAPIKeyAuth(
        api_key=_api_key,
        metadata={
            "model_tpm_limit": {"gpt-3.5-turbo": 1000}
        },  # Rate limit for requested model
    )

    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict_with_matching_model_limits,
        cache=local_cache,
        data={"model": "gpt-3.5-turbo"},  # Requesting same model
        call_type="",
    )

    assert (
        should_rate_limit_called
    ), "should_rate_limit should be called when model-specific limits match requested model"


@pytest.mark.asyncio
async def test_tpm_api_key_rate_limits_v3():

    _api_key = "sk-12345"
    _api_key_hash = hash_token(_api_key)
    model = "gpt-3.5-turbo"
    rpm_limit = 2
    tpm_limit = 2

    rpms = {model: rpm_limit}
    tpms = {model: tpm_limit}

    user_api_key_dict = UserAPIKeyAuth(
        api_key=_api_key_hash,
        key_alias=_api_key,
        rpm_limit_per_model=rpms,
        tpm_limit_per_model=tpms,
        models=[],
    )
    
    user_api_key_dict.metadata["model_tpm_limit"] = tpms
    user_api_key_dict.metadata["model_rpm_limit"] = rpms

    local_cache = DualCache()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )

    # Mock should_rate_limit to capture the descriptors
    captured_descriptors = None
    original_should_rate_limit = parallel_request_handler.should_rate_limit

    async def mock_should_rate_limit(descriptors, **kwargs):
        nonlocal captured_descriptors
        captured_descriptors = descriptors
        # Return Error response to ensure HTTPException
        return {
            "overall_code": "OVER_LIMIT",
            "statuses": [{'code': 'OK', 'current_limit': 2, 'limit_remaining': 1, 'rate_limit_type': 'requests', 'descriptor_key': 'model_per_key'},
                         {'code': 'OVER_LIMIT', 'current_limit': 2, 'limit_remaining': -18, 'rate_limit_type': 'tokens', 'descriptor_key': 'model_per_key'}]
        }
        
    parallel_request_handler.should_rate_limit = mock_should_rate_limit
    
    # Test the pre-call hook
    error = None
    try:
       await parallel_request_handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={"model": model},
            call_type="",
        )
    except HTTPException as e:
        error=e
        assert e.status_code == 429
        assert "rate_limit_type" in e.headers
        assert e.headers.get("rate_limit_type") == "tokens"
        assert "retry-after" in e.headers
        
    
    assert error is not None, "An Exception must be thrown"
    assert captured_descriptors is not None, "Rate limit descriptors should be captured"
    
    model_per_key_descriptor = None
    for descriptor in captured_descriptors:
        if descriptor["key"] == "model_per_key":
            model_per_key_descriptor = descriptor
            break

    assert model_per_key_descriptor is not None, "Api-Key descriptor should be present"
    assert model_per_key_descriptor["value"] == f"{_api_key_hash}:{model}", "Api-Key value should combine api_key and model"
    assert model_per_key_descriptor["rate_limit"]["requests_per_unit"] == rpm_limit, "Api-Key RPM limit should be set"
    assert model_per_key_descriptor["rate_limit"]["tokens_per_unit"] == tpm_limit, "Api-Key TPM limit should be set"


@pytest.mark.asyncio
async def test_rpm_api_key_rate_limits_v3():

    _api_key = "sk-12345"
    _api_key_hash = hash_token(_api_key)
    model = "gpt-3.5-turbo"
    rpm_limit = 2
    tpm_limit = 2

    rpms = {model: rpm_limit}
    tpms = {model: tpm_limit}

    user_api_key_dict = UserAPIKeyAuth(
        api_key=_api_key_hash,
        key_alias=_api_key,
        rpm_limit_per_model=rpms,
        tpm_limit_per_model=tpms,
        models=[],
    )
    
    user_api_key_dict.metadata["model_tpm_limit"] = tpms
    user_api_key_dict.metadata["model_rpm_limit"] = rpms

    local_cache = DualCache()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )

    # Mock should_rate_limit to capture the descriptors
    captured_descriptors = None
    original_should_rate_limit = parallel_request_handler.should_rate_limit

    async def mock_should_rate_limit(descriptors, **kwargs):
        nonlocal captured_descriptors
        captured_descriptors = descriptors
        # Return Error response to ensure HTTPException
        return {
            "overall_code": "OVER_LIMIT",
            "statuses": [{'code': 'OVER_LIMIT', 'current_limit': 2, 'limit_remaining': -2, 'rate_limit_type': 'requests', 'descriptor_key': 'model_per_key'},
                         {'code': 'OK', 'current_limit': 2, 'limit_remaining': 2, 'rate_limit_type': 'tokens', 'descriptor_key': 'model_per_key'}]
        }
        
    parallel_request_handler.should_rate_limit = mock_should_rate_limit
    
    # Test the pre-call hook
    error = None
    try:
       await parallel_request_handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={"model": model},
            call_type="",
        )
    except HTTPException as e:
        error=e
        assert e.status_code == 429
        assert "rate_limit_type" in e.headers
        assert e.headers.get("rate_limit_type") == "requests"
        assert "retry-after" in e.headers
    
    assert error is not None, "An Exception must be thrown"
    assert captured_descriptors is not None, "Rate limit descriptors should be captured"
    
    model_per_key_descriptor = None
    for descriptor in captured_descriptors:
        if descriptor["key"] == "model_per_key":
            model_per_key_descriptor = descriptor
            break

    assert model_per_key_descriptor is not None, "Api-Key descriptor should be present"
    assert model_per_key_descriptor["value"] == f"{_api_key_hash}:{model}", "Api-Key value should combine api_key and model"
    assert model_per_key_descriptor["rate_limit"]["requests_per_unit"] == rpm_limit, "Api-Key RPM limit should be set"
    assert model_per_key_descriptor["rate_limit"]["tokens_per_unit"] == tpm_limit, "Api-Key TPM limit should be set"

@pytest.mark.asyncio
async def test_team_member_rate_limits_v3():
    """
    Test that team member RPM/TPM rate limits are properly applied for team member combinations.
    """
    _api_key = "sk-12345"
    _api_key = hash_token(_api_key)
    _team_id = "team_123"
    _user_id = "user_456"
    
    user_api_key_dict = UserAPIKeyAuth(
        api_key=_api_key,
        team_id=_team_id,
        user_id=_user_id,
        team_member_rpm_limit=10,
        team_member_tpm_limit=1000,
    )
    
    local_cache = DualCache()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )

    # Mock should_rate_limit to capture the descriptors
    captured_descriptors = None
    original_should_rate_limit = parallel_request_handler.should_rate_limit

    async def mock_should_rate_limit(descriptors, **kwargs):
        nonlocal captured_descriptors
        captured_descriptors = descriptors
        # Return OK response to avoid HTTPException
        return {
            "overall_code": "OK",
            "statuses": []
        }

    parallel_request_handler.should_rate_limit = mock_should_rate_limit

    # Test the pre-call hook
    
    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data={"model": "gpt-3.5-turbo"},
        call_type="",
    )

    # Verify team member descriptor was created
    assert captured_descriptors is not None, "Rate limit descriptors should be captured"
    
    team_member_descriptor = None
    for descriptor in captured_descriptors:
        if descriptor["key"] == "team_member":
            team_member_descriptor = descriptor
            break
    
    assert team_member_descriptor is not None, "Team member descriptor should be present"
    assert team_member_descriptor["value"] == f"{_team_id}:{_user_id}", "Team member value should combine team_id and user_id"
    assert team_member_descriptor["rate_limit"]["requests_per_unit"] == 10, "Team member RPM limit should be set"
    assert team_member_descriptor["rate_limit"]["tokens_per_unit"] == 1000, "Team member TPM limit should be set"


@pytest.mark.asyncio
async def test_dynamic_rate_limiting_v3():
    """
    Test that dynamic rate limiting only enforces limits when model has failures.
    
    When rpm_limit_type is set to "dynamic":
    - If model has no failures, rate limits should NOT be enforced (allow exceeding)
    - If model has failures above threshold, rate limits SHOULD be enforced
    """
    _api_key = "sk-12345"
    _api_key_hash = hash_token(_api_key)
    model = "gpt-3.5-turbo"
    
    # Set a low RPM limit to make testing easier
    user_api_key_dict = UserAPIKeyAuth(
        api_key=_api_key_hash,
        rpm_limit=2,
        metadata={"rpm_limit_type": "dynamic"},
    )
    
    local_cache = DualCache()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )
    
    # Mock should_rate_limit to track if limits are enforced
    captured_descriptors = []
    
    async def mock_should_rate_limit(descriptors, **kwargs):
        captured_descriptors.clear()
        captured_descriptors.extend(descriptors)
        return {"overall_code": "OK", "statuses": []}
    
    parallel_request_handler.should_rate_limit = mock_should_rate_limit
    
    # Test 1: No failures - rate limits should NOT be enforced (rpm_limit should be None)
    async def mock_check_no_failures(*args, **kwargs):
        return False
    
    parallel_request_handler._check_model_has_recent_failures = mock_check_no_failures
    
    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data={"model": model},
        call_type="",
    )
    
    # Find the API key descriptor
    api_key_descriptor = None
    for descriptor in captured_descriptors:
        if descriptor["key"] == "api_key":
            api_key_descriptor = descriptor
            break
    
    assert api_key_descriptor is not None, "API key descriptor should be present"
    assert (
        api_key_descriptor["rate_limit"]["requests_per_unit"] is None
    ), "RPM limit should be None when dynamic mode and no failures"
    
    # Test 2: With failures - rate limits SHOULD be enforced (rpm_limit should be set)
    async def mock_check_with_failures(*args, **kwargs):
        return True
    
    parallel_request_handler._check_model_has_recent_failures = mock_check_with_failures
    captured_descriptors.clear()
    
    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data={"model": model},
        call_type="",
    )
    
    # Find the API key descriptor again
    api_key_descriptor = None
    for descriptor in captured_descriptors:
        if descriptor["key"] == "api_key":
            api_key_descriptor = descriptor
            break
    
    assert api_key_descriptor is not None, "API key descriptor should be present"
    assert (
        api_key_descriptor["rate_limit"]["requests_per_unit"] == 2
    ), "RPM limit should be enforced when dynamic mode and failures detected"


@pytest.mark.asyncio
async def test_async_increment_tokens_with_ttl_preservation():
    """
    Test TTL preservation functionality for token increment operations.
    
    This test verifies that:
    1. Keys are created with proper TTL on first increment
    2. TTL is preserved on subsequent increments (not reset)
    3. Both TTL and non-TTL operations work correctly in the same call
    
    Environment variables required:
    - REDIS_HOST: Redis server hostname
    - REDIS_PORT: Redis server port
    - REDIS_PASSWORD: Redis password (optional)
    
    Test scenario:
    1. First call: Create keys with TTL=60s and TTL=None
    2. Wait 2 seconds
    3. Second call: Increment same keys
    4. Verify TTL decreased but wasn't reset to 60s
    """
    import os
    import time

    from litellm.caching.redis_cache import RedisCache
    from litellm.types.caching import RedisPipelineIncrementOperation

    # Skip test if Redis environment variables are not set
    redis_host = os.getenv("REDIS_HOST")
    redis_port = os.getenv("REDIS_PORT") 
    redis_password = os.getenv("REDIS_PASSWORD")
    
    if not redis_host or not redis_port:
        pytest.skip("Redis environment variables (REDIS_HOST, REDIS_PORT) not set")
    
    # Setup Redis cache
    redis_cache = RedisCache(
        host=redis_host,
        port=int(redis_port),
        password=redis_password,
    )
    
    local_cache = DualCache(redis_cache=redis_cache)
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )
    
    # Verify Redis connection is working
    try:
        await redis_cache.ping()
    except Exception as e:
        pytest.skip(f"Redis connection failed: {str(e)}")
    
    # Verify the TTL preservation script is registered
    if parallel_request_handler.token_increment_script is None:
        pytest.skip("Token increment script not available - Redis Lua scripting may not be supported")
    
    # Test keys - use hash tags to ensure they map to same Redis cluster slot
    test_key_with_ttl = "{test_ttl}:with_ttl"
    test_key_without_ttl = "{test_ttl}:without_ttl"
    
    try:
        # Clean up any existing test keys
        try:
            await redis_cache.async_delete_cache(test_key_with_ttl)
            await redis_cache.async_delete_cache(test_key_without_ttl)
        except Exception:
            # Keys might not exist, ignore cleanup errors
            pass
        
        # First increment: Create operations with mixed TTL scenarios
        pipeline_operations_first = [
            RedisPipelineIncrementOperation(
                key=test_key_with_ttl,
                increment_value=10.0,
                ttl=60
            ),
            RedisPipelineIncrementOperation(
                key=test_key_without_ttl,
                increment_value=5.0,
                ttl=None  # No TTL
            )
        ]
        
        # Execute first increment
        await parallel_request_handler.async_increment_tokens_with_ttl_preservation(
            pipeline_operations=pipeline_operations_first
        )
        
        # Small delay to ensure Redis has processed the commands
        await asyncio.sleep(0.1)
        
        # Verify keys exist and check initial TTL
        ttl_after_first = await redis_cache.async_get_ttl(test_key_with_ttl)
        value_after_first_with_ttl = await redis_cache.async_get_cache(test_key_with_ttl)
        value_after_first_without_ttl = await redis_cache.async_get_cache(test_key_without_ttl)
        
        assert value_after_first_with_ttl == 10.0, f"First increment should set value to 10.0, got {value_after_first_with_ttl}"
        assert value_after_first_without_ttl == 5.0, "First increment should set value to 5.0"
        assert ttl_after_first is not None and ttl_after_first > 0, "Key with TTL should have positive TTL after first increment"
        assert ttl_after_first <= 60, "TTL should not exceed the set value"
        
        # Check TTL for key without TTL (should be None, meaning no expiry)
        ttl_no_ttl_key = await redis_cache.async_get_ttl(test_key_without_ttl)
        assert ttl_no_ttl_key is None, "Key without TTL should have no expiry (None from async_get_ttl)"
        
        # Wait a moment to ensure TTL decreases
        await asyncio.sleep(2)
        
        # Second increment: Same operations to test TTL preservation
        pipeline_operations_second = [
            RedisPipelineIncrementOperation(
                key=test_key_with_ttl,
                increment_value=15.0,
                ttl=60  # Same TTL value
            ),
            RedisPipelineIncrementOperation(
                key=test_key_without_ttl,
                increment_value=7.0,
                ttl=None  # No TTL
            )
        ]
        
        # Execute second increment
        await parallel_request_handler.async_increment_tokens_with_ttl_preservation(
            pipeline_operations=pipeline_operations_second
        )
        
        # Small delay to ensure Redis has processed the commands
        await asyncio.sleep(0.1)
        
        # Verify TTL preservation and value updates
        ttl_after_second = await redis_cache.async_get_ttl(test_key_with_ttl)
        value_after_second_with_ttl = await redis_cache.async_get_cache(test_key_with_ttl)
        value_after_second_without_ttl = await redis_cache.async_get_cache(test_key_without_ttl)
        
        assert value_after_second_with_ttl == 25.0, "Second increment should update value to 25.0"
        assert value_after_second_without_ttl == 12.0, "Second increment should update value to 12.0"
        
        # Critical test: TTL should be preserved (not reset to 60)
        assert ttl_after_second is not None, "TTL should still exist"
        assert ttl_after_second < ttl_after_first, "TTL should have decreased (not been reset)"
        assert ttl_after_second > 0, "TTL should still be positive"
        
        # TTL should not be close to the original 60 seconds (proving it wasn't reset)
        assert ttl_after_second < 59, "TTL should be significantly less than original, proving preservation"
        
        # Key without TTL should still have no expiry
        ttl_no_ttl_key_after_second = await redis_cache.async_get_ttl(test_key_without_ttl)
        assert ttl_no_ttl_key_after_second is None, "Key without TTL should still have no expiry"
        
    finally:
        # Clean up test keys
        try:
            await redis_cache.async_delete_cache(test_key_with_ttl)
            await redis_cache.async_delete_cache(test_key_without_ttl)
        except Exception:
            # Ignore cleanup errors
            pass
        
        # Properly close Redis connections to prevent warnings
        try:
            await redis_cache.disconnect()
        except Exception:
            # Ignore disconnect errors
            pass


@pytest.mark.asyncio
async def test_async_increment_tokens_fallback_behavior():
    """
    Test fallback behavior when Lua script is not available.
    """
    from litellm.types.caching import RedisPipelineIncrementOperation
    
    local_cache = DualCache()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )
    
    # Mock the token_increment_script to None to simulate unavailable script
    parallel_request_handler.token_increment_script = None
    
    # Mock the fallback method
    fallback_called = False
    original_method = parallel_request_handler.internal_usage_cache.dual_cache.async_increment_cache_pipeline
    
    async def mock_fallback(*args, **kwargs):
        nonlocal fallback_called
        fallback_called = True
        return await original_method(*args, **kwargs)
    
    parallel_request_handler.internal_usage_cache.dual_cache.async_increment_cache_pipeline = mock_fallback
    
    # Test operations
    pipeline_operations = [
        RedisPipelineIncrementOperation(
            key="test_fallback_key",
            increment_value=10.0,
            ttl=60
        )
    ]
    
    # Execute increment
    await parallel_request_handler.async_increment_tokens_with_ttl_preservation(
        pipeline_operations=pipeline_operations
    )
    
    # Verify fallback was called
    assert fallback_called, "Fallback method should be called when Lua script is not available"


# Redis Cluster Compatibility Tests
def test_group_keys_by_hash_tag_regular_redis():
    """
    Test that keys are correctly grouped for regular Redis (non-cluster).
    
    For regular Redis, all keys should be grouped together under a single group.
    """
    local_cache = DualCache()
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )
    
    # Test keys with different hash tags
    test_keys = [
        "{api_key:sk-123}:window",
        "{api_key:sk-123}:requests", 
        "{api_key:sk-123}:tokens",
        "{user:user-456}:window",
        "{user:user-456}:requests",
        "{team:team-789}:window",
        "{team:team-789}:tokens",
        "no_hash_tag_key"
    ]
    
    # Group the keys (should be single group for regular Redis)
    groups = handler._group_keys_by_hash_tag(test_keys)
    
    # Verify all keys are in single group for regular Redis
    assert len(groups) == 1, f"Expected 1 group for regular Redis, got {len(groups)}"
    assert "all_keys" in groups, "Expected 'all_keys' group for regular Redis"
    assert set(groups["all_keys"]) == set(test_keys), "All keys should be in single group"


def test_group_keys_by_hash_tag_redis_cluster():
    """
    Test that keys are correctly grouped by Redis cluster slots when using Redis cluster.
    
    This ensures that keys are grouped by their slot number for cluster compatibility.
    """
    from unittest.mock import patch
    
    local_cache = DualCache()
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )
    
    # Mock _is_redis_cluster to return True
    with patch.object(handler, '_is_redis_cluster', return_value=True):
        # Test keys with different hash tags
        test_keys = [
            "{api_key:sk-123}:window",
            "{api_key:sk-123}:requests", 
            "{user:user-456}:window",
            "{user:user-456}:requests",
        ]
        
        # Group the keys (should be grouped by slot for Redis cluster)
        groups = handler._group_keys_by_hash_tag(test_keys)
        
        # Verify keys are grouped by slot
        assert len(groups) >= 1, "Should have at least 1 slot group"
        
        # All group keys should start with "slot_"
        for group_key in groups.keys():
            assert group_key.startswith("slot_"), f"Group key {group_key} should start with 'slot_'"
        
        # Verify all original keys are present across groups
        all_grouped_keys = []
        for group_keys in groups.values():
            all_grouped_keys.extend(group_keys)
        assert set(all_grouped_keys) == set(test_keys), "All keys should be present in groups"


def test_keyslot_for_redis_cluster():
    """
    Test the keyslot calculation for Redis cluster.
    """
    local_cache = DualCache()
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )
    
    # Test basic key
    slot1 = handler.keyslot_for_redis_cluster("user:1000")
    assert 0 <= slot1 < 16384, "Slot should be in valid range"
    
    # Test key with hash tag
    slot2 = handler.keyslot_for_redis_cluster("foo{bar}baz")
    slot3 = handler.keyslot_for_redis_cluster("{bar}")
    assert slot2 == slot3, "Keys with same hash tag should have same slot"
    
    # Test keys with same hash tag should have same slot
    slot4 = handler.keyslot_for_redis_cluster("{api_key:sk-123}:requests")
    slot5 = handler.keyslot_for_redis_cluster("{api_key:sk-123}:window")
    assert slot4 == slot5, "Keys with same hash tag should have same slot"


@pytest.mark.asyncio
async def test_execute_redis_batch_rate_limiter_script_cluster_compatibility():
    """
    Test that the Redis batch rate limiter script execution handles cluster compatibility
    by grouping keys and falling back gracefully on errors.
    
    This simulates the Redis cluster error scenario and verifies fallback behavior.
    """
    from unittest.mock import AsyncMock, patch
    
    local_cache = DualCache()
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )
    
    # Mock _is_redis_cluster to return True for this test
    with patch.object(handler, '_is_redis_cluster', return_value=True):
        # Mock script that simulates Redis cluster slot conflict
        mock_script = AsyncMock()
        mock_script.side_effect = [
            Exception("EVALSHA - all keys must map to the same key slot"),  # First group fails
            [1234, 1, 1234, 2]  # Second group succeeds
        ]
        handler.batch_rate_limiter_script = mock_script
        
        # Mock in-memory fallback (returns 2 values for 2 keys: window_start, counter)
        handler.in_memory_cache_sliding_window = AsyncMock(return_value=[1234, 1])
        
        # Test keys from different hash tags (would fail in cluster without grouping)
        test_keys = [
            "{api_key:sk-123}:window",
            "{api_key:sk-123}:requests",
            "{user:user-456}:window", 
            "{user:user-456}:requests"
        ]
        
        # Execute the method
        results = await handler._execute_redis_batch_rate_limiter_script(
            keys_to_fetch=test_keys,
            now_int=1234
        )
        
        # Verify results: 2 from fallback + 4 from successful script = 6 total
        assert len(results) == 6, f"Expected 6 results, got {len(results)}"
        
        # Verify script was called twice (once per slot group)
        assert mock_script.call_count == 2
        
        # Verify fallback was called for the failed group
        handler.in_memory_cache_sliding_window.assert_called_once()
        
        # Verify the calls were made with grouped keys
        call_args_list = mock_script.call_args_list
        
        # Both calls should have keys, but we can't predict exact grouping without knowing slots
        # Just verify that keys were grouped and calls were made
        assert len(call_args_list) == 2, "Should have made 2 script calls"
        
        # Verify all keys were processed
        all_processed_keys = []
        for call_args in call_args_list:
            all_processed_keys.extend(call_args[1]['keys'])
        
        # Should have processed all keys (some might be duplicated due to fallback)
        unique_processed_keys = set(all_processed_keys)
        assert len(unique_processed_keys) >= 2, "Should have processed at least some keys"


@pytest.mark.asyncio
async def test_multiple_rate_limits_per_descriptor():
    """
    Test that the IndexError fix works correctly when a descriptor has multiple rate limit types.

    This specifically tests the scenario where:
    1. A descriptor has multiple rate limit types (requests, tokens, max_parallel_requests)
    2. Multiple statuses are generated for a single descriptor
    3. The old floor(i / 2) mapping would fail with IndexError
    4. The new descriptor_key-based lookup works correctly
    """
    _api_key = "sk-12345"
    _api_key_hash = hash_token(_api_key)

    # Create a user with multiple rate limit types to trigger multiple statuses per descriptor
    user_api_key_dict = UserAPIKeyAuth(
        api_key=_api_key_hash,
        rpm_limit=2,  # requests limit
        tpm_limit=10,  # tokens limit
        max_parallel_requests=1,  # parallel requests limit
    )

    local_cache = DualCache()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )

    # Mock should_rate_limit to return a response with multiple statuses where one hits the limit
    # This simulates the case where we have more statuses than descriptors due to multiple rate limit types
    async def mock_should_rate_limit(descriptors, **kwargs):
        # Verify we have one descriptor but will generate multiple statuses
        assert len(descriptors) == 1, "Should have exactly one api_key descriptor"
        assert descriptors[0]["key"] == "api_key", "Descriptor should be for api_key"

        # Return multiple statuses for the single descriptor (requests OK, tokens OK, parallel OVER_LIMIT)
        return {
            "overall_code": "OVER_LIMIT",
            "statuses": [
                {
                    "code": "OK",
                    "current_limit": 2,
                    "limit_remaining": 1,
                    "rate_limit_type": "requests",
                    "descriptor_key": "api_key"
                },
                {
                    "code": "OK",
                    "current_limit": 10,
                    "limit_remaining": 8,
                    "rate_limit_type": "tokens",
                    "descriptor_key": "api_key"
                },
                {
                    "code": "OVER_LIMIT",
                    "current_limit": 1,
                    "limit_remaining": -1,
                    "rate_limit_type": "max_parallel_requests",
                    "descriptor_key": "api_key"
                }
            ]
        }

    parallel_request_handler.should_rate_limit = mock_should_rate_limit

    # Test the pre-call hook - this should raise HTTPException but NOT IndexError
    with pytest.raises(HTTPException) as exc_info:
        await parallel_request_handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={"model": "gpt-3.5-turbo"},
            call_type="",
        )

    # Verify the exception details are correct and use the descriptor_key approach
    assert exc_info.value.status_code == 429
    assert "Rate limit exceeded for api_key:" in exc_info.value.detail
    assert "max_parallel_requests" in exc_info.value.detail
    assert "Current limit: 1" in exc_info.value.detail
    assert "Remaining: 0" in exc_info.value.detail  # max(0, -1) = 0

    # Verify headers are set correctly
    assert exc_info.value.headers.get("rate_limit_type") == "max_parallel_requests"
    assert "retry-after" in exc_info.value.headers
    assert "reset_at" in exc_info.value.headers


@pytest.mark.asyncio
async def test_missing_descriptor_fallback():
    """
    Test that the fallback works when a descriptor_key cannot be found in the descriptors list.

    This tests an edge case where somehow the descriptor_key in status doesn't match
    any descriptor key (shouldn't happen in normal operation but good for robustness).
    """
    _api_key = "sk-12345"
    _api_key_hash = hash_token(_api_key)

    user_api_key_dict = UserAPIKeyAuth(
        api_key=_api_key_hash,
        rpm_limit=2,
    )

    local_cache = DualCache()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )

    # Mock should_rate_limit to return a status with descriptor_key that doesn't match descriptors
    async def mock_should_rate_limit(descriptors, **kwargs):
        # Return a status with a mismatched descriptor_key to test fallback
        return {
            "overall_code": "OVER_LIMIT",
            "statuses": [
                {
                    "code": "OVER_LIMIT",
                    "current_limit": 2,
                    "limit_remaining": -1,
                    "rate_limit_type": "requests",
                    "descriptor_key": "nonexistent_key"  # This won't match any descriptor
                }
            ]
        }

    parallel_request_handler.should_rate_limit = mock_should_rate_limit

    # Test the pre-call hook - should handle missing descriptor gracefully
    with pytest.raises(HTTPException) as exc_info:
        await parallel_request_handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={"model": "gpt-3.5-turbo"},
            call_type="",
        )

    # Verify the exception uses fallback values
    assert exc_info.value.status_code == 429
    assert "Rate limit exceeded for nonexistent_key: unknown" in exc_info.value.detail
    assert "requests" in exc_info.value.detail
    assert "Current limit: 2" in exc_info.value.detail


@pytest.mark.asyncio
async def test_execute_token_increment_script_cluster_compatibility():
    """
    Test that token increment script execution handles Redis cluster compatibility
    by grouping operations by slot.
    
    This ensures token increments work correctly in cluster environments.
    """
    from typing import List
    from unittest.mock import AsyncMock, patch

    from litellm.types.caching import RedisPipelineIncrementOperation
    
    local_cache = DualCache()
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )
    
    # Mock _is_redis_cluster to return True for this test
    with patch.object(handler, '_is_redis_cluster', return_value=True):
        # Mock script
        mock_script = AsyncMock()
        handler.token_increment_script = mock_script
        
        # Create pipeline operations with different hash tags
        pipeline_operations: List[RedisPipelineIncrementOperation] = [
            {
                "key": "{api_key:sk-123}:tokens",
                "increment_value": 100,
                "ttl": 60
            },
            {
                "key": "{api_key:sk-123}:max_parallel_requests", 
                "increment_value": -1,
                "ttl": 60
            },
            {
                "key": "{user:user-456}:tokens",
                "increment_value": 50,
                "ttl": 60
            }
        ]
        
        # Execute the method
        await handler._execute_token_increment_script(pipeline_operations)
        
        # Verify script was called (at least once, possibly more depending on slot grouping)
        assert mock_script.call_count >= 1, "Script should be called at least once"
        
        call_args_list = mock_script.call_args_list
        
        # Verify all operations were processed
        all_processed_keys = []
        for call_args in call_args_list:
            all_processed_keys.extend(call_args[1]['keys'])
        
        # Should have processed all 3 keys
        expected_keys = {
            "{api_key:sk-123}:tokens",
            "{api_key:sk-123}:max_parallel_requests",
            "{user:user-456}:tokens"
        }
        assert set(all_processed_keys) == expected_keys, "All operation keys should be processed"
        
        # Verify args structure is correct for each call
        for call_args in call_args_list:
            keys = call_args[1]['keys']
            args = call_args[1]['args']
            # Each key should have 2 args (increment_value, ttl)
            assert len(args) == len(keys) * 2, f"Each key should have 2 args, got {len(args)} args for {len(keys)} keys"
