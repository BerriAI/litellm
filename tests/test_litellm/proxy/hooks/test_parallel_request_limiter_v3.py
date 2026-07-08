"""
Unit Tests for the max parallel request limiter v3 for the proxy
"""

import asyncio
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pytest
from fastapi import HTTPException

import litellm
from litellm import Router
from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    MAX_PARALLEL_SLOT_ACQUIRED_KEY,
    PARALLEL_REQUEST_SLOT_TTL_SECONDS,
)
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    _PROXY_MaxParallelRequestsHandler_v3 as _PROXY_MaxParallelRequestsHandler,
)
from litellm.proxy.utils import InternalUsageCache, ProxyLogging, hash_token
from litellm.types.caching import RedisPipelineIncrementOperation
from litellm.types.utils import (
    EmbeddingResponse,
    ModelResponse,
    TextCompletionResponse,
    Usage,
)


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


@pytest.mark.parametrize(
    "throttle_pct, expected_rpm, expected_tpm",
    [
        (None, 100, 1000),  # no throttle -> configured limits
        (0.1, 10, 100),  # 10% of configured
        (0.5, 50, 500),
    ],
)
def test_api_key_descriptor_applies_budget_throttle(
    throttle_pct, expected_rpm, expected_tpm
):
    """The api_key rate-limit descriptor scales the key's configured TPM/RPM by
    the request-scoped budget_throttle_pct, leaving the configured limits intact."""
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache())
    )
    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-throttle"),
        rpm_limit=100,
        tpm_limit=1000,
        budget_throttle_pct=throttle_pct,
    )

    descriptors = handler._create_rate_limit_descriptors(
        user_api_key_dict=user_api_key_dict,
        data={},
        rpm_limit_type=None,
        tpm_limit_type=None,
        model_has_failures=False,
    )

    api_key_descriptor = next(d for d in descriptors if d["key"] == "api_key")
    assert api_key_descriptor["rate_limit"]["requests_per_unit"] == expected_rpm
    assert api_key_descriptor["rate_limit"]["tokens_per_unit"] == expected_tpm


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
async def test_normal_router_call_tpm_v3(
    monkeypatch, rate_limit_object, time_controller
):
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

    # First request should succeed. Include messages + a tight max_tokens so
    # the atomic reserve_tpm_tokens path populates the :tokens counter with a
    # predictable amount — the pre-call hook no longer touches :tokens via
    # should_rate_limit.
    # Estimate: input ~ 1 token (`"hi"`), max_tokens = 5 → reservation = 6,
    # which fits under the tpm_limit of 10.
    pre_call_data = {
        "model": "azure-model",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 5,
    }
    expected_reservation = parallel_request_handler._estimate_tokens_for_request(
        data=pre_call_data
    )
    assert (
        expected_reservation < 10
    ), "Test premise: reservation must fit under tpm_limit=10"

    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data=pre_call_data,
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

    # Verify the token count is tracked (populated by reserve_tpm_tokens).
    counter_value = await local_cache.async_get_cache(key=counter_key)
    print(f"local_cache: {local_cache.in_memory_cache.cache_dict}")

    assert (
        counter_value is not None
    ), f"Counter value should be stored in cache for {counter_key}"

    # Manually increment the token counter to simulate token usage from previous call
    # This simulates what would happen after a successful call
    await local_cache.async_increment_cache(
        key=counter_key, value=15, ttl=2
    )  # Use up most of our 10 token limit

    # Make another request to test rate limiting - this should fail as we've consumed tokens
    with pytest.raises(HTTPException) as exc_info:
        await parallel_request_handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data=pre_call_data,
            call_type="",
        )

    # Wait for window to expire
    time_controller.advance(3)

    # Make request after window expiry
    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data=pre_call_data,
        call_type="",
    )

    # Verify new window — counter resets and is repopulated to the new
    # reservation amount (no longer the +1-per-request inflation artifact).
    final_counter_value = await local_cache.async_get_cache(key=counter_key)

    assert final_counter_value == expected_reservation, (
        f"Counter should reset to a fresh reservation ({expected_reservation}) "
        f"after window expiry, got {final_counter_value}"
    )


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
    # Use standard_logging_object which is the canonical source for metadata
    mock_kwargs = {
        "standard_logging_object": {
            "metadata": {
                "user_api_key_hash": _api_key,
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
        len(captured_operations) == 1
    ), "Should have 1 operation: the TPM increment (parallel slots are released via the gauge, not the pipeline)"

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


@pytest.mark.parametrize(
    "response_obj",
    [
        EmbeddingResponse(
            model="text-embedding-3-small",
            usage=Usage(prompt_tokens=50, completion_tokens=0, total_tokens=50),
        ),
        TextCompletionResponse(
            model="gpt-3.5-turbo-instruct",
            usage=Usage(prompt_tokens=20, completion_tokens=30, total_tokens=50),
        ),
    ],
)
@pytest.mark.asyncio
async def test_async_log_success_event_counts_non_chat_response_tokens(
    monkeypatch, response_obj
):
    """
    Embedding and text completion responses must increment the TPM counter,
    not just chat completion ModelResponse objects.
    """
    monkeypatch.setenv("LITELLM_RATE_LIMIT_WINDOW_SIZE", "60")

    _api_key = hash_token("sk-12345")
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache())
    )
    monkeypatch.setattr(
        parallel_request_handler, "get_rate_limit_type", lambda: "total"
    )

    mock_kwargs = {
        "standard_logging_object": {"metadata": {"user_api_key_hash": _api_key}},
        "model": response_obj.model,
    }

    captured_operations = []

    async def mock_increment_pipeline(increment_list, **kwargs):
        captured_operations.extend(increment_list)
        return True

    monkeypatch.setattr(
        parallel_request_handler.internal_usage_cache.dual_cache,
        "async_increment_cache_pipeline",
        mock_increment_pipeline,
    )

    await parallel_request_handler.async_log_success_event(
        kwargs=mock_kwargs,
        response_obj=response_obj,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    tpm_operation = next(
        (op for op in captured_operations if op["key"].endswith(":tokens")), None
    )
    assert tpm_operation is not None, "Should have a TPM increment operation"
    assert tpm_operation["increment_value"] == 50


@pytest.mark.asyncio
async def test_async_log_failure_event_v3():
    """
    async_log_failure_event releases exactly this request's slot id: the
    first release removes it, and repeated or unknown-slot releases are
    no-ops that can never free another request's slot (releasing more than
    was acquired is what previously let concurrency exceed the limit).
    """
    _api_key = "sk-12345"
    _api_key = hash_token(_api_key)
    local_cache = DualCache()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )
    counter_key = f"{{api_key:{_api_key}}}:max_parallel_requests"

    await _seed_max_parallel_requests_slots(local_cache, counter_key, ["slot-a", "slot-b"])

    def kwargs_with_slot(slot_id):
        return {
            "metadata": {
                MAX_PARALLEL_SLOT_ACQUIRED_KEY: {
                    "slot_id": slot_id,
                    "counter_keys": [counter_key],
                }
            },
            "standard_logging_object": {"metadata": {"user_api_key_hash": _api_key}},
        }

    async def in_flight():
        return parallel_request_handler._gauge_in_flight_from_cache_value(
            await local_cache.async_get_cache(key=counter_key)
        )

    await parallel_request_handler.async_log_failure_event(
        kwargs=kwargs_with_slot("slot-a"), response_obj=None, start_time=None, end_time=None
    )
    assert await in_flight() == 1

    for slot_id in ("slot-a", "slot-unknown", "slot-a"):
        await parallel_request_handler.async_log_failure_event(
            kwargs=kwargs_with_slot(slot_id), response_obj=None, start_time=None, end_time=None
        )
    assert await in_flight() == 1

    await parallel_request_handler.async_log_failure_event(
        kwargs=kwargs_with_slot("slot-b"), response_obj=None, start_time=None, end_time=None
    )
    assert await in_flight() == 0


@pytest.mark.asyncio
async def test_failure_event_without_acquired_slot_does_not_release_v3():
    """
    Failure callbacks also fire for requests rejected at pre-call, which never
    acquired a parallel slot. Releasing on those frees a slot still owned by
    another in-flight request, so every 429 would raise effective concurrency
    above the configured limit. Without the acquired-slot marker the gauge
    must stay untouched.
    """
    _api_key = hash_token("sk-12345")
    local_cache = DualCache()
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )
    counter_key = f"{{api_key:{_api_key}}}:max_parallel_requests"

    await _seed_max_parallel_requests_slots(
        local_cache, counter_key, ["slot-a", "slot-b", "slot-c"]
    )

    await handler.async_log_failure_event(
        kwargs={
            "standard_logging_object": {"metadata": {"user_api_key_hash": _api_key}}
        },
        response_obj=None,
        start_time=None,
        end_time=None,
    )
    assert (
        handler._gauge_in_flight_from_cache_value(
            await local_cache.async_get_cache(key=counter_key)
        )
        == 3
    )


@pytest.mark.asyncio
async def test_max_parallel_requests_not_reset_by_window_roll_v3():
    """
    max_parallel_requests is a concurrency gauge, not a windowed counter: the
    rate-limit window rolling over must not reset it while requests are still
    in flight. Previously the gauge shared the sliding-window reset with
    RPM/TPM, so every window roll forgot all in-flight requests and admitted
    a fresh batch of `limit` on top of what was still running.
    """
    controller = TimeController()
    local_cache = DualCache()
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache),
        time_provider=controller.now,
    )
    _api_key = hash_token("sk-12345")
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key, max_parallel_requests=2)

    for _ in range(2):
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={"model": "gpt-3.5-turbo"},
            call_type="",
        )

    controller.advance(handler.window_size + 1)

    with pytest.raises(HTTPException) as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={"model": "gpt-3.5-turbo"},
            call_type="",
        )
    assert exc_info.value.status_code == 429
    assert "max_parallel_requests" in exc_info.value.detail


@pytest.mark.asyncio
async def test_rejected_request_does_not_consume_parallel_slot_v3():
    """
    A 429-rejected request must not occupy a parallel-request slot: nothing
    ever releases a slot for a request that was never admitted, so the old
    increment-then-check behavior wedged the gauge above the limit and
    rejected requests that should have been admitted after a release.
    """
    local_cache = DualCache()
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )
    _api_key = hash_token("sk-12345")
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key, max_parallel_requests=1)

    admitted_data: Dict[str, Any] = {"model": "gpt-3.5-turbo"}
    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data=admitted_data,
        call_type="",
    )
    acquisition = admitted_data["metadata"][MAX_PARALLEL_SLOT_ACQUIRED_KEY]
    assert isinstance(acquisition, dict)
    assert isinstance(acquisition["slot_id"], str) and acquisition["slot_id"]
    assert acquisition["counter_keys"] == [f"{{api_key:{_api_key}}}:max_parallel_requests"]

    for _ in range(3):
        with pytest.raises(HTTPException):
            await handler.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=local_cache,
                data={"model": "gpt-3.5-turbo"},
                call_type="",
            )

    await handler.async_log_failure_event(
        kwargs={
            "metadata": {MAX_PARALLEL_SLOT_ACQUIRED_KEY: acquisition},
            "standard_logging_object": {"metadata": {"user_api_key_hash": _api_key}},
        },
        response_obj=None,
        start_time=None,
        end_time=None,
    )

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data={"model": "gpt-3.5-turbo"},
        call_type="",
    )


@pytest.mark.asyncio
async def test_parallel_gauge_uses_atomic_redis_script_v3():
    """
    With Redis available, gauge admission goes through the atomic
    check-and-acquire script (limit, slot TTL, and this request's slot id as
    args), the returned in-flight count is mirrored into the local cache,
    and an over-limit script result maps to a 429 without occupying a slot.
    """
    local_cache = DualCache()
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )
    _api_key = hash_token("sk-12345")
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key, max_parallel_requests=5)
    counter_key = f"{{api_key:{_api_key}}}:max_parallel_requests"

    captured_calls = []

    async def fake_acquire(keys, args):
        captured_calls.append((list(keys), list(args)))
        return [0, 3]

    handler.parallel_acquire_script = fake_acquire

    data: Dict[str, Any] = {"model": "gpt-3.5-turbo"}
    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data=data,
        call_type="",
    )
    stashed_acquisition = data["metadata"][MAX_PARALLEL_SLOT_ACQUIRED_KEY]
    assert isinstance(stashed_acquisition, dict)
    stashed_slot_id = stashed_acquisition["slot_id"]
    assert isinstance(stashed_slot_id, str) and stashed_slot_id
    assert stashed_acquisition["counter_keys"] == [counter_key]
    assert captured_calls == [
        ([counter_key], [5, PARALLEL_REQUEST_SLOT_TTL_SECONDS, stashed_slot_id])
    ]
    assert (
        await handler.internal_usage_cache.async_get_cache(
            key=counter_key, litellm_parent_otel_span=None, local_only=True
        )
        == 3
    )
    gauge_statuses = [
        s
        for s in data["litellm_proxy_rate_limit_response"]["statuses"]
        if s["rate_limit_type"] == "max_parallel_requests"
    ]
    assert gauge_statuses == [
        {
            "code": "OK",
            "current_limit": 5,
            "limit_remaining": 2,
            "rate_limit_type": "max_parallel_requests",
            "descriptor_key": "api_key",
        }
    ]

    async def fake_acquire_over_limit(keys, args):
        return [1, 1, 5, 5]

    handler.parallel_acquire_script = fake_acquire_over_limit

    with pytest.raises(HTTPException) as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={"model": "gpt-3.5-turbo"},
            call_type="",
        )
    assert exc_info.value.status_code == 429
    assert "max_parallel_requests" in exc_info.value.detail


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
            "statuses": [
                {
                    "code": "OK",
                    "current_limit": 2,
                    "limit_remaining": 1,
                    "rate_limit_type": "requests",
                    "descriptor_key": "model_per_key",
                },
                {
                    "code": "OVER_LIMIT",
                    "current_limit": 2,
                    "limit_remaining": -18,
                    "rate_limit_type": "tokens",
                    "descriptor_key": "model_per_key",
                },
            ],
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
        error = e
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
    assert (
        model_per_key_descriptor["value"] == f"{_api_key_hash}:{model}"
    ), "Api-Key value should combine api_key and model"
    assert (
        model_per_key_descriptor["rate_limit"]["requests_per_unit"] == rpm_limit
    ), "Api-Key RPM limit should be set"
    assert (
        model_per_key_descriptor["rate_limit"]["tokens_per_unit"] == tpm_limit
    ), "Api-Key TPM limit should be set"


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
            "statuses": [
                {
                    "code": "OVER_LIMIT",
                    "current_limit": 2,
                    "limit_remaining": -2,
                    "rate_limit_type": "requests",
                    "descriptor_key": "model_per_key",
                },
                {
                    "code": "OK",
                    "current_limit": 2,
                    "limit_remaining": 2,
                    "rate_limit_type": "tokens",
                    "descriptor_key": "model_per_key",
                },
            ],
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
        error = e
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
    assert (
        model_per_key_descriptor["value"] == f"{_api_key_hash}:{model}"
    ), "Api-Key value should combine api_key and model"
    assert (
        model_per_key_descriptor["rate_limit"]["requests_per_unit"] == rpm_limit
    ), "Api-Key RPM limit should be set"
    assert (
        model_per_key_descriptor["rate_limit"]["tokens_per_unit"] == tpm_limit
    ), "Api-Key TPM limit should be set"


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
        return {"overall_code": "OK", "statuses": []}

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

    assert (
        team_member_descriptor is not None
    ), "Team member descriptor should be present"
    assert (
        team_member_descriptor["value"] == f"{_team_id}:{_user_id}"
    ), "Team member value should combine team_id and user_id"
    assert (
        team_member_descriptor["rate_limit"]["requests_per_unit"] == 10
    ), "Team member RPM limit should be set"
    assert (
        team_member_descriptor["rate_limit"]["tokens_per_unit"] == 1000
    ), "Team member TPM limit should be set"


@pytest.mark.asyncio
async def test_team_member_rate_limits_v3_raises_429_when_over_limit():
    """
    When should_rate_limit reports OVER_LIMIT for the team_member descriptor, the
    pre-call hook raises HTTP 429 with rate_limit headers — same contract as
    test_rpm_api_key_rate_limits_v3 / test_tpm_api_key_rate_limits_v3.
    """
    _api_key = hash_token("sk-12345")
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

    captured_descriptors = None

    async def mock_should_rate_limit(descriptors, **kwargs):
        nonlocal captured_descriptors
        captured_descriptors = descriptors
        return {
            "overall_code": "OVER_LIMIT",
            "statuses": [
                {
                    "code": "OVER_LIMIT",
                    "current_limit": 10,
                    "limit_remaining": -1,
                    "rate_limit_type": "requests",
                    "descriptor_key": "team_member",
                },
                {
                    "code": "OK",
                    "current_limit": 1000,
                    "limit_remaining": 500,
                    "rate_limit_type": "tokens",
                    "descriptor_key": "team_member",
                },
            ],
        }

    parallel_request_handler.should_rate_limit = mock_should_rate_limit

    error = None
    try:
        await parallel_request_handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={"model": "gpt-3.5-turbo"},
            call_type="",
        )
    except HTTPException as e:
        error = e
        assert e.status_code == 429
        assert "rate_limit_type" in e.headers
        assert e.headers.get("rate_limit_type") == "requests"
        assert "retry-after" in e.headers

    assert error is not None, "An Exception must be thrown"
    assert captured_descriptors is not None, "Rate limit descriptors should be captured"
    team_member_descriptor = None
    for descriptor in captured_descriptors:
        if descriptor["key"] == "team_member":
            team_member_descriptor = descriptor
            break
    assert team_member_descriptor is not None
    assert team_member_descriptor["value"] == f"{_team_id}:{_user_id}"


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


@pytest.mark.flaky(retries=3, delay=2)
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
        pytest.skip(
            "Token increment script not available - Redis Lua scripting may not be supported"
        )

    # Test keys - use hash tags to ensure they map to same Redis cluster slot
    # Use a unique suffix per test run to avoid stale state from prior runs
    import uuid

    unique_suffix = str(uuid.uuid4())[:8]
    test_key_with_ttl = f"{{test_ttl}}:with_ttl:{unique_suffix}"
    test_key_without_ttl = f"{{test_ttl}}:without_ttl:{unique_suffix}"

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
                key=test_key_with_ttl, increment_value=10.0, ttl=60
            ),
            RedisPipelineIncrementOperation(
                key=test_key_without_ttl, increment_value=5.0, ttl=None  # No TTL
            ),
        ]

        # Execute first increment
        await parallel_request_handler.async_increment_tokens_with_ttl_preservation(
            pipeline_operations=pipeline_operations_first
        )

        # Small delay to ensure Redis has processed the commands
        await asyncio.sleep(0.1)

        # Verify keys exist and check initial TTL
        ttl_after_first = await redis_cache.async_get_ttl(test_key_with_ttl)
        value_after_first_with_ttl = await redis_cache.async_get_cache(
            test_key_with_ttl
        )
        value_after_first_without_ttl = await redis_cache.async_get_cache(
            test_key_without_ttl
        )

        assert (
            value_after_first_with_ttl == 10.0
        ), f"First increment should set value to 10.0, got {value_after_first_with_ttl}"
        assert (
            value_after_first_without_ttl == 5.0
        ), "First increment should set value to 5.0"
        assert (
            ttl_after_first is not None and ttl_after_first > 0
        ), "Key with TTL should have positive TTL after first increment"
        assert ttl_after_first <= 60, "TTL should not exceed the set value"

        # Check TTL for key without TTL (should be None, meaning no expiry)
        ttl_no_ttl_key = await redis_cache.async_get_ttl(test_key_without_ttl)
        assert (
            ttl_no_ttl_key is None
        ), "Key without TTL should have no expiry (None from async_get_ttl)"

        # Wait a moment to ensure TTL decreases
        await asyncio.sleep(2)

        # Second increment: Same operations to test TTL preservation
        pipeline_operations_second = [
            RedisPipelineIncrementOperation(
                key=test_key_with_ttl, increment_value=15.0, ttl=60  # Same TTL value
            ),
            RedisPipelineIncrementOperation(
                key=test_key_without_ttl, increment_value=7.0, ttl=None  # No TTL
            ),
        ]

        # Execute second increment
        await parallel_request_handler.async_increment_tokens_with_ttl_preservation(
            pipeline_operations=pipeline_operations_second
        )

        # Small delay to ensure Redis has processed the commands
        await asyncio.sleep(0.1)

        # Verify TTL preservation and value updates
        ttl_after_second = await redis_cache.async_get_ttl(test_key_with_ttl)
        value_after_second_with_ttl = await redis_cache.async_get_cache(
            test_key_with_ttl
        )
        value_after_second_without_ttl = await redis_cache.async_get_cache(
            test_key_without_ttl
        )

        assert (
            value_after_second_with_ttl == 25.0
        ), "Second increment should update value to 25.0"
        assert (
            value_after_second_without_ttl == 12.0
        ), "Second increment should update value to 12.0"

        # Critical test: TTL should be preserved (not reset to 60)
        assert ttl_after_second is not None, "TTL should still exist"
        assert (
            ttl_after_second < ttl_after_first
        ), "TTL should have decreased (not been reset)"
        assert ttl_after_second > 0, "TTL should still be positive"

        # TTL should not be close to the original 60 seconds (proving it wasn't reset)
        assert (
            ttl_after_second < 59
        ), "TTL should be significantly less than original, proving preservation"

        # Key without TTL should still have no expiry
        ttl_no_ttl_key_after_second = await redis_cache.async_get_ttl(
            test_key_without_ttl
        )
        assert (
            ttl_no_ttl_key_after_second is None
        ), "Key without TTL should still have no expiry"

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
    original_method = (
        parallel_request_handler.internal_usage_cache.dual_cache.async_increment_cache_pipeline
    )

    async def mock_fallback(*args, **kwargs):
        nonlocal fallback_called
        fallback_called = True
        return await original_method(*args, **kwargs)

    parallel_request_handler.internal_usage_cache.dual_cache.async_increment_cache_pipeline = (
        mock_fallback
    )

    # Test operations
    pipeline_operations = [
        RedisPipelineIncrementOperation(
            key="test_fallback_key", increment_value=10.0, ttl=60
        )
    ]

    # Execute increment
    await parallel_request_handler.async_increment_tokens_with_ttl_preservation(
        pipeline_operations=pipeline_operations
    )

    # Verify fallback was called
    assert (
        fallback_called
    ), "Fallback method should be called when Lua script is not available"


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
        "no_hash_tag_key",
    ]

    # Group the keys (should be single group for regular Redis)
    groups = handler._group_keys_by_hash_tag(test_keys)

    # Verify all keys are in single group for regular Redis
    assert len(groups) == 1, f"Expected 1 group for regular Redis, got {len(groups)}"
    assert "all_keys" in groups, "Expected 'all_keys' group for regular Redis"
    assert set(groups["all_keys"]) == set(
        test_keys
    ), "All keys should be in single group"


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
    with patch.object(handler, "_is_redis_cluster", return_value=True):
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
            assert group_key.startswith(
                "slot_"
            ), f"Group key {group_key} should start with 'slot_'"

        # Verify all original keys are present across groups
        all_grouped_keys = []
        for group_keys in groups.values():
            all_grouped_keys.extend(group_keys)
        assert set(all_grouped_keys) == set(
            test_keys
        ), "All keys should be present in groups"


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
    with patch.object(handler, "_is_redis_cluster", return_value=True):
        # Mock script that simulates Redis cluster slot conflict
        mock_script = AsyncMock()
        mock_script.side_effect = [
            Exception(
                "EVALSHA - all keys must map to the same key slot"
            ),  # First group fails
            [1234, 1, 1234, 2],  # Second group succeeds
        ]
        handler.batch_rate_limiter_script = mock_script

        # Mock in-memory fallback (returns 2 values for 2 keys: window_start, counter)
        handler.in_memory_cache_sliding_window = AsyncMock(return_value=[1234, 1])

        # Test keys from different hash tags (would fail in cluster without grouping)
        test_keys = [
            "{api_key:sk-123}:window",
            "{api_key:sk-123}:requests",
            "{user:user-456}:window",
            "{user:user-456}:requests",
        ]

        # Execute the method
        results = await handler._execute_redis_batch_rate_limiter_script(
            keys_to_fetch=test_keys, now_int=1234
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
            all_processed_keys.extend(call_args[1]["keys"])

        # Should have processed all keys (some might be duplicated due to fallback)
        unique_processed_keys = set(all_processed_keys)
        assert (
            len(unique_processed_keys) >= 2
        ), "Should have processed at least some keys"


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
                    "descriptor_key": "api_key",
                },
                {
                    "code": "OK",
                    "current_limit": 10,
                    "limit_remaining": 8,
                    "rate_limit_type": "tokens",
                    "descriptor_key": "api_key",
                },
                {
                    "code": "OVER_LIMIT",
                    "current_limit": 1,
                    "limit_remaining": -1,
                    "rate_limit_type": "max_parallel_requests",
                    "descriptor_key": "api_key",
                },
            ],
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
                    "descriptor_key": "nonexistent_key",  # This won't match any descriptor
                }
            ],
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
async def test_get_rate_limit_type_default_is_total(monkeypatch):
    """
    Test that get_rate_limit_type returns 'total' as the default when no setting is specified.

    This verifies the change from 'output' to 'total' as the default value.
    """
    local_cache = DualCache()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )

    # Mock general_settings to return empty dict (no token_rate_limit_type set)
    import litellm.proxy.proxy_server as proxy_server

    original_settings = getattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(proxy_server, "general_settings", {})

    try:
        result = parallel_request_handler.get_rate_limit_type()
        assert (
            result == "total"
        ), f"Default rate limit type should be 'total', got '{result}'"
    finally:
        monkeypatch.setattr(proxy_server, "general_settings", original_settings)


@pytest.mark.asyncio
async def test_get_rate_limit_type_invalid_falls_back_to_total(monkeypatch):
    """
    Test that get_rate_limit_type falls back to 'total' when an invalid value is specified.
    """
    local_cache = DualCache()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )

    # Mock general_settings to return an invalid token_rate_limit_type
    import litellm.proxy.proxy_server as proxy_server

    original_settings = getattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(
        proxy_server, "general_settings", {"token_rate_limit_type": "invalid_type"}
    )

    try:
        result = parallel_request_handler.get_rate_limit_type()
        assert (
            result == "total"
        ), f"Invalid rate limit type should fall back to 'total', got '{result}'"
    finally:
        monkeypatch.setattr(proxy_server, "general_settings", original_settings)


@pytest.mark.parametrize(
    "token_rate_limit_type,expected_field",
    [
        ("input", "prompt_tokens"),
        ("output", "completion_tokens"),
        ("total", "total_tokens"),
    ],
)
@pytest.mark.asyncio
async def test_async_log_success_event_with_dict_usage(
    monkeypatch, token_rate_limit_type, expected_field
):
    """
    Test that async_log_success_event correctly handles usage as a dict (Responses API format).

    The Responses API returns usage as a dict in ResponsesAPIResponse instead of a Usage object.
    This test verifies that token counting works correctly with dict-based usage.
    """
    from unittest.mock import MagicMock

    _api_key = "sk-12345"
    _api_key = hash_token(_api_key)
    local_cache = DualCache()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )

    # Mock the get_rate_limit_type method
    def mock_get_rate_limit_type():
        return token_rate_limit_type

    monkeypatch.setattr(
        parallel_request_handler, "get_rate_limit_type", mock_get_rate_limit_type
    )

    # Create a mock response object with usage as a dict (Responses API format)
    from litellm.types.utils import BaseLiteLLMOpenAIResponseObject

    # Use spec to make isinstance checks work correctly with MagicMock
    mock_response = MagicMock(spec=BaseLiteLLMOpenAIResponseObject)
    mock_response.usage = {
        "prompt_tokens": 25,
        "completion_tokens": 35,
        "total_tokens": 60,
    }

    # Create mock kwargs for the success event
    mock_kwargs = {
        "standard_logging_object": {
            "metadata": {
                "user_api_key_hash": _api_key,
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

    # Find the TPM increment operation
    tpm_operation = None
    for op in captured_operations:
        if op["key"].endswith(":tokens"):
            tpm_operation = op
            break

    assert tpm_operation is not None, "Should have a TPM increment operation"

    # Check that the correct token count was used based on the rate limit type
    expected_tokens = {
        "input": 25,  # prompt_tokens
        "output": 35,  # completion_tokens
        "total": 60,  # total_tokens
    }

    assert (
        tpm_operation["increment_value"] == expected_tokens[token_rate_limit_type]
    ), f"Expected {expected_tokens[token_rate_limit_type]} tokens for type '{token_rate_limit_type}', got {tpm_operation['increment_value']}"


@pytest.mark.asyncio
async def test_async_log_success_event_with_dict_usage_missing_fields(monkeypatch):
    """
    Test that async_log_success_event handles dict usage with missing fields gracefully.

    When usage dict is missing expected fields, it should default to 0.
    """
    from unittest.mock import MagicMock

    _api_key = "sk-12345"
    _api_key = hash_token(_api_key)
    local_cache = DualCache()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )

    # Mock the get_rate_limit_type method
    def mock_get_rate_limit_type():
        return "output"

    monkeypatch.setattr(
        parallel_request_handler, "get_rate_limit_type", mock_get_rate_limit_type
    )

    # Create a mock response object with usage as a dict missing some fields
    mock_response = MagicMock()
    mock_response.usage = {
        "prompt_tokens": 25,
        # completion_tokens is missing
        # total_tokens is missing
    }
    from litellm.types.utils import BaseLiteLLMOpenAIResponseObject

    mock_response.__class__ = type(
        "MockResponse", (BaseLiteLLMOpenAIResponseObject,), {}
    )

    # Create mock kwargs for the success event
    mock_kwargs = {
        "standard_logging_object": {
            "metadata": {
                "user_api_key_hash": _api_key,
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

    # Call the success event handler - should not raise exception
    await parallel_request_handler.async_log_success_event(
        kwargs=mock_kwargs,
        response_obj=mock_response,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    # When total_tokens resolves to 0 (missing fields) and there's no reservation,
    # the reconciliation delta is 0 — no TPM increment should be emitted.
    tpm_ops = [op for op in captured_operations if op["key"].endswith(":tokens")]
    assert tpm_ops == [], f"Expected no TPM ops when usage is empty, got: {tpm_ops}"


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
    with patch.object(handler, "_is_redis_cluster", return_value=True):
        # Mock script
        mock_script = AsyncMock()
        handler.token_increment_script = mock_script

        # Create pipeline operations with different hash tags
        pipeline_operations: List[RedisPipelineIncrementOperation] = [
            {"key": "{api_key:sk-123}:tokens", "increment_value": 100, "ttl": 60},
            {
                "key": "{api_key:sk-123}:max_parallel_requests",
                "increment_value": -1,
                "ttl": 60,
            },
            {"key": "{user:user-456}:tokens", "increment_value": 50, "ttl": 60},
        ]

        # Execute the method
        await handler._execute_token_increment_script(pipeline_operations)

        # Verify script was called (at least once, possibly more depending on slot grouping)
        assert mock_script.call_count >= 1, "Script should be called at least once"

        call_args_list = mock_script.call_args_list

        # Verify all operations were processed
        all_processed_keys = []
        for call_args in call_args_list:
            all_processed_keys.extend(call_args[1]["keys"])

        # Should have processed all 3 keys
        expected_keys = {
            "{api_key:sk-123}:tokens",
            "{api_key:sk-123}:max_parallel_requests",
            "{user:user-456}:tokens",
        }
        assert (
            set(all_processed_keys) == expected_keys
        ), "All operation keys should be processed"

        # Verify args structure is correct for each call
        for call_args in call_args_list:
            keys = call_args[1]["keys"]
            args = call_args[1]["args"]
            # Each key should have 2 args (increment_value, ttl)
            assert (
                len(args) == len(keys) * 2
            ), f"Each key should have 2 args, got {len(args)} args for {len(keys)} keys"


@pytest.mark.asyncio
async def test_agent_level_rate_limit_descriptors():
    """
    Test that agent-level rate limit descriptors are created when
    an agent has rpm_limit and/or tpm_limit configured.
    """
    from unittest.mock import patch

    from litellm.types.agents import AgentResponse

    _api_key = "sk-12345"
    _api_key = hash_token(_api_key)
    _agent_id = "agent_abc123"

    user_api_key_dict = UserAPIKeyAuth(
        api_key=_api_key,
        agent_id=_agent_id,
    )

    local_cache = DualCache()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )

    mock_agent = AgentResponse(
        agent_id=_agent_id,
        agent_name="test-agent",
        agent_card_params={"name": "Test Agent"},
        rpm_limit=50,
        tpm_limit=5000,
    )

    captured_descriptors = None

    async def mock_should_rate_limit(descriptors, **kwargs):
        nonlocal captured_descriptors
        captured_descriptors = descriptors
        return {"overall_code": "OK", "statuses": []}

    parallel_request_handler.should_rate_limit = mock_should_rate_limit

    with patch(
        "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry.get_agent_by_id",
        return_value=mock_agent,
    ):
        await parallel_request_handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={"model": "gpt-4"},
            call_type="",
        )

    assert captured_descriptors is not None

    agent_descriptor = None
    for d in captured_descriptors:
        if d["key"] == "agent":
            agent_descriptor = d
            break

    assert agent_descriptor is not None, "Agent descriptor should be present"
    assert agent_descriptor["value"] == _agent_id
    assert agent_descriptor["rate_limit"]["requests_per_unit"] == 50
    assert agent_descriptor["rate_limit"]["tokens_per_unit"] == 5000


@pytest.mark.asyncio
async def test_agent_session_rate_limit_descriptors():
    """
    Test that session-level rate limit descriptors are created when
    an agent has session_rpm_limit/session_tpm_limit and a session_id is present.
    """
    from unittest.mock import patch

    from litellm.types.agents import AgentResponse

    _api_key = "sk-12345"
    _api_key = hash_token(_api_key)
    _agent_id = "agent_abc123"
    _session_id = "sess_xyz789"

    user_api_key_dict = UserAPIKeyAuth(
        api_key=_api_key,
        agent_id=_agent_id,
    )

    local_cache = DualCache()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )

    mock_agent = AgentResponse(
        agent_id=_agent_id,
        agent_name="test-agent",
        agent_card_params={"name": "Test Agent"},
        session_rpm_limit=10,
        session_tpm_limit=1000,
    )

    captured_descriptors = None

    async def mock_should_rate_limit(descriptors, **kwargs):
        nonlocal captured_descriptors
        captured_descriptors = descriptors
        return {"overall_code": "OK", "statuses": []}

    parallel_request_handler.should_rate_limit = mock_should_rate_limit

    with patch(
        "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry.get_agent_by_id",
        return_value=mock_agent,
    ):
        await parallel_request_handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={
                "model": "gpt-4",
                "metadata": {"session_id": _session_id},
            },
            call_type="",
        )

    assert captured_descriptors is not None

    session_descriptor = None
    for d in captured_descriptors:
        if d["key"] == "agent_session":
            session_descriptor = d
            break

    assert session_descriptor is not None, "Agent session descriptor should be present"
    assert session_descriptor["value"] == f"{_agent_id}:{_session_id}"
    assert session_descriptor["rate_limit"]["requests_per_unit"] == 10
    assert session_descriptor["rate_limit"]["tokens_per_unit"] == 1000


@pytest.mark.asyncio
async def test_agent_session_rate_limit_skipped_without_session_id():
    """
    Test that session-level rate limit descriptors are NOT created
    when no session_id is available in the request.
    """
    from unittest.mock import patch

    from litellm.types.agents import AgentResponse

    _api_key = "sk-12345"
    _api_key = hash_token(_api_key)
    _agent_id = "agent_abc123"

    user_api_key_dict = UserAPIKeyAuth(
        api_key=_api_key,
        agent_id=_agent_id,
    )

    local_cache = DualCache()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )

    mock_agent = AgentResponse(
        agent_id=_agent_id,
        agent_name="test-agent",
        agent_card_params={"name": "Test Agent"},
        session_rpm_limit=10,
        session_tpm_limit=1000,
    )

    captured_descriptors = None

    async def mock_should_rate_limit(descriptors, **kwargs):
        nonlocal captured_descriptors
        captured_descriptors = descriptors
        return {"overall_code": "OK", "statuses": []}

    parallel_request_handler.should_rate_limit = mock_should_rate_limit

    with patch(
        "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry.get_agent_by_id",
        return_value=mock_agent,
    ):
        await parallel_request_handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={"model": "gpt-4"},
            call_type="",
        )

    # should_rate_limit should not have been called (no agent-level limits, only session limits
    # but no session_id)
    assert captured_descriptors is None, (
        "No descriptors should be created when agent has only session limits "
        "but no session_id in request"
    )


@pytest.mark.asyncio
async def test_agent_rate_limit_from_metadata_agent_id():
    """
    Test that agent rate limits work when agent_id comes from
    request metadata (header) rather than from the API key.
    """
    from unittest.mock import patch

    from litellm.types.agents import AgentResponse

    _api_key = "sk-12345"
    _api_key = hash_token(_api_key)
    _agent_id = "agent_from_header"

    user_api_key_dict = UserAPIKeyAuth(
        api_key=_api_key,
    )

    local_cache = DualCache()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )

    mock_agent = AgentResponse(
        agent_id=_agent_id,
        agent_name="header-agent",
        agent_card_params={"name": "Header Agent"},
        rpm_limit=25,
        tpm_limit=2500,
    )

    captured_descriptors = None

    async def mock_should_rate_limit(descriptors, **kwargs):
        nonlocal captured_descriptors
        captured_descriptors = descriptors
        return {"overall_code": "OK", "statuses": []}

    parallel_request_handler.should_rate_limit = mock_should_rate_limit

    with patch(
        "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry.get_agent_by_id",
        return_value=mock_agent,
    ):
        await parallel_request_handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={
                "model": "gpt-4",
                "metadata": {"agent_id": _agent_id},
            },
            call_type="",
        )

    assert captured_descriptors is not None

    agent_descriptor = None
    for d in captured_descriptors:
        if d["key"] == "agent":
            agent_descriptor = d
            break

    assert (
        agent_descriptor is not None
    ), "Agent descriptor should be created from metadata agent_id"
    assert agent_descriptor["value"] == _agent_id
    assert agent_descriptor["rate_limit"]["requests_per_unit"] == 25


@pytest.mark.asyncio
async def test_agent_both_agent_and_session_rate_limits():
    """
    Test that both agent-level and session-level descriptors are created
    when both types of limits are configured on the agent.
    """
    from unittest.mock import patch

    from litellm.types.agents import AgentResponse

    _api_key = "sk-12345"
    _api_key = hash_token(_api_key)
    _agent_id = "agent_dual"
    _session_id = "sess_dual"

    user_api_key_dict = UserAPIKeyAuth(
        api_key=_api_key,
        agent_id=_agent_id,
    )

    local_cache = DualCache()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )

    mock_agent = AgentResponse(
        agent_id=_agent_id,
        agent_name="dual-agent",
        agent_card_params={"name": "Dual Agent"},
        rpm_limit=100,
        tpm_limit=10000,
        session_rpm_limit=20,
        session_tpm_limit=2000,
    )

    captured_descriptors = None

    async def mock_should_rate_limit(descriptors, **kwargs):
        nonlocal captured_descriptors
        captured_descriptors = descriptors
        return {"overall_code": "OK", "statuses": []}

    parallel_request_handler.should_rate_limit = mock_should_rate_limit

    with patch(
        "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry.get_agent_by_id",
        return_value=mock_agent,
    ):
        await parallel_request_handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={
                "model": "gpt-4",
                "metadata": {"session_id": _session_id},
            },
            call_type="",
        )

    assert captured_descriptors is not None

    agent_descriptor = None
    session_descriptor = None
    for d in captured_descriptors:
        if d["key"] == "agent":
            agent_descriptor = d
        elif d["key"] == "agent_session":
            session_descriptor = d

    assert agent_descriptor is not None, "Agent-level descriptor should be present"
    assert agent_descriptor["rate_limit"]["requests_per_unit"] == 100
    assert agent_descriptor["rate_limit"]["tokens_per_unit"] == 10000

    assert session_descriptor is not None, "Session-level descriptor should be present"
    assert session_descriptor["value"] == f"{_agent_id}:{_session_id}"
    assert session_descriptor["rate_limit"]["requests_per_unit"] == 20
    assert session_descriptor["rate_limit"]["tokens_per_unit"] == 2000


@pytest.mark.asyncio
async def test_agent_rate_limit_tpm_increment_on_success(monkeypatch):
    """
    Test that async_log_success_event increments agent and session
    TPM counters when agent_id and session_id are in metadata.
    """
    _api_key = "sk-12345"
    _api_key = hash_token(_api_key)
    _agent_id = "agent_tpm_test"
    _session_id = "sess_tpm_test"

    local_cache = DualCache()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )

    def mock_get_rate_limit_type():
        return "total"

    monkeypatch.setattr(
        parallel_request_handler, "get_rate_limit_type", mock_get_rate_limit_type
    )

    mock_usage = Usage(prompt_tokens=20, completion_tokens=30, total_tokens=50)
    mock_response = ModelResponse(
        id="mock-response",
        object="chat.completion",
        created=int(datetime.now().timestamp()),
        model="gpt-4",
        usage=mock_usage,
        choices=[],
    )

    mock_kwargs = {
        "standard_logging_object": {
            "metadata": {
                "user_api_key_hash": _api_key,
                "user_api_key_user_id": None,
                "user_api_key_team_id": None,
                "user_api_key_end_user_id": None,
                "agent_id": _agent_id,
                "session_id": _session_id,
            }
        },
        "model": "gpt-4",
    }

    captured_operations = []

    async def mock_increment_pipeline(increment_list, **kwargs):
        captured_operations.extend(increment_list)
        return True

    monkeypatch.setattr(
        parallel_request_handler.internal_usage_cache.dual_cache,
        "async_increment_cache_pipeline",
        mock_increment_pipeline,
    )

    await parallel_request_handler.async_log_success_event(
        kwargs=mock_kwargs,
        response_obj=mock_response,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    agent_tpm_op = None
    session_tpm_op = None
    for op in captured_operations:
        if op["key"] == f"{{agent:{_agent_id}}}:tokens":
            agent_tpm_op = op
        elif op["key"] == f"{{agent_session:{_agent_id}:{_session_id}}}:tokens":
            session_tpm_op = op

    assert agent_tpm_op is not None, "Agent TPM increment should be present"
    assert agent_tpm_op["increment_value"] == 50

    assert session_tpm_op is not None, "Session TPM increment should be present"
    assert session_tpm_op["increment_value"] == 50


@pytest.mark.asyncio
async def test_agent_rate_limit_429_on_over_limit(monkeypatch, time_controller):
    """
    Test end-to-end that agent rate limiting returns 429 when the agent
    RPM limit is exceeded.
    """
    from unittest.mock import patch

    from litellm.types.agents import AgentResponse

    monkeypatch.setenv("LITELLM_RATE_LIMIT_WINDOW_SIZE", "2")
    _api_key = "sk-12345"
    _api_key = hash_token(_api_key)
    _agent_id = "agent_429_test"

    user_api_key_dict = UserAPIKeyAuth(
        api_key=_api_key,
        agent_id=_agent_id,
    )

    local_cache = DualCache()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache),
        time_provider=time_controller.now,
    )

    mock_agent = AgentResponse(
        agent_id=_agent_id,
        agent_name="rate-limited-agent",
        agent_card_params={"name": "Rate Limited Agent"},
        rpm_limit=2,
    )

    window_starts: Dict[str, int] = {}
    request_counts: Dict[str, int] = {}

    async def mock_batch_rate_limiter(*args, **kwargs):
        keys = kwargs.get("keys") if kwargs else args[0]
        args_list = kwargs.get("args") if kwargs else args[1]
        now = args_list[0]
        window_size = args_list[1]
        results = []
        for i in range(0, len(keys), 2):
            window_key = keys[i]
            counter_key = keys[i + 1]
            prev_window = window_starts.get(window_key)
            prev_counter = request_counts.get(counter_key, 0)
            if prev_window is None or (now - prev_window) >= window_size:
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

    with patch(
        "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry.get_agent_by_id",
        return_value=mock_agent,
    ):
        await parallel_request_handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={"model": "gpt-4"},
            call_type="",
        )

        await parallel_request_handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={"model": "gpt-4"},
            call_type="",
        )

        with pytest.raises(HTTPException) as exc_info:
            await parallel_request_handler.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=local_cache,
                data={"model": "gpt-4"},
                call_type="",
            )

        assert exc_info.value.status_code == 429
        assert "agent" in exc_info.value.detail


class TestGetTotalTokensFromUsageCacheExclusion:
    """
    Tests for _get_total_tokens_from_usage cache token exclusion.

    Issue: AWS Bedrock and similar providers exclude cache tokens from TPM calculation,
    but LiteLLM was including them, causing up to 10x difference in rate limiting.
    """

    @pytest.fixture
    def handler(self):
        """Create a handler instance for testing."""
        local_cache = DualCache()
        return _PROXY_MaxParallelRequestsHandler(
            internal_usage_cache=InternalUsageCache(local_cache),
        )

    def test_excludes_cached_tokens_from_total(self, handler):
        """Cached tokens should be excluded from total token count."""
        from litellm.types.utils import PromptTokensDetailsWrapper

        usage = Usage(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
            prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=800),
        )

        # Total should be 1500 - 800 = 700
        result = handler._get_total_tokens_from_usage(usage, "total")
        assert result == 700, f"Expected 700 (1500 - 800 cached), got {result}"

    def test_excludes_cached_tokens_from_input(self, handler):
        """Cached tokens should be excluded from input token count."""
        from litellm.types.utils import PromptTokensDetailsWrapper

        usage = Usage(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
            prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=800),
        )

        # Input should be 1000 - 800 = 200
        result = handler._get_total_tokens_from_usage(usage, "input")
        assert result == 200, f"Expected 200 (1000 - 800 cached), got {result}"

    def test_does_not_exclude_cached_tokens_from_output(self, handler):
        """Cached tokens should NOT affect output token count."""
        from litellm.types.utils import PromptTokensDetailsWrapper

        usage = Usage(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
            prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=800),
        )

        # Output tokens should be unchanged
        result = handler._get_total_tokens_from_usage(usage, "output")
        assert result == 500, f"Expected 500 (no change for output), got {result}"

    def test_handles_no_cached_tokens(self, handler):
        """Should work correctly when no cached tokens present."""
        usage = Usage(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
        )

        result = handler._get_total_tokens_from_usage(usage, "total")
        assert result == 1500, f"Expected 1500 (no cache), got {result}"

    def test_handles_dict_usage_with_cached_tokens(self, handler):
        """Should handle dict usage format (Responses API) with cached tokens."""
        usage = {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "total_tokens": 1500,
            "prompt_tokens_details": {"cached_tokens": 600},
        }

        result = handler._get_total_tokens_from_usage(usage, "total")
        assert result == 900, f"Expected 900 (1500 - 600 cached), got {result}"

    def test_handles_none_usage(self, handler):
        """Should handle None usage gracefully."""
        result = handler._get_total_tokens_from_usage(None, "total")
        assert result == 0, f"Expected 0 for None usage, got {result}"


@pytest.mark.asyncio
async def test_project_model_rate_limits_enforced_v3():
    """
    Regression test: project-level model-specific rate limits must be enforced.

    Bug: When a key belongs to a project that has model_rpm_limit/model_tpm_limit
    in project_metadata, those limits were never checked — only model-level limits
    were applied. This test verifies the fix.
    """
    _api_key = hash_token("sk-project-test")
    local_cache = DualCache()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )

    captured_descriptors = []

    async def mock_should_rate_limit(descriptors, **kwargs):
        captured_descriptors.extend(descriptors)
        return {"overall_code": "OK", "statuses": []}

    parallel_request_handler.should_rate_limit = mock_should_rate_limit

    # Key with project_metadata containing model-specific rate limits
    user_api_key_dict = UserAPIKeyAuth(
        api_key=_api_key,
        project_id="proj-abc123",
        project_metadata={
            "model_rpm_limit": {"gpt-4": 5},
            "model_tpm_limit": {"gpt-4": 1000},
        },
    )

    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data={"model": "gpt-4"},
        call_type="",
    )

    descriptor_keys = [d["key"] for d in captured_descriptors]
    assert (
        "model_per_project" in descriptor_keys
    ), f"Expected model_per_project descriptor, got: {descriptor_keys}"

    model_per_project = next(
        d for d in captured_descriptors if d["key"] == "model_per_project"
    )
    assert model_per_project["value"] == "proj-abc123:gpt-4"
    assert model_per_project["rate_limit"]["requests_per_unit"] == 5
    assert model_per_project["rate_limit"]["tokens_per_unit"] == 1000


@pytest.mark.asyncio
async def test_project_model_rate_limits_not_triggered_for_other_model_v3():
    """Project model limits should not trigger for a model not in project_metadata."""
    _api_key = hash_token("sk-project-test-2")
    local_cache = DualCache()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )

    captured_descriptors = []

    async def mock_should_rate_limit(descriptors, **kwargs):
        captured_descriptors.extend(descriptors)
        return {"overall_code": "OK", "statuses": []}

    parallel_request_handler.should_rate_limit = mock_should_rate_limit

    user_api_key_dict = UserAPIKeyAuth(
        api_key=_api_key,
        project_id="proj-abc123",
        project_metadata={
            "model_rpm_limit": {"gpt-4": 5},
        },
    )

    # Request for gpt-3.5-turbo — project only limits gpt-4
    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data={"model": "gpt-3.5-turbo"},
        call_type="",
    )

    descriptor_keys = [d["key"] for d in captured_descriptors]
    assert (
        "model_per_project" not in descriptor_keys
    ), f"model_per_project should not be added for unrelated model, got: {descriptor_keys}"


@pytest.mark.asyncio
async def test_pre_call_hook_does_not_leak_internal_stash_to_request_body():
    """Regression for #27001: stash keys must stay in metadata, never on
    the top level of ``data`` (which gets forwarded as the provider body)."""
    from litellm.proxy.hooks.parallel_request_limiter_v3 import (
        _LITELLM_STASH_KEYS,
        RATE_LIMIT_DESCRIPTORS_KEY,
        TPM_RESERVED_TOKENS_KEY,
    )

    _api_key = hash_token("sk-leak-regression")
    user_api_key_dict = UserAPIKeyAuth(
        api_key=_api_key,
        tpm_limit=1000,
        rpm_limit=5,
    )
    local_cache = DualCache()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache),
    )

    async def mock_should_rate_limit(descriptors, **kwargs):
        return {"overall_code": "OK", "statuses": []}

    async def mock_reserve_tpm_tokens(descriptors, estimated_tokens, **kwargs):
        return {
            "overall_code": "OK",
            "statuses": [
                {
                    "code": "OK",
                    "current_limit": 1000,
                    "limit_remaining": 1000 - estimated_tokens,
                    "descriptor_key": d["key"],
                    "descriptor_value": d["value"],
                    "rate_limit_type": "tokens",
                }
                for d in descriptors
            ],
        }

    parallel_request_handler.should_rate_limit = mock_should_rate_limit
    parallel_request_handler.reserve_tpm_tokens = mock_reserve_tpm_tokens

    data: Dict[str, Any] = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 10,
    }

    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data=data,
        call_type="completion",
    )

    leaked = [k for k in _LITELLM_STASH_KEYS if k in data]
    assert not leaked, f"stash keys leaked to top level: {leaked}"

    metadata = data.get("metadata") or {}
    assert metadata.get(TPM_RESERVED_TOKENS_KEY)
    assert isinstance(metadata.get(RATE_LIMIT_DESCRIPTORS_KEY), list)


@pytest.mark.asyncio
async def test_pre_call_hook_rejects_caller_supplied_stash_values():
    """Caller cannot pre-populate stash keys in body metadata to drive a
    later TPM refund against an arbitrary scope."""
    from litellm.proxy.hooks.parallel_request_limiter_v3 import (
        _LITELLM_STASH_KEYS,
        RATE_LIMIT_DESCRIPTORS_KEY,
        TPM_RESERVED_TOKENS_KEY,
    )

    user_api_key_dict = UserAPIKeyAuth(api_key=hash_token("sk-no-limits"))
    local_cache = DualCache()
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache),
    )

    victim_descriptors = [
        {
            "key": "api_key",
            "value": "victim-key-hash",
            "rate_limit": {"tokens_per_unit": 10000, "window_size": 60},
        }
    ]
    data: Dict[str, Any] = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
        TPM_RESERVED_TOKENS_KEY: 9999,
        RATE_LIMIT_DESCRIPTORS_KEY: victim_descriptors,
        "metadata": {
            TPM_RESERVED_TOKENS_KEY: 9999,
            RATE_LIMIT_DESCRIPTORS_KEY: victim_descriptors,
        },
        "litellm_metadata": {
            TPM_RESERVED_TOKENS_KEY: 9999,
            RATE_LIMIT_DESCRIPTORS_KEY: victim_descriptors,
        },
    }

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data=data,
        call_type="completion",
    )

    for channel in (
        data,
        data.get("metadata") or {},
        data.get("litellm_metadata") or {},
    ):
        leaked = [k for k in _LITELLM_STASH_KEYS if k in channel]
        assert not leaked, f"caller-supplied stash survived in {channel!r}: {leaked}"


# ----------------------- Per-MCP-server rate limiting (v3) -----------------------


def _make_mcp_handler():
    local_cache = DualCache()
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )
    return handler, local_cache


def _find_descriptor(descriptors, key):
    return next((d for d in descriptors if d["key"] == key), None)


def _build_mcp_descriptors(handler, user_api_key_dict, data, call_type="call_mcp_tool"):
    return handler._create_rate_limit_descriptors(
        user_api_key_dict=user_api_key_dict,
        data=data,
        rpm_limit_type=None,
        tpm_limit_type=None,
        model_has_failures=False,
        call_type=call_type,
    )


def test_mcp_per_key_descriptor_created_for_matching_server_v3():
    handler, _ = _make_mcp_handler()
    api_key = hash_token("sk-mcp-key")
    user_api_key_dict = UserAPIKeyAuth(
        api_key=api_key,
        metadata={"mcp_rpm_limit": {"github": 5}},
    )

    descriptors = _build_mcp_descriptors(
        handler, user_api_key_dict, {"mcp_server_name": "github"}
    )

    descriptor = _find_descriptor(descriptors, "mcp_per_key")
    assert descriptor is not None
    assert descriptor["value"] == f"{api_key}:github"
    assert descriptor["rate_limit"]["requests_per_unit"] == 5
    # MCP tool calls have no token usage; tokens_per_unit must stay None so the
    # TPM reservation path is never engaged (otherwise budget would leak).
    assert descriptor["rate_limit"]["tokens_per_unit"] is None


def test_mcp_per_key_descriptor_skipped_for_non_matching_server_v3():
    handler, _ = _make_mcp_handler()
    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-mcp-key"),
        metadata={"mcp_rpm_limit": {"github": 5}},
    )

    descriptors = _build_mcp_descriptors(
        handler, user_api_key_dict, {"mcp_server_name": "slack"}
    )

    assert _find_descriptor(descriptors, "mcp_per_key") is None


def test_mcp_descriptor_skipped_for_non_mcp_request_v3():
    """A non-MCP request must not create an MCP descriptor even if the caller
    injects mcp_server_name in the body; otherwise an LLM call could consume a
    target server's MCP quota and 429 legitimate tool calls."""
    handler, _ = _make_mcp_handler()
    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-mcp-key"),
        metadata={"mcp_rpm_limit": {"github": 5}},
    )

    descriptors = _build_mcp_descriptors(
        handler,
        user_api_key_dict,
        {"model": "gpt-4", "mcp_server_name": "github"},
        call_type="completion",
    )

    assert _find_descriptor(descriptors, "mcp_per_key") is None


def test_mcp_descriptor_skipped_for_raw_rest_body_v3():
    handler, _ = _make_mcp_handler()
    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-mcp-key"),
        team_id="team-1",
        metadata={"mcp_rpm_limit": {"github": 5}},
        team_metadata={"mcp_rpm_limit": {"github": 3}},
    )

    descriptors = _build_mcp_descriptors(
        handler,
        user_api_key_dict,
        {
            "server_id": "slack",
            "name": "demo-tool",
            "arguments": {},
            "mcp_server_name": "github",
        },
    )

    assert _find_descriptor(descriptors, "mcp_per_key") is None
    assert _find_descriptor(descriptors, "mcp_per_team") is None


def test_mcp_per_team_descriptor_created_from_team_metadata_v3():
    handler, _ = _make_mcp_handler()
    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-mcp-key"),
        team_id="team-1",
        team_metadata={"mcp_rpm_limit": {"github": 3}},
    )

    descriptors = _build_mcp_descriptors(
        handler, user_api_key_dict, {"mcp_server_name": "github"}
    )

    descriptor = _find_descriptor(descriptors, "mcp_per_team")
    assert descriptor is not None
    assert descriptor["value"] == "team-1:github"
    assert descriptor["rate_limit"]["requests_per_unit"] == 3
    assert descriptor["rate_limit"]["tokens_per_unit"] is None


@pytest.mark.asyncio
async def test_mcp_per_key_rpm_enforced_v3(monkeypatch):
    """
    A key configured with mcp_rpm_limit={"github": 2} must allow 2 calls to the
    github MCP server within the window and reject the 3rd with a 429, while
    calls to a different MCP server are unaffected.
    """
    monkeypatch.setenv("LITELLM_RATE_LIMIT_WINDOW_SIZE", "60")
    api_key = hash_token("sk-mcp-enforce")
    local_cache = DualCache()
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )

    window_starts: Dict[str, int] = {}
    request_counts: Dict[str, int] = {}

    async def mock_batch_rate_limiter(*args, **kwargs):
        keys = kwargs.get("keys") if kwargs else args[0]
        args_list = kwargs.get("args") if kwargs else args[1]
        now = args_list[0]
        window_size = args_list[1]
        results = []
        for i in range(0, len(keys), 2):
            window_key = keys[i]
            counter_key = keys[i + 1]
            prev_window = window_starts.get(window_key)
            prev_counter = request_counts.get(counter_key, 0)
            if prev_window is None or (now - prev_window) >= window_size:
                window_starts[window_key] = now
                new_counter = 1
            else:
                new_counter = prev_counter + 1
            request_counts[counter_key] = new_counter
            results.append(now)
            results.append(new_counter)
        return results

    handler.batch_rate_limiter_script = mock_batch_rate_limiter

    user_api_key_dict = UserAPIKeyAuth(
        api_key=api_key,
        metadata={"mcp_rpm_limit": {"github": 2}},
    )

    for _ in range(2):
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={"mcp_server_name": "github"},
            call_type="call_mcp_tool",
        )

    with pytest.raises(HTTPException) as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={"mcp_server_name": "github"},
            call_type="call_mcp_tool",
        )
    assert exc_info.value.status_code == 429

    # A different server has no configured limit -> not rate limited.
    for _ in range(5):
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={"mcp_server_name": "slack"},
            call_type="call_mcp_tool",
        )

    # The TPM counter must never be created for an MCP descriptor.
    assert not any(":tokens" in key and "github" in key for key in request_counts)


def test_get_key_mcp_rpm_limit_precedence():
    from litellm.proxy.auth.auth_utils import (
        get_key_mcp_rpm_limit,
        get_team_mcp_rpm_limit,
    )

    # Key metadata takes precedence over team metadata.
    key_first = UserAPIKeyAuth(
        api_key=hash_token("sk-mcp-key"),
        metadata={"mcp_rpm_limit": {"github": 10}},
        team_metadata={"mcp_rpm_limit": {"github": 99}},
    )
    assert get_key_mcp_rpm_limit(key_first) == {"github": 10}

    # Falls back to team metadata when key has none.
    team_only = UserAPIKeyAuth(
        api_key=hash_token("sk-mcp-key"),
        team_metadata={"mcp_rpm_limit": {"github": 7}},
    )
    assert get_key_mcp_rpm_limit(team_only) == {"github": 7}
    assert get_team_mcp_rpm_limit(team_only) == {"github": 7}

    # No configuration anywhere.
    none_set = UserAPIKeyAuth(api_key=hash_token("sk-mcp-key"))
    assert get_key_mcp_rpm_limit(none_set) is None
    assert get_team_mcp_rpm_limit(none_set) is None


_TEST_SLOT_ID = "slot-disconnect-test"


async def _seed_max_parallel_requests_slots(
    dual_cache: DualCache, counter_key: str, slot_ids: List[str]
) -> None:
    await dual_cache.async_set_cache(
        key=counter_key,
        value={slot_id: time.time() for slot_id in slot_ids},
        local_only=True,
    )


async def _build_seeded_limiter():
    """Build a v3 limiter whose api-key slot registry already holds the pre-call slot."""
    api_key = hash_token("sk-disconnect")
    cache = DualCache()
    limiter = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(cache)
    )
    counter_key = f"{{api_key:{api_key}}}:max_parallel_requests"
    await _seed_max_parallel_requests_slots(cache, counter_key, [_TEST_SLOT_ID])
    user_api_key_dict = UserAPIKeyAuth(api_key=api_key, max_parallel_requests=2)
    return limiter, cache, counter_key, user_api_key_dict


@contextmanager
def _override_litellm_callbacks(new_callbacks):
    """Swap litellm.callbacks so _callback_capabilities recomputes deterministically."""
    saved = litellm.callbacks
    litellm.callbacks = new_callbacks
    try:
        yield
    finally:
        litellm.callbacks = saved


async def _drain_release_task():
    # The disconnect release is scheduled fire-and-forget via create_task.
    for _ in range(5):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_release_max_parallel_requests_on_disconnect_v3():
    """
    Regression for issue #27955: a stream cancelled mid-flight must release the
    pre-call +1 reservation. The success/failure logging callbacks never fire
    on cancellation, so without an explicit release the api-key counter climbs
    by one per cancelled request until the key wedges at its limit. The release
    must decrement the api-key max_parallel_requests counter by exactly one.
    """
    _api_key = hash_token("sk-12345")
    local_cache = DualCache()
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key, max_parallel_requests=2)
    counter_key = f"{{api_key:{_api_key}}}:max_parallel_requests"

    await _seed_max_parallel_requests_slots(local_cache, counter_key, [_TEST_SLOT_ID])
    assert handler._gauge_in_flight_from_cache_value(
        await local_cache.async_get_cache(key=counter_key)
    ) == 1

    await handler.async_release_max_parallel_requests_on_disconnect(
        user_api_key_dict,
        request_data={
            "metadata": {
                MAX_PARALLEL_SLOT_ACQUIRED_KEY: {
                    "slot_id": _TEST_SLOT_ID,
                    "counter_keys": [counter_key],
                }
            }
        },
    )

    assert handler._gauge_in_flight_from_cache_value(
        await local_cache.async_get_cache(key=counter_key)
    ) == 0


@pytest.mark.asyncio
async def test_release_on_disconnect_works_when_key_config_changed_v3():
    """
    The disconnect release must be driven by the stashed acquisition, not the
    key object's current max_parallel_requests configuration: if the limit is
    cleared on the key while a request is in flight, the acquired slot still
    has to be released or it lingers until TTL pruning.
    """
    _api_key = hash_token("sk-12345")
    local_cache = DualCache()
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )
    counter_key = f"{{api_key:{_api_key}}}:max_parallel_requests"
    await _seed_max_parallel_requests_slots(local_cache, counter_key, [_TEST_SLOT_ID])

    await handler.async_release_max_parallel_requests_on_disconnect(
        UserAPIKeyAuth(api_key=_api_key, max_parallel_requests=None),
        request_data={
            "metadata": {
                MAX_PARALLEL_SLOT_ACQUIRED_KEY: {
                    "slot_id": _TEST_SLOT_ID,
                    "counter_keys": [counter_key],
                }
            }
        },
    )
    assert handler._gauge_in_flight_from_cache_value(
        await local_cache.async_get_cache(key=counter_key)
    ) == 0


@pytest.mark.asyncio
async def test_in_memory_fallback_respects_mirrored_redis_count_v3():
    """
    When Redis scripting fails after having worked, the local cache holds the
    integer in-flight count mirrored from the last successful script call.
    The in-memory fallback must treat that count as real occupancy (and
    release must decrement it, floored at 0), not start over from an empty
    registry, which would double the admitted concurrency during a Redis
    outage.
    """
    _api_key = hash_token("sk-12345")
    local_cache = DualCache()
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key, max_parallel_requests=5)
    counter_key = f"{{api_key:{_api_key}}}:max_parallel_requests"

    async def failing_script(keys, args):
        raise ConnectionError("redis unavailable")

    handler.parallel_acquire_script = failing_script
    handler.parallel_release_script = failing_script

    await local_cache.async_set_cache(key=counter_key, value=5, local_only=True)
    with pytest.raises(HTTPException) as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={"model": "gpt-3.5-turbo"},
            call_type="",
        )
    assert exc_info.value.status_code == 429

    await local_cache.async_set_cache(key=counter_key, value=4, local_only=True)
    admitted_data: Dict[str, Any] = {"model": "gpt-3.5-turbo"}
    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data=admitted_data,
        call_type="",
    )
    assert await local_cache.async_get_cache(key=counter_key) == 5

    await handler.async_log_failure_event(
        kwargs={
            "metadata": admitted_data["metadata"],
            "standard_logging_object": {"metadata": {"user_api_key_hash": _api_key}},
        },
        response_obj=None,
        start_time=None,
        end_time=None,
    )
    assert await local_cache.async_get_cache(key=counter_key) == 4


@pytest.mark.asyncio
async def test_release_max_parallel_requests_on_disconnect_noop_v3():
    """
    The release must be a no-op when the key never reserved a parallel slot
    (no api_key, or max_parallel_requests unset). Otherwise a cancelled
    no-limit request would drive an unrelated counter negative.
    """
    _api_key = hash_token("sk-12345")
    local_cache = DualCache()
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )
    counter_key = f"{{api_key:{_api_key}}}:max_parallel_requests"

    await handler.async_release_max_parallel_requests_on_disconnect(
        UserAPIKeyAuth(api_key=_api_key, max_parallel_requests=None)
    )
    assert await local_cache.async_get_cache(key=counter_key) is None

    await handler.async_release_max_parallel_requests_on_disconnect(
        UserAPIKeyAuth(api_key=None, max_parallel_requests=5)
    )
    assert await local_cache.async_get_cache(key=counter_key) is None


@pytest.mark.parametrize("disconnect", ["cancel", "aclose"])
@pytest.mark.asyncio
async def test_async_streaming_data_generator_releases_counter_on_disconnect_v3(
    disconnect,
):
    """
    Regression for issue #27955 on the outer SSE generator (used by /v1/messages
    and other event-stream routes). A client that disconnects mid-stream raises
    GeneratorExit (aclose) or CancelledError into async_streaming_data_generator;
    both are BaseException and bypass the success/failure logging callbacks, so
    the generator itself must refund the pre-call max_parallel_requests +1.
    Releasing inside the nested iterator hook does not work because that
    generator is only closed on garbage collection, which is non-deterministic.
    """
    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

    limiter, cache, counter_key, user_api_key_dict = await _build_seeded_limiter()
    assert limiter._gauge_in_flight_from_cache_value(
        await cache.async_get_cache(key=counter_key)
    ) == 1

    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    proxy_logging_obj.proxy_hook_mapping["parallel_request_limiter"] = limiter

    async def upstream():
        yield ModelResponse()
        if disconnect == "cancel":
            raise asyncio.CancelledError()
        while True:
            yield ModelResponse()

    with _override_litellm_callbacks([]):
        gen = ProxyBaseLLMRequestProcessing.async_sse_data_generator(
            response=upstream(),
            user_api_key_dict=user_api_key_dict,
            request_data={
                "model": "claude-test",
                "metadata": {
                    MAX_PARALLEL_SLOT_ACQUIRED_KEY: {
                        "slot_id": _TEST_SLOT_ID,
                        "counter_keys": [counter_key],
                    }
                },
            },
            proxy_logging_obj=proxy_logging_obj,
        )
        await gen.__anext__()
        if disconnect == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await gen.__anext__()
        else:
            await gen.aclose()
        await _drain_release_task()

    assert limiter._gauge_in_flight_from_cache_value(
        await cache.async_get_cache(key=counter_key)
    ) == 0


@pytest.mark.parametrize("disconnect", ["cancel", "aclose"])
@pytest.mark.asyncio
async def test_async_data_generator_releases_counter_on_disconnect_v3(disconnect):
    """
    Regression for issue #27955 on the chat-completions outer generator
    (proxy_server.async_data_generator). With only the v3 parallel limiter
    enabled, needs_iterator_wrap() is False, so this generator iterates the
    upstream response directly and the iterator hook is bypassed entirely -- the
    gap that let a disconnect leak the slot in the default limiter-only config.
    A mid-stream disconnect must still refund the pre-call +1.
    """
    import litellm.proxy.proxy_server as proxy_server

    limiter, cache, counter_key, user_api_key_dict = await _build_seeded_limiter()
    proxy_logging_obj = proxy_server.proxy_logging_obj
    saved_hook = proxy_logging_obj.proxy_hook_mapping.get("parallel_request_limiter")
    proxy_logging_obj.proxy_hook_mapping["parallel_request_limiter"] = limiter

    async def upstream():
        yield ModelResponse()
        if disconnect == "cancel":
            raise asyncio.CancelledError()
        while True:
            yield ModelResponse()

    try:
        with _override_litellm_callbacks([]):
            assert proxy_logging_obj.needs_iterator_wrap() is False
            gen = proxy_server.async_data_generator(
                response=upstream(),
                user_api_key_dict=user_api_key_dict,
                request_data={
                    "model": "gpt-test",
                    "metadata": {
                    MAX_PARALLEL_SLOT_ACQUIRED_KEY: {
                        "slot_id": _TEST_SLOT_ID,
                        "counter_keys": [counter_key],
                    }
                },
                },
            )
            await gen.__anext__()
            if disconnect == "cancel":
                with pytest.raises(asyncio.CancelledError):
                    await gen.__anext__()
            else:
                await gen.aclose()
            await _drain_release_task()
        assert limiter._gauge_in_flight_from_cache_value(
            await cache.async_get_cache(key=counter_key)
        ) == 0
    finally:
        if saved_hook is not None:
            proxy_logging_obj.proxy_hook_mapping["parallel_request_limiter"] = (
                saved_hook
            )
        else:
            proxy_logging_obj.proxy_hook_mapping.pop("parallel_request_limiter", None)


@pytest.mark.asyncio
async def test_async_data_generator_releases_counter_when_wrapped_v3():
    """
    Companion to the no-wrap case for issue #27955. With an iterator-override
    callback active, needs_iterator_wrap() is True and async_data_generator
    drives the chained iterator hook. The refund must still fire exactly once
    from the outer generator: the counter returns to 0 (not -1), proving the
    nested hook does not also refund and there is no double decrement.
    """
    from litellm.integrations.custom_logger import CustomLogger
    import litellm.proxy.proxy_server as proxy_server

    class _PassthroughIteratorOverride(CustomLogger):
        async def async_post_call_streaming_iterator_hook(
            self, user_api_key_dict, response, request_data
        ):
            async for chunk in response:
                yield chunk

    limiter, cache, counter_key, user_api_key_dict = await _build_seeded_limiter()
    proxy_logging_obj = proxy_server.proxy_logging_obj
    saved_hook = proxy_logging_obj.proxy_hook_mapping.get("parallel_request_limiter")
    proxy_logging_obj.proxy_hook_mapping["parallel_request_limiter"] = limiter

    async def upstream():
        while True:
            yield ModelResponse()

    try:
        with _override_litellm_callbacks([_PassthroughIteratorOverride()]):
            assert proxy_logging_obj.needs_iterator_wrap() is True
            gen = proxy_server.async_data_generator(
                response=upstream(),
                user_api_key_dict=user_api_key_dict,
                request_data={
                    "model": "gpt-test",
                    "metadata": {
                    MAX_PARALLEL_SLOT_ACQUIRED_KEY: {
                        "slot_id": _TEST_SLOT_ID,
                        "counter_keys": [counter_key],
                    }
                },
                },
            )
            await gen.__anext__()
            await gen.aclose()
            await _drain_release_task()
        assert limiter._gauge_in_flight_from_cache_value(
            await cache.async_get_cache(key=counter_key)
        ) == 0
    finally:
        if saved_hook is not None:
            proxy_logging_obj.proxy_hook_mapping["parallel_request_limiter"] = (
                saved_hook
            )
        else:
            proxy_logging_obj.proxy_hook_mapping.pop("parallel_request_limiter", None)


def test_tpm_reservation_enabled_by_default(monkeypatch):
    """Upfront TPM reservation is on unless explicitly disabled via env."""
    monkeypatch.delenv("LITELLM_TPM_TOKEN_RESERVATION_ENABLED", raising=False)
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache())
    )
    assert handler.tpm_reservation_enabled is True


@pytest.mark.parametrize("value", ["false", "False", "FALSE"])
def test_tpm_reservation_disabled_via_env(monkeypatch, value):
    monkeypatch.setenv("LITELLM_TPM_TOKEN_RESERVATION_ENABLED", value)
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache())
    )
    assert handler.tpm_reservation_enabled is False


@pytest.mark.asyncio
async def test_pre_call_hook_reserves_tpm_when_enabled(monkeypatch):
    """
    With reservation enabled, the pre-call hook reserves the estimated token
    budget upfront and tells should_rate_limit to skip the :tokens counter so
    only the reservation path owns it.
    """
    monkeypatch.delenv("LITELLM_TPM_TOKEN_RESERVATION_ENABLED", raising=False)
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache())
    )

    user_api_key_dict = UserAPIKeyAuth(api_key=hash_token("sk-tpm"), tpm_limit=10_000)

    should_rate_limit_calls: List[Dict[str, Any]] = []
    original_should_rate_limit = handler.should_rate_limit

    async def spy_should_rate_limit(*args, **kwargs):
        should_rate_limit_calls.append(kwargs)
        return await original_should_rate_limit(*args, **kwargs)

    reserve_calls: List[int] = []
    original_reserve = handler.reserve_tpm_tokens

    async def spy_reserve(*args, **kwargs):
        reserve_calls.append(kwargs.get("estimated_tokens"))
        return await original_reserve(*args, **kwargs)

    monkeypatch.setattr(handler, "should_rate_limit", spy_should_rate_limit)
    monkeypatch.setattr(handler, "reserve_tpm_tokens", spy_reserve)

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=handler.internal_usage_cache.dual_cache,
        data={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
        call_type="completion",
    )

    assert len(reserve_calls) == 1, "reservation must run when enabled"
    assert should_rate_limit_calls[0]["skip_tpm_check"] is True


@pytest.mark.asyncio
async def test_pre_call_hook_skips_reservation_when_disabled(monkeypatch):
    """
    With reservation disabled, the pre-call hook never calls reserve_tpm_tokens
    and enforces TPM directly in should_rate_limit (skip_tpm_check=False), the
    pre-v1.82 post-call accounting behavior.
    """
    monkeypatch.setenv("LITELLM_TPM_TOKEN_RESERVATION_ENABLED", "false")
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache())
    )

    user_api_key_dict = UserAPIKeyAuth(api_key=hash_token("sk-tpm"), tpm_limit=10_000)

    should_rate_limit_calls: List[Dict[str, Any]] = []
    original_should_rate_limit = handler.should_rate_limit

    async def spy_should_rate_limit(*args, **kwargs):
        should_rate_limit_calls.append(kwargs)
        return await original_should_rate_limit(*args, **kwargs)

    reserve_calls: List[Any] = []

    async def spy_reserve(*args, **kwargs):
        reserve_calls.append(kwargs)
        raise AssertionError("reserve_tpm_tokens must not run when disabled")

    monkeypatch.setattr(handler, "should_rate_limit", spy_should_rate_limit)
    monkeypatch.setattr(handler, "reserve_tpm_tokens", spy_reserve)

    data = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=handler.internal_usage_cache.dual_cache,
        data=data,
        call_type="completion",
    )

    assert reserve_calls == [], "reservation must be skipped when disabled"
    assert should_rate_limit_calls[0]["skip_tpm_check"] is False
    # No reservation stash leaks into the request metadata.
    from litellm.proxy.hooks.parallel_request_limiter_v3 import (
        TPM_RESERVED_TOKENS_KEY,
    )

    assert TPM_RESERVED_TOKENS_KEY not in (data.get("metadata") or {})


@pytest.mark.asyncio
async def test_per_tag_rate_limit_independent_counters_v3(monkeypatch):
    """
    A single key with per-tag RPM limits tracks each tag independently: a tag
    at its limit returns 429 while a different (unlimited) tag keeps flowing,
    governed only by the generous key-level limit.
    """
    monkeypatch.setenv("LITELLM_RATE_LIMIT_WINDOW_SIZE", "60")
    _api_key = hash_token("sk-per-tag-rpm")
    user_api_key_dict = UserAPIKeyAuth(
        api_key=_api_key,
        rpm_limit=100,
        metadata={"tag_rpm_limit": {"cell-1": 2}},
    )
    local_cache = DualCache()
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )

    async def call(tag: str) -> None:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={"model": "gpt-3.5-turbo", "metadata": {"tags": [tag]}},
            call_type="",
        )

    await call("cell-1")
    await call("cell-1")
    with pytest.raises(HTTPException) as exc_info:
        await call("cell-1")
    assert exc_info.value.status_code == 429
    assert "tag_per_key" in str(exc_info.value.detail)

    # cell-2 has no configured tag limit, so cell-1's exhausted counter must
    # not block it; only the generous key-level limit applies.
    for _ in range(5):
        await call("cell-2")


@pytest.mark.asyncio
async def test_per_tag_descriptor_creation_v3():
    """
    _create_rate_limit_descriptors emits a tag_per_key descriptor carrying the
    configured RPM limit only for request tags present in the configured map.
    """
    _api_key = hash_token("sk-per-tag-desc")
    user_api_key_dict = UserAPIKeyAuth(
        api_key=_api_key,
        metadata={"tag_rpm_limit": {"cell-1": 5}},
    )
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache())
    )

    descriptors = handler._create_rate_limit_descriptors(
        user_api_key_dict=user_api_key_dict,
        data={"model": "gpt-3.5-turbo", "metadata": {"tags": ["cell-1", "cell-2"]}},
        rpm_limit_type=None,
        tpm_limit_type=None,
        model_has_failures=False,
    )

    tag_descriptors = [d for d in descriptors if d["key"] == "tag_per_key"]
    assert len(tag_descriptors) == 1, "only the configured tag yields a descriptor"
    descriptor = tag_descriptors[0]
    assert descriptor["value"] == f"{_api_key}:cell-1"
    assert descriptor["rate_limit"]["requests_per_unit"] == 5


@pytest.mark.asyncio
async def test_per_tag_descriptor_absent_without_config_v3():
    """No tag_per_key descriptor is created when the key has no tag limits."""
    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-no-tag"),
        rpm_limit=10,
    )
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache())
    )

    descriptors = handler._create_rate_limit_descriptors(
        user_api_key_dict=user_api_key_dict,
        data={"model": "gpt-3.5-turbo", "metadata": {"tags": ["cell-1"]}},
        rpm_limit_type=None,
        tpm_limit_type=None,
        model_has_failures=False,
    )

    assert not [d for d in descriptors if d["key"] == "tag_per_key"]


@pytest.mark.asyncio
async def test_per_tag_untagged_request_governed_by_key_limit_v3(monkeypatch):
    """
    Per-tag limits are opt-in sub-limits under the key-level ceiling, not a
    standalone enforcement boundary: a request that carries no tag (or a tag
    without a configured limit) is not rejected by any tag counter, but it is
    still bounded by the key-level rpm_limit. This pins the documented
    untagged-fallback behavior so a future "fail closed on missing tag" change
    would fail here instead of silently breaking it.
    """
    monkeypatch.setenv("LITELLM_RATE_LIMIT_WINDOW_SIZE", "60")
    _api_key = hash_token("sk-untagged-fallback")
    user_api_key_dict = UserAPIKeyAuth(
        api_key=_api_key,
        rpm_limit=3,
        metadata={"tag_rpm_limit": {"cell-1": 2}},
    )
    local_cache = DualCache()
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )

    async def call(metadata: dict) -> None:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={"model": "gpt-3.5-turbo", "metadata": metadata},
            call_type="",
        )

    # Untagged and unconfigured-tag requests share the key-level budget of 3
    # and never hit a tag_per_key counter.
    await call({})
    await call({"tags": ["cell-99"]})
    await call({})
    with pytest.raises(HTTPException) as exc_info:
        await call({"tags": ["cell-99"]})
    assert exc_info.value.status_code == 429
    assert "tag_per_key" not in str(exc_info.value.detail)


# --------------------------------------------------------------------------
# Streaming success logging mirrors x-ratelimit-* remaining values into
# standard_logging_object.hidden_params.additional_headers so Prometheus /
# logging callbacks see them for streams too (non-streaming already gets
# them via async_post_call_success_hook, which the streaming path skips).
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_end_to_end_populates_slp_ratelimit_headers(monkeypatch):
    """
    End-to-end regression: on a streaming request, the same pre-call +
    success-callback pair the proxy uses must land ``x-ratelimit-*``
    remaining/limit values in
    ``kwargs["standard_logging_object"]["hidden_params"]["additional_headers"]``.
    """
    monkeypatch.setenv("LITELLM_RATE_LIMIT_WINDOW_SIZE", "60")
    _api_key = hash_token("sk-stream-e2e")
    user_api_key_dict = UserAPIKeyAuth(
        api_key=_api_key,
        rpm_limit=100,
        tpm_limit=10000,
    )
    local_cache = DualCache()
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )

    # Real pre-call: populates data and stashes the response into metadata
    # so the success callback can find it via litellm_params.metadata.
    data: Dict[str, Any] = {
        "model": "gpt-4o-mini",
        "metadata": {},
        "stream": True,
    }
    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data=data,
        call_type="",
    )

    # Simulate the wrapper handing the pre-call metadata dict to the
    # completion() call: it becomes kwargs["litellm_params"]["metadata"] by
    # the time the success callback fires.
    mock_response = ModelResponse(
        id="mock-stream-e2e",
        object="chat.completion",
        created=int(datetime.now().timestamp()),
        model="gpt-4o-mini",
        usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        choices=[],
    )
    mock_kwargs: Dict[str, Any] = {
        "standard_logging_object": {
            "metadata": {
                "user_api_key_hash": _api_key,
                "user_api_key_user_id": None,
                "user_api_key_team_id": None,
                "user_api_key_end_user_id": None,
            }
        },
        "litellm_params": {"metadata": data["metadata"]},
        "model": "gpt-4o-mini",
    }

    async def _noop_increment(increment_list, **_):
        return True

    monkeypatch.setattr(
        handler.internal_usage_cache.dual_cache,
        "async_increment_cache_pipeline",
        _noop_increment,
    )

    # async_logging_hook runs before async_log_success_event, so any
    # downstream callback that reads the SLP sees the mirrored values.
    await handler.async_logging_hook(
        kwargs=mock_kwargs,
        result=mock_response,
        call_type="acompletion",
    )

    additional_headers = (
        mock_kwargs["standard_logging_object"]
        .get("hidden_params", {})
        .get("additional_headers", {})
    )

    # api_key-scoped remaining/limit values are the baseline every request
    # emits and must always reach the SLP.
    remaining_keys = [
        k for k in additional_headers if "-remaining-" in k
    ]
    assert (
        remaining_keys
    ), f"streaming success must populate remaining values, got {additional_headers!r}"
    limit_keys = [k for k in additional_headers if "-limit-" in k]
    assert limit_keys, "streaming success must also populate limit values"
    assert (
        additional_headers.get("x-ratelimit-api_key-remaining-requests") == 99
    ), (
        "api_key remaining requests should reflect the just-consumed slot;"
        f" got {additional_headers!r}"
    )


@pytest.mark.asyncio
async def test_streaming_populates_model_per_key_ratelimit_headers(monkeypatch):
    """
    Streaming must land the per-(key, model) remaining/limit values in the
    SLP under ``x-ratelimit-model_per_key-{remaining|limit}-{requests,tokens}``.
    """
    monkeypatch.setenv("LITELLM_RATE_LIMIT_WINDOW_SIZE", "60")
    _api_key = hash_token("sk-stream-mirror")
    user_api_key_dict = UserAPIKeyAuth(
        api_key=_api_key,
        metadata={
            "model_rpm_limit": {"gpt-4o-mini": 100},
            "model_tpm_limit": {"gpt-4o-mini": 10000},
        },
    )
    local_cache = DualCache()
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )

    async def _noop_increment(increment_list, **_):
        return True

    monkeypatch.setattr(
        handler.internal_usage_cache.dual_cache,
        "async_increment_cache_pipeline",
        _noop_increment,
    )

    data: Dict[str, Any] = {"model": "gpt-4o-mini", "metadata": {}, "stream": True}
    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data=data,
        call_type="",
    )

    mock_response = ModelResponse(
        id="mock-stream",
        object="chat.completion",
        created=int(datetime.now().timestamp()),
        model="gpt-4o-mini",
        usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        choices=[],
    )

    mock_kwargs: Dict[str, Any] = {
        "standard_logging_object": {
            "metadata": {
                "user_api_key_hash": _api_key,
                "user_api_key_user_id": None,
                "user_api_key_team_id": None,
                "user_api_key_end_user_id": None,
            }
        },
        "litellm_params": {"metadata": data["metadata"]},
        "model": "gpt-4o-mini",
    }

    await handler.async_logging_hook(
        kwargs=mock_kwargs,
        result=mock_response,
        call_type="acompletion",
    )

    hidden_params = mock_kwargs["standard_logging_object"].get("hidden_params") or {}
    additional_headers = hidden_params.get("additional_headers") or {}

    assert (
        additional_headers.get("x-ratelimit-model_per_key-remaining-requests") == 99
    ), f"got {additional_headers!r}"
    assert additional_headers.get("x-ratelimit-model_per_key-limit-requests") == 100

    # response._hidden_params is also updated for late readers.
    response_hidden = getattr(mock_response, "_hidden_params", None) or {}
    response_headers = response_hidden.get("additional_headers") or {}
    assert response_headers.get("x-ratelimit-model_per_key-remaining-requests") == 99


@pytest.mark.asyncio
async def test_async_log_success_event_no_mirror_when_no_snapshot(monkeypatch):
    """
    No pre-call snapshot (no descriptors matched) -> no fabricated
    ``x-ratelimit-*`` headers.
    """
    monkeypatch.setenv("LITELLM_RATE_LIMIT_WINDOW_SIZE", "60")
    _api_key = hash_token("sk-stream-no-mirror")
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache())
    )

    async def _noop_increment(increment_list, **_):
        return True

    monkeypatch.setattr(
        handler.internal_usage_cache.dual_cache,
        "async_increment_cache_pipeline",
        _noop_increment,
    )

    mock_response = ModelResponse(
        id="mock-stream-none",
        object="chat.completion",
        created=int(datetime.now().timestamp()),
        model="gpt-4o-mini",
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        choices=[],
    )

    mock_kwargs: Dict[str, Any] = {
        "standard_logging_object": {
            "metadata": {
                "user_api_key_hash": _api_key,
                "user_api_key_user_id": None,
                "user_api_key_team_id": None,
                "user_api_key_end_user_id": None,
            }
        },
        "litellm_params": {"metadata": {}},
        "model": "gpt-4o-mini",
    }

    await handler.async_logging_hook(
        kwargs=mock_kwargs,
        result=mock_response,
        call_type="acompletion",
    )

    hidden_params = mock_kwargs["standard_logging_object"].get("hidden_params") or {}
    additional_headers = hidden_params.get("additional_headers") or {}
    ratelimit_keys = [k for k in additional_headers if k.startswith("x-ratelimit-")]
    assert (
        not ratelimit_keys
    ), f"no snapshot must produce no rate-limit headers, got {ratelimit_keys}"


@pytest.mark.asyncio
async def test_streaming_mirror_matches_non_streaming_header_shape(monkeypatch):
    """
    Given the same pre-call state, streaming and non-streaming must write
    the identical ``x-ratelimit-*`` key/value shape to their respective
    ``additional_headers`` slots.
    """
    monkeypatch.setenv("LITELLM_RATE_LIMIT_WINDOW_SIZE", "60")
    _api_key = hash_token("sk-shape")
    user_api_key_dict = UserAPIKeyAuth(
        api_key=_api_key,
        metadata={
            "model_rpm_limit": {"gpt-4o-mini": 50},
            "model_tpm_limit": {"gpt-4o-mini": 5000},
        },
    )
    local_cache = DualCache()
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )

    async def _noop_increment(increment_list, **_):
        return True

    monkeypatch.setattr(
        handler.internal_usage_cache.dual_cache,
        "async_increment_cache_pipeline",
        _noop_increment,
    )

    # Drive pre-call once so both paths have the same authoritative snapshot.
    data: Dict[str, Any] = {"model": "gpt-4o-mini", "metadata": {}}
    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data=data,
        call_type="",
    )

    # Non-streaming path: async_post_call_success_hook mutates response._hidden_params.
    non_stream_response = ModelResponse(
        id="mock-non-stream",
        object="chat.completion",
        created=int(datetime.now().timestamp()),
        model="gpt-4o-mini",
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        choices=[],
    )
    non_stream_response._hidden_params = {}
    await handler.async_post_call_success_hook(
        data=data,
        user_api_key_dict=user_api_key_dict,
        response=non_stream_response,
    )
    non_stream_headers = non_stream_response._hidden_params.get(
        "additional_headers", {}
    )

    # Streaming path: async_logging_hook mirrors into standard_logging_object.
    stream_kwargs: Dict[str, Any] = {
        "standard_logging_object": {
            "metadata": {"user_api_key_hash": _api_key}
        },
        "litellm_params": {"metadata": data["metadata"]},
        "model": "gpt-4o-mini",
    }
    stream_response = ModelResponse(
        id="mock-stream",
        object="chat.completion",
        created=int(datetime.now().timestamp()),
        model="gpt-4o-mini",
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        choices=[],
    )
    await handler.async_logging_hook(
        kwargs=stream_kwargs,
        result=stream_response,
        call_type="acompletion",
    )
    stream_slp_headers = (
        stream_kwargs["standard_logging_object"]
        .get("hidden_params", {})
        .get("additional_headers", {})
    )

    def _rl_only(headers: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in headers.items() if k.startswith("x-ratelimit-")}

    assert _rl_only(stream_slp_headers) == _rl_only(non_stream_headers), (
        f"streaming={_rl_only(stream_slp_headers)}"
        f" non_streaming={_rl_only(non_stream_headers)}"
    )
    assert "x-ratelimit-model_per_key-remaining-requests" in stream_slp_headers
