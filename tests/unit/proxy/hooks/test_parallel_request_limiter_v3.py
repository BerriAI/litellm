"""
Unit Tests for the max parallel request limiter v3 for the proxy
"""

import asyncio
import logging
import time
from collections.abc import Iterator, Sequence
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import Any, Final
from unittest.mock import patch

import pytest
from fastapi import HTTPException

import litellm
from litellm import Router
from litellm.caching.caching import DualCache
from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY
from litellm.exceptions import ServiceUnavailableError
from litellm.litellm_core_utils.model_latest_aliases import get_model_latest_aliases
from litellm.litellm_core_utils.ptu_shares import PTUTeamCeiling
from litellm.litellm_core_utils.routing_common_utils import resolve_model_group_alias
from litellm.litellm_core_utils.total_deadline import TotalTimeoutExpired, total_deadline
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    PARALLEL_REQUEST_SLOT_TTL_SECONDS,
    ParallelSlotAcquisition,
    PROXY_MaxParallelRequestsHandler_v3,
    RateLimitDescriptor,
    RateLimitedModel,
    RateLimitResponse,
    ReservationOperation,
    TagRateLimit,
    _request_stash,
    get_or_create_request_stash,
    get_request_stash,
)
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    PROXY_MaxParallelRequestsHandler_v3 as _PROXY_MaxParallelRequestsHandler,
)
from litellm.proxy.spend_tracking.input_tokens import models_counted_for
from litellm.proxy.utils import InternalUsageCache, ProxyLogging, hash_token
from litellm.redis.stores.rate_limits import (
    CounterIncrement,
    CounterKind,
    LocalValue,
    RateLimitCounter,
    RateLimitEntry,
    RateLimitGauge,
    RateLimitScope,
    RateLimitStore,
)
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.mcp import MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer
from litellm.types.utils import (
    CallTypes,
    EmbeddingResponse,
    ModelResponse,
    PromptTokensDetailsWrapper,
    TextCompletionResponse,
    Usage,
)
from tests._support.clock_loops import AutojumpClockLoop
from tests.unit.redis.stores.fake_rate_limits import FakeRateLimits, scope_key


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


def _counter(key: str, value: str, kind: CounterKind = "tokens") -> RateLimitCounter:
    return RateLimitScope(key, value).counter(kind)


def _local(handler: _PROXY_MaxParallelRequestsHandler, entry: RateLimitEntry) -> LocalValue | None:
    return handler.rate_limits.local((entry,))[0]


def _increment_of(operation: ReservationOperation) -> CounterIncrement:
    return operation if isinstance(operation, CounterIncrement) else operation.increment


def _capture_token_increments(fakes: FakeRateLimits) -> list[dict[str, object]]:
    """Answers the TTL-preserving token increment script and records main's ``(key, increment_value, ttl)``."""
    captured: Final[list[dict[str, object]]] = []

    async def token_increment(keys: list[str], args: list[float]) -> list[float]:
        amounts: Final = args[::2]
        captured.extend(
            {"key": key, "increment_value": amount, "ttl": ttl} for key, amount, ttl in zip(keys, amounts, args[1::2])
        )
        return amounts

    fakes.token_increment_script = token_increment
    return captured


@pytest.fixture(autouse=True)
def _isolated_request_stash():
    token = _request_stash.set(None)
    yield
    _request_stash.reset(token)


@pytest.fixture
def reservation() -> bool:
    return True


@pytest.fixture
def collision_limiter(
    reservation: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[DualCache, PROXY_MaxParallelRequestsHandler_v3, str, str]:
    monkeypatch.setenv("LITELLM_TPM_TOKEN_RESERVATION_ENABLED", str(reservation).lower())
    alias: Final = "anthropic/claude-opus-latest"
    target: Final = f"anthropic/{get_model_latest_aliases()['anthropic']['claude-opus-latest']}"
    router: Final = Router(
        model_list=[{"model_name": "anthropic/*", "litellm_params": {"model": "anthropic/*"}}],
        model_group_alias={alias: target},
    )
    cache: Final = DualCache()
    handler: Final = PROXY_MaxParallelRequestsHandler_v3(
        internal_usage_cache=InternalUsageCache(cache),
        model_group_resolver=partial(resolve_model_group_alias, router.registry().aliases()),
        time_provider=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    return cache, handler, alias, target


@pytest.mark.parametrize(
    "throttle_pct, expected_rpm, expected_tpm",
    [
        (None, 100, 1000),  # no throttle -> configured limits
        (0.1, 10, 100),  # 10% of configured
        (0.5, 50, 500),
    ],
)
def test_api_key_descriptor_applies_budget_throttle(throttle_pct, expected_rpm, expected_tpm):
    """The api_key rate-limit descriptor scales the key's configured TPM/RPM by
    the request-scoped budget_throttle_pct, leaving the configured limits intact."""
    handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(DualCache()))
    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-throttle"),
        rpm_limit=100,
        tpm_limit=1000,
        budget_throttle_pct=throttle_pct,
    )

    descriptors = handler.create_rate_limit_descriptors(
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
    _api_key = "sk-98765"
    _api_key = hash_token(_api_key)
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key, rpm_limit=3)
    local_cache = DualCache()
    fakes = FakeRateLimits()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache),
        time_provider=time_controller.now,
        rate_limit_store=fakes.store(),
    )

    # Mock the batch_rate_limiter_script to simulate window expiry and use correct key construction
    window_starts: dict[str, int] = {}
    request_counts: dict[str, int] = {}

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
            else:
                new_counter = prev_counter + 1
                request_counts[counter_key] = new_counter
            results.append(now)
            results.append(new_counter)
        return results

    fakes.batch_rate_limiter_script = mock_batch_rate_limiter

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
    # the native memory tier expires the local mirror on the real clock, not the patched time.time
    await asyncio.sleep(2.1)

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
    _api_key = "sk-98765"
    _api_key = hash_token(_api_key)
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key, rpm_limit=3)
    local_cache = DualCache()
    fakes = FakeRateLimits()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache),
        time_provider=time_controller.now,
        rate_limit_store=fakes.store(),
    )

    # Mock the batch_rate_limiter_script to simulate window expiry and use correct key construction
    window_starts: dict[str, int] = {}
    request_counts: dict[str, int] = {}

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
            else:
                new_counter = prev_counter + 1
                request_counts[counter_key] = new_counter
            results.append(now)
            results.append(new_counter)
        return results

    fakes.batch_rate_limiter_script = mock_batch_rate_limiter

    # Make first request
    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )

    # Verify both counter and window values are stored in cache
    counter_key = _counter("api_key", _api_key, "requests")
    window_key = counter_key.window

    window_value = _local(parallel_request_handler, window_key)
    counter_value = _local(parallel_request_handler, counter_key)

    assert window_value is not None, "Window value should be stored in cache"
    assert counter_value is not None, "Counter value should be stored in cache"
    assert counter_value == 1, "Counter should be 1 after first request"

    # Make second request
    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )

    # Verify counter increased but window stayed same
    new_window_value = _local(parallel_request_handler, window_key)
    new_counter_value = _local(parallel_request_handler, counter_key)

    assert new_window_value == window_value, "Window value should not change within window"
    assert new_counter_value == 2, "Counter should be 2 after second request"

    # Wait for window to expire
    time_controller.advance(3)

    # Make request after window expiry
    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )

    # Verify new window and reset counter
    final_window_value = _local(parallel_request_handler, window_key)
    final_counter_value = _local(parallel_request_handler, counter_key)

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

    _api_key = "sk-98765"
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
    fakes = FakeRateLimits()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache),
        time_provider=time_controller.now,
        rate_limit_store=fakes.store(),
    )

    # Mock the batch_rate_limiter_script to simulate window expiry and use correct key construction
    window_starts: dict[str, int] = {}
    request_counts: dict[str, int] = {}

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
            else:
                new_counter = prev_counter + 1
                request_counts[counter_key] = new_counter
            results.append(now)
            results.append(new_counter)
        return results

    fakes.batch_rate_limiter_script = mock_batch_rate_limiter
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
    counter_key = _counter(rate_limit_object, value)

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
    expected_reservation = parallel_request_handler._estimate_tokens_for_request(data=pre_call_data)
    assert expected_reservation < 10, "Test premise: reservation must fit under tpm_limit=10"

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
    counter_value = _local(parallel_request_handler, counter_key)

    assert counter_value is not None, f"Counter value should be stored in cache for {counter_key}"

    # Manually increment the token counter to simulate token usage from previous call
    # This simulates what would happen after a successful call
    await parallel_request_handler.rate_limits.increment(
        CounterIncrement(counter_key, 15, 2)
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
    final_counter_value = _local(parallel_request_handler, counter_key)

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

    _api_key = "sk-98765"
    _api_key = hash_token(_api_key)
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key, tpm_limit=100)
    local_cache = DualCache()
    fakes = FakeRateLimits()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache),
        rate_limit_store=fakes.store(),
    )

    # Mock the get_rate_limit_type method directly since it imports general_settings internally
    def mock_get_rate_limit_type():
        return token_rate_limit_type

    monkeypatch.setattr(parallel_request_handler, "get_rate_limit_type", mock_get_rate_limit_type)

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
    captured_operations = _capture_token_increments(fakes)

    # Call the success event handler
    await parallel_request_handler.async_log_success_event(
        kwargs=mock_kwargs,
        response_obj=mock_response,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    # Verify that the correct token count was used based on the rate limit type
    assert len(captured_operations) == 1, (
        "Should have 1 operation: the TPM increment (parallel slots are released via the gauge, not the pipeline)"
    )

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

    assert tpm_operation["increment_value"] == expected_tokens[token_rate_limit_type], (
        f"Expected {expected_tokens[token_rate_limit_type]} tokens for type '{token_rate_limit_type}', got {tpm_operation['increment_value']}"
    )


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
async def test_async_log_success_event_counts_non_chat_response_tokens(monkeypatch, response_obj):
    """
    Embedding and text completion responses must increment the TPM counter,
    not just chat completion ModelResponse objects.
    """
    monkeypatch.setenv("LITELLM_RATE_LIMIT_WINDOW_SIZE", "60")

    _api_key = hash_token("sk-98765")
    fakes = FakeRateLimits()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache()),
        rate_limit_store=fakes.store(),
    )
    monkeypatch.setattr(parallel_request_handler, "get_rate_limit_type", lambda: "total")

    mock_kwargs = {
        "standard_logging_object": {"metadata": {"user_api_key_hash": _api_key}},
        "model": response_obj.model,
    }

    captured_operations = _capture_token_increments(fakes)

    await parallel_request_handler.async_log_success_event(
        kwargs=mock_kwargs,
        response_obj=response_obj,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    tpm_operation = next((op for op in captured_operations if op["key"].endswith(":tokens")), None)
    assert tpm_operation is not None, "Should have a TPM increment operation"
    assert tpm_operation["increment_value"] == 50


@pytest.mark.asyncio
async def test_parallel_gauge_uses_atomic_redis_script_v3():
    """
    With Redis available, gauge admission goes through the atomic
    check-and-acquire script (limit, slot TTL, and this request's slot id as
    args), the returned in-flight count is mirrored into the local cache,
    and an over-limit script result maps to a 429 without occupying a slot.
    """
    local_cache = DualCache()
    fakes = FakeRateLimits()
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache),
        rate_limit_store=fakes.store(),
    )
    _api_key = hash_token("sk-98765")
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key, max_parallel_requests=5)
    counter_key = RateLimitScope("api_key", _api_key).gauge

    captured_calls = []

    async def fake_acquire(keys, args):
        captured_calls.append((list(keys), list(args)))
        return [0, 3]

    fakes.parallel_acquire_script = fake_acquire

    data: dict[str, Any] = {"model": "gpt-3.5-turbo"}
    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data=data,
        call_type="",
    )
    stashed_acquisition = get_request_stash().parallel_slot
    assert stashed_acquisition is not None
    stashed_slot_id = stashed_acquisition["slot_id"]
    assert isinstance(stashed_slot_id, str) and stashed_slot_id
    assert stashed_acquisition["gauges"] == (counter_key.scope,)
    assert captured_calls == [
        ([f"{{api_key:{_api_key}}}:max_parallel_requests"], [5, PARALLEL_REQUEST_SLOT_TTL_SECONDS, stashed_slot_id])
    ]
    assert _local(handler, counter_key) == 3
    gauge_statuses = [
        s
        for s in get_request_stash().rate_limit_response["statuses"]
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

    fakes.parallel_acquire_script = fake_acquire_over_limit

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
async def test_async_increment_tokens_fallback_behavior():
    """
    Without Redis there is no TTL-preserving script, so the increment lands in memory.
    """
    fakes = FakeRateLimits(connected=False)
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache()), rate_limit_store=fakes.store()
    )
    counter = _counter("test", "fallback_key")

    await parallel_request_handler.async_increment_tokens_with_ttl_preservation(
        pipeline_operations=[CounterIncrement(counter, 10.0, 60)]
    )

    assert _local(parallel_request_handler, counter) == 10.0
    assert fakes.redis_increments == []


@pytest.mark.asyncio
async def test_async_increment_tokens_falls_back_to_the_pipeline_when_the_script_fails():
    async def failing_script(keys: list[str], args: list[object]) -> object:
        raise ConnectionError(_CONNECTION_REFUSED)

    fakes = FakeRateLimits(token_increment_script=failing_script)
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache()), rate_limit_store=fakes.store()
    )
    counter = _counter("test", "fallback_key")

    await parallel_request_handler.async_increment_tokens_with_ttl_preservation(
        pipeline_operations=[CounterIncrement(counter, 10.0, 60)]
    )

    assert fakes.redis_increments == [("{test:fallback_key}:tokens", 10.0, 60)]
    assert _local(parallel_request_handler, counter) == 10.0


# Redis Cluster Compatibility Tests
@pytest.mark.asyncio
async def test_execute_redis_batch_rate_limiter_script_cluster_compatibility():
    """
    Each slot group runs its own batch script call, a group whose call fails falls back to the
    in-memory window, and the values come back in the caller's counter order.
    """
    from unittest.mock import AsyncMock

    mock_script = AsyncMock(side_effect=[Exception("EVALSHA - all keys must map to the same key slot"), [1234, 2]])
    fakes = FakeRateLimits(batch_rate_limiter_script=mock_script)
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache()),
        force_hash_tag_grouping_resolver=lambda: True,
        rate_limit_store=fakes.store(),
    )

    results = await handler._execute_redis_batch_rate_limiter_script(
        counters=[_counter("api_key", "sk-123", "requests"), _counter("user", "user-456", "requests")],
        now_int=1234,
    )

    assert results == [("1234", 1), ("1234", 2)]
    assert [call.kwargs["keys"] for call in mock_script.call_args_list] == [
        ["{api_key:sk-123}:window", "{api_key:sk-123}:requests"],
        ["{user:user-456}:window", "{user:user-456}:requests"],
    ]


@pytest.mark.asyncio
async def test_missing_descriptor_fallback():
    """
    Test that the fallback works when a descriptor_key cannot be found in the descriptors list.

    This tests an edge case where somehow the descriptor_key in status doesn't match
    any descriptor key (shouldn't happen in normal operation but good for robustness).
    """
    _api_key = "sk-98765"
    _api_key_hash = hash_token(_api_key)

    user_api_key_dict = UserAPIKeyAuth(
        api_key=_api_key_hash,
        rpm_limit=2,
    )

    local_cache = DualCache()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(local_cache))

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
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(local_cache))

    # Mock general_settings to return empty dict (no token_rate_limit_type set)
    from litellm.proxy import proxy_server

    original_settings = getattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(proxy_server, "general_settings", {})

    try:
        result = parallel_request_handler.get_rate_limit_type()
        assert result == "total", f"Default rate limit type should be 'total', got '{result}'"
    finally:
        monkeypatch.setattr(proxy_server, "general_settings", original_settings)


@pytest.mark.asyncio
async def test_get_rate_limit_type_invalid_falls_back_to_total(monkeypatch):
    """
    Test that get_rate_limit_type falls back to 'total' when an invalid value is specified.
    """
    local_cache = DualCache()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(local_cache))

    # Mock general_settings to return an invalid token_rate_limit_type
    from litellm.proxy import proxy_server

    original_settings = getattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(proxy_server, "general_settings", {"token_rate_limit_type": "invalid_type"})

    try:
        result = parallel_request_handler.get_rate_limit_type()
        assert result == "total", f"Invalid rate limit type should fall back to 'total', got '{result}'"
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
async def test_async_log_success_event_with_dict_usage(monkeypatch, token_rate_limit_type, expected_field):
    """
    Test that async_log_success_event correctly handles usage as a dict (Responses API format).

    The Responses API returns usage as a dict in ResponsesAPIResponse instead of a Usage object.
    This test verifies that token counting works correctly with dict-based usage.
    """
    from unittest.mock import MagicMock

    _api_key = "sk-98765"
    _api_key = hash_token(_api_key)
    local_cache = DualCache()
    fakes = FakeRateLimits()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache),
        rate_limit_store=fakes.store(),
    )

    # Mock the get_rate_limit_type method
    def mock_get_rate_limit_type():
        return token_rate_limit_type

    monkeypatch.setattr(parallel_request_handler, "get_rate_limit_type", mock_get_rate_limit_type)

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
    captured_operations = _capture_token_increments(fakes)

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

    assert tpm_operation["increment_value"] == expected_tokens[token_rate_limit_type], (
        f"Expected {expected_tokens[token_rate_limit_type]} tokens for type '{token_rate_limit_type}', got {tpm_operation['increment_value']}"
    )


@pytest.mark.asyncio
async def test_async_log_success_event_with_dict_usage_missing_fields(monkeypatch):
    """
    Test that async_log_success_event handles dict usage with missing fields gracefully.

    When usage dict is missing expected fields, it should default to 0.
    """
    from unittest.mock import MagicMock

    _api_key = "sk-98765"
    _api_key = hash_token(_api_key)
    local_cache = DualCache()
    fakes = FakeRateLimits()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache),
        rate_limit_store=fakes.store(),
    )

    # Mock the get_rate_limit_type method
    def mock_get_rate_limit_type():
        return "output"

    monkeypatch.setattr(parallel_request_handler, "get_rate_limit_type", mock_get_rate_limit_type)

    # Create a mock response object with usage as a dict missing some fields
    mock_response = MagicMock()
    mock_response.usage = {
        "prompt_tokens": 25,
        # completion_tokens is missing
        # total_tokens is missing
    }
    from litellm.types.utils import BaseLiteLLMOpenAIResponseObject

    mock_response.__class__ = type("MockResponse", (BaseLiteLLMOpenAIResponseObject,), {})

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
    captured_operations = _capture_token_increments(fakes)

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
    Token increments run one script call per slot group, each with an (increment, ttl) pair per key.
    """
    from unittest.mock import AsyncMock

    mock_script = AsyncMock(return_value=[0.0, 0.0])
    fakes = FakeRateLimits(token_increment_script=mock_script)
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache()),
        force_hash_tag_grouping_resolver=lambda: True,
        rate_limit_store=fakes.store(),
    )

    await handler._execute_token_increment_script(
        [
            CounterIncrement(_counter("api_key", "sk-123"), 100, 60),
            CounterIncrement(_counter("api_key", "sk-123", "requests"), 1, 60),
            CounterIncrement(_counter("user", "user-456"), 50, 60),
        ]
    )

    assert [(call.kwargs["keys"], call.kwargs["args"]) for call in mock_script.call_args_list] == [
        (["{api_key:sk-123}:tokens", "{api_key:sk-123}:requests"], [100, 60, 1, 60]),
        (["{user:user-456}:tokens"], [50, 60]),
    ]


@pytest.mark.asyncio
async def test_agent_rate_limit_tpm_increment_on_success(monkeypatch):
    """
    Test that async_log_success_event increments agent and session
    TPM counters when agent_id and session_id are in metadata.
    """
    _api_key = "sk-98765"
    _api_key = hash_token(_api_key)
    _agent_id = "agent_tpm_test"
    _session_id = "sess_tpm_test"

    local_cache = DualCache()
    fakes = FakeRateLimits()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache),
        rate_limit_store=fakes.store(),
    )

    def mock_get_rate_limit_type():
        return "total"

    monkeypatch.setattr(parallel_request_handler, "get_rate_limit_type", mock_get_rate_limit_type)

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

    captured_operations = _capture_token_increments(fakes)

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

    from litellm.types.agents import AgentResponse

    monkeypatch.setenv("LITELLM_RATE_LIMIT_WINDOW_SIZE", "2")
    _api_key = "sk-98765"
    _api_key = hash_token(_api_key)
    _agent_id = "agent_429_test"

    user_api_key_dict = UserAPIKeyAuth(
        api_key=_api_key,
        agent_id=_agent_id,
    )

    local_cache = DualCache()
    fakes = FakeRateLimits()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache),
        time_provider=time_controller.now,
        rate_limit_store=fakes.store(),
    )

    mock_agent = AgentResponse(
        agent_id=_agent_id,
        agent_name="rate-limited-agent",
        agent_card_params={"name": "Rate Limited Agent"},
        rpm_limit=2,
    )

    window_starts: dict[str, int] = {}
    request_counts: dict[str, int] = {}

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
            else:
                new_counter = prev_counter + 1
                request_counts[counter_key] = new_counter
            results.append(now)
            results.append(new_counter)
        return results

    fakes.batch_rate_limiter_script = mock_batch_rate_limiter

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
        return PROXY_MaxParallelRequestsHandler_v3(
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


def _responses_websocket_turn_events(total_tokens: int) -> list[dict[str, object]]:
    response: Final = {"id": "resp_ws_turn", "object": "response", "created_at": 1700000000, "model": "gpt-5.4"}
    return [
        {"type": "response.created", "response": {**response, "status": "in_progress", "output": []}},
        {
            "type": "response.completed",
            "response": {
                **response,
                "status": "completed",
                "output": [],
                "usage": {"input_tokens": total_tokens - 10, "output_tokens": 10, "total_tokens": total_tokens},
            },
        },
    ]


# ----------------------- Per-MCP-server rate limiting (v3) -----------------------


def _make_mcp_handler() -> tuple[PROXY_MaxParallelRequestsHandler_v3, DualCache]:
    local_cache: Final = DualCache()
    handler: Final = PROXY_MaxParallelRequestsHandler_v3(internal_usage_cache=InternalUsageCache(local_cache))
    return handler, local_cache


@pytest.mark.asyncio
async def test_inherited_mcp_team_rate_limit_deduplicates_team_and_source_descriptors_v3() -> None:
    handler, _ = _make_mcp_handler()
    server: Final = MCPServer(
        server_id="math-a",
        name="math_a",
        server_name="math_a",
        transport=MCPTransport.http,
    )
    inherited_auth: Final = UserAPIKeyAuth(
        api_key=hash_token("sk-inherited"),
        team_id="team-a",
        team_metadata={"mcp_rpm_limit": {"math_a": 1}},
    ).model_copy(update={"mcp_source_team_rpm_limits": {"team-a": {"math_a": 1}}})
    descriptors: Final[list[RateLimitDescriptor]] = []
    handler._add_mcp_per_key_rate_limit_descriptor(inherited_auth, "math_a", descriptors)

    assert descriptors == []
    team_descriptors: Final[list[RateLimitDescriptor]] = []
    handler._add_mcp_per_team_rate_limit_descriptor(inherited_auth, "math_a", team_descriptors)
    assert [(descriptor["key"], descriptor["value"]) for descriptor in team_descriptors] == [
        ("mcp_per_team", "team-a:math_a")
    ]
    await handler.enforce_mcp_server_rate_limits(inherited_auth, server)

    with pytest.raises(ProxyRateLimitError, match="mcp_per_team"):
        await handler.enforce_mcp_server_rate_limits(inherited_auth, server)


def test_inherited_mcp_team_rate_limit_uses_source_limit_when_primary_metadata_is_missing() -> None:
    handler, _ = _make_mcp_handler()
    auth: Final = UserAPIKeyAuth(
        api_key=hash_token("sk-inherited-source-limit"),
        team_id="team-a",
    ).model_copy(update={"mcp_source_team_rpm_limits": {"team-a": {"math_a": 1}}})
    descriptors: Final[list[RateLimitDescriptor]] = []

    handler._add_mcp_per_team_rate_limit_descriptor(auth, "math_a", descriptors)

    assert descriptors == [
        {
            "key": "mcp_per_team",
            "value": "team-a:math_a",
            "rate_limit": {
                "requests_per_unit": 1,
                "tokens_per_unit": None,
                "window_size": handler.window_size,
            },
        }
    ]


@pytest.mark.asyncio
async def test_billed_personal_key_auth_produces_team_rpm_descriptors(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import AsyncMock, MagicMock

    from litellm.models.budget import LiteLLM_BudgetTable
    from litellm.models.team_membership import LiteLLM_TeamMembership
    from litellm.proxy import proxy_server
    from litellm.proxy._types import LiteLLM_ObjectPermissionTable, LiteLLM_TeamTable
    from litellm.proxy.auth import auth_checks
    from litellm.proxy.mcp_server.auth.user_api_key_auth_mcp import MCPRequestHandler
    from litellm.proxy.mcp_server.mcp_server_manager import global_mcp_server_manager

    server: Final = MCPServer(
        server_id="math_a",
        name="math_a",
        server_name="math_a",
        transport=MCPTransport.http,
    )
    team: Final = LiteLLM_TeamTable(
        team_id="team-a",
        organization_id="org-team-a",
        rpm_limit=7,
        object_permission_id="perm-team-a",
        object_permission=LiteLLM_ObjectPermissionTable(
            object_permission_id="perm-team-a",
            mcp_servers=["math_a"],
            mcp_tool_permissions={"math_a": ["add"]},
        ),
    )
    membership: Final = LiteLLM_TeamMembership(
        user_id="owner",
        team_id="team-a",
        litellm_budget_table=LiteLLM_BudgetTable(rpm_limit=3),
    )
    user_api_key_auth: Final = UserAPIKeyAuth(
        api_key=hash_token("sk-billed-personal"),
        user_id="owner",
    )

    monkeypatch.setattr(litellm, "personal_key_model_access_from_teams", True)
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    monkeypatch.setattr(proxy_server, "should_load_db_object", lambda _: False)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", DualCache())
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", MagicMock())
    monkeypatch.setattr(auth_checks, "personal_key_owner_teams", AsyncMock(return_value=(team,)))
    monkeypatch.setattr(auth_checks, "get_team_object", AsyncMock(return_value=team))
    monkeypatch.setattr(auth_checks, "get_team_membership", AsyncMock(return_value=membership))
    monkeypatch.setattr(auth_checks, "get_org_object", AsyncMock(return_value=None))
    monkeypatch.setitem(global_mcp_server_manager.registry, server.server_id, server)

    billed_auth: Final = await MCPRequestHandler.billing_auth_for_tool_call(user_api_key_auth, "math_a-add")
    handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(DualCache()))

    descriptors: Final = handler.create_rate_limit_descriptors(
        user_api_key_dict=billed_auth,
        data={},
        rpm_limit_type=None,
        tpm_limit_type=None,
        model_has_failures=False,
    )

    assert [
        (descriptor["key"], descriptor["value"], descriptor["rate_limit"]["requests_per_unit"])
        for descriptor in descriptors
        if descriptor["key"] in ("team", "team_member")
    ] == [("team", "team-a", 7), ("team_member", "team-a:owner", 3)]


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


def _seed_max_parallel_requests_slots(
    handler: _PROXY_MaxParallelRequestsHandler, gauge: RateLimitGauge, slot_ids: list[str]
) -> None:
    handler.rate_limits.set_local(
        gauge, {slot_id: time.time() for slot_id in slot_ids}, PARALLEL_REQUEST_SLOT_TTL_SECONDS
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("release_from", ["post_call", "stream_cleanup"])
async def test_concurrent_success_callbacks_release_parallel_slot_once_when_redis_fails(release_from: str) -> None:
    from unittest.mock import AsyncMock

    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

    api_key = hash_token("sk-concurrent-release")
    local_cache = DualCache()
    fakes = FakeRateLimits()
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache),
        rate_limit_store=fakes.store(),
    )
    user_api_key_dict = UserAPIKeyAuth(api_key=api_key, max_parallel_requests=2)
    call_id = "concurrent-release-owner"
    parallel_key = RateLimitScope("api_key", api_key).gauge
    release_started = asyncio.Event()
    allow_redis_failure = asyncio.Event()

    async def failing_release(keys: Sequence[str], args: Sequence[object]) -> list[int]:
        release_started.set()
        await allow_redis_failure.wait()
        raise ConnectionError("redis unavailable")

    release_script = AsyncMock(side_effect=failing_release)
    fakes.parallel_release_script = release_script
    handler.rate_limits.set_local(parallel_key, 2, None)
    stash = get_or_create_request_stash()
    stash.owner_litellm_call_id = call_id
    stash.parallel_slot = ParallelSlotAcquisition(
        slot_id="slot-concurrent-release",
        gauges=(parallel_key.scope,),
    )
    data = {"litellm_call_id": call_id}
    logger: Final = ProxyLogging(user_api_key_cache=local_cache)
    logger.proxy_hook_mapping["parallel_request_limiter"] = handler

    post_call_task = asyncio.create_task(
        ProxyBaseLLMRequestProcessing.finalize_streaming_generator_cleanup(
            request=None,
            request_data=data,
            response=None,
            stream_completed=True,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=logger,
        )
        if release_from == "stream_cleanup"
        else handler.async_post_call_success_hook(
            data=data,
            user_api_key_dict=user_api_key_dict,
            response=ResponsesAPIResponse(
                id="resp_concurrent_release",
                created_at=0,
                model="gpt-4o-mini",
                object="response",
                output=[],
                status="completed",
            ),
        )
    )
    await asyncio.wait_for(release_started.wait(), timeout=5)
    logging_task = asyncio.create_task(
        handler.async_log_success_event(
            kwargs=data,
            response_obj=None,
            start_time=None,
            end_time=None,
        )
    )
    allow_redis_failure.set()
    await asyncio.wait_for(
        asyncio.gather(post_call_task, logging_task),
        timeout=5,
    )

    assert release_script.await_count == 1
    assert _local(handler, parallel_key) == 1
    assert stash.parallel_slot is None


@pytest.mark.asyncio
async def test_read_only_gauge_check_counts_without_acquiring_v3():
    """
    read_only callers (e.g. the context-compaction pre-check) must observe
    the in-flight count via the count script without registering a slot, and
    a count-script failure must degrade to the local mirror instead of
    raising.
    """
    _api_key = hash_token("sk-98765")
    local_cache = DualCache()
    fakes = FakeRateLimits()
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache),
        rate_limit_store=fakes.store(),
    )
    counter_key = RateLimitScope("api_key", _api_key).gauge
    descriptors = [
        {
            "key": "api_key",
            "value": _api_key,
            "rate_limit": {"max_parallel_requests": 5},
        }
    ]

    captured_calls = []

    async def fake_count(keys, args):
        captured_calls.append((list(keys), list(args)))
        return [3]

    fakes.parallel_count_script = fake_count

    response = await handler.should_rate_limit(descriptors=descriptors, read_only=True)
    assert captured_calls == [([f"{{api_key:{_api_key}}}:max_parallel_requests"], [PARALLEL_REQUEST_SLOT_TTL_SECONDS])]
    assert response["overall_code"] == "OK"
    assert response["statuses"] == [
        {
            "code": "OK",
            "current_limit": 5,
            "limit_remaining": 2,
            "rate_limit_type": "max_parallel_requests",
            "descriptor_key": "api_key",
        }
    ]
    assert _local(handler, counter_key) is None

    async def failing_count(keys, args):
        raise ConnectionError("redis unavailable")

    fakes.parallel_count_script = failing_count
    _seed_max_parallel_requests_slots(handler, counter_key, ["s1", "s2", "s3", "s4", "s5"])
    response = await handler.should_rate_limit(descriptors=descriptors, read_only=True)
    assert response["overall_code"] == "OVER_LIMIT"
    assert response["statuses"][0]["rate_limit_type"] == "max_parallel_requests"


@pytest.mark.asyncio
async def test_redis_release_script_updates_local_mirror_v3():
    """
    With Redis available, releases go through the release script with this
    request's slot id per gauge key, and the returned in-flight counts are
    mirrored into the local cache so the local first-pass check stays fresh.
    """
    _api_key = hash_token("sk-98765")
    local_cache = DualCache()
    fakes = FakeRateLimits()
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache),
        rate_limit_store=fakes.store(),
    )
    counter_key = RateLimitScope("api_key", _api_key).gauge

    captured_calls = []

    async def fake_release(keys, args):
        captured_calls.append((list(keys), list(args)))
        return [2]

    fakes.parallel_release_script = fake_release

    get_or_create_request_stash().parallel_slot = ParallelSlotAcquisition(
        slot_id="slot-redis-test",
        gauges=(counter_key.scope,),
    )
    await handler.async_log_failure_event(
        kwargs={
            "standard_logging_object": {"metadata": {"user_api_key_hash": _api_key}},
        },
        response_obj=None,
        start_time=None,
        end_time=None,
    )
    assert captured_calls == [([f"{{api_key:{_api_key}}}:max_parallel_requests"], ["slot-redis-test"])]
    assert _local(handler, counter_key) == 2


@pytest.mark.asyncio
async def test_tpm_over_limit_rejection_releases_parallel_slot_v3(monkeypatch):
    """
    When the TPM reservation phase rejects a request AFTER the gauge slot was
    acquired earlier in the same pre-call hook, the slot must be released
    before the 429 is raised; otherwise every TPM rejection would leak a
    slot until TTL pruning.
    """
    monkeypatch.delenv("LITELLM_TPM_TOKEN_RESERVATION_ENABLED", raising=False)
    _api_key = hash_token("sk-98765")
    local_cache = DualCache()
    handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(local_cache))
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key, max_parallel_requests=5, tpm_limit=100)
    counter_key = RateLimitScope("api_key", _api_key).gauge

    async def over_limit_reservation(descriptors, estimated_tokens, parent_otel_span=None, **_kwargs):
        return {
            "overall_code": "OVER_LIMIT",
            "statuses": [
                {
                    "code": "OVER_LIMIT",
                    "current_limit": 100,
                    "limit_remaining": 0,
                    "rate_limit_type": "tokens",
                    "descriptor_key": "api_key",
                }
            ],
        }

    monkeypatch.setattr(handler, "reserve_tpm_tokens", over_limit_reservation)

    with pytest.raises(HTTPException) as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "hi"}]},
            call_type="",
        )
    assert exc_info.value.status_code == 429
    assert handler._gauge_in_flight_from_cache_value(_local(handler, counter_key)) == 0


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
    _api_key = hash_token("sk-98765")
    local_cache = DualCache()
    fakes = FakeRateLimits()
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache),
        rate_limit_store=fakes.store(),
    )
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key, max_parallel_requests=5)
    counter_key = RateLimitScope("api_key", _api_key).gauge

    async def failing_script(keys, args):
        raise ConnectionError("redis unavailable")

    fakes.parallel_acquire_script = failing_script
    fakes.parallel_release_script = failing_script

    handler.rate_limits.set_local(counter_key, 5, None)
    with pytest.raises(HTTPException) as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={"model": "gpt-3.5-turbo"},
            call_type="",
        )
    assert exc_info.value.status_code == 429

    handler.rate_limits.set_local(counter_key, 4, None)
    admitted_data: dict[str, Any] = {"model": "gpt-3.5-turbo"}
    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data=admitted_data,
        call_type="",
    )
    assert _local(handler, counter_key) == 5

    await handler.async_log_failure_event(
        kwargs={
            "standard_logging_object": {"metadata": {"user_api_key_hash": _api_key}},
        },
        response_obj=None,
        start_time=None,
        end_time=None,
    )
    assert _local(handler, counter_key) == 4


def test_tpm_reservation_enabled_by_default(monkeypatch):
    """Upfront TPM reservation is on unless explicitly disabled via env."""
    monkeypatch.delenv("LITELLM_TPM_TOKEN_RESERVATION_ENABLED", raising=False)
    handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(DualCache()))
    assert handler.tpm_reservation_enabled is True


@pytest.mark.parametrize("value", ["false", "False", "FALSE"])
def test_tpm_reservation_disabled_via_env(monkeypatch, value):
    monkeypatch.setenv("LITELLM_TPM_TOKEN_RESERVATION_ENABLED", value)
    handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(DualCache()))
    assert handler.tpm_reservation_enabled is False


@pytest.mark.asyncio
async def test_pre_call_hook_reserves_tpm_when_enabled(monkeypatch):
    """
    With reservation enabled, the pre-call hook reserves the estimated token
    budget upfront and tells should_rate_limit to skip the :tokens counter so
    only the reservation path owns it.
    """
    monkeypatch.delenv("LITELLM_TPM_TOKEN_RESERVATION_ENABLED", raising=False)
    handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(DualCache()))

    user_api_key_dict = UserAPIKeyAuth(api_key=hash_token("sk-tpm"), tpm_limit=10_000)

    should_rate_limit_calls: list[dict[str, Any]] = []
    original_should_rate_limit = handler.should_rate_limit

    async def spy_should_rate_limit(*args, **kwargs):
        should_rate_limit_calls.append(kwargs)
        return await original_should_rate_limit(*args, **kwargs)

    reserve_calls: list[int] = []
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
    handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(DualCache()))

    user_api_key_dict = UserAPIKeyAuth(api_key=hash_token("sk-tpm"), tpm_limit=10_000)

    should_rate_limit_calls: list[dict[str, Any]] = []
    original_should_rate_limit = handler.should_rate_limit

    async def spy_should_rate_limit(*args, **kwargs):
        should_rate_limit_calls.append(kwargs)
        return await original_should_rate_limit(*args, **kwargs)

    reserve_calls: list[Any] = []

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
    assert get_request_stash().reserved_tokens == 0


@pytest.mark.asyncio
async def test_per_tag_descriptor_creation_v3():
    """
    create_rate_limit_descriptors emits a tag_per_key descriptor carrying the
    configured RPM limit only for request tags present in the configured map.
    """
    _api_key = hash_token("sk-per-tag-desc")
    user_api_key_dict = UserAPIKeyAuth(
        api_key=_api_key,
        metadata={"tag_rpm_limit": {"cell-1": 5}},
    )
    handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(DualCache()))

    descriptors = handler.create_rate_limit_descriptors(
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
    handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(DualCache()))

    descriptors = handler.create_rate_limit_descriptors(
        user_api_key_dict=user_api_key_dict,
        data={"model": "gpt-3.5-turbo", "metadata": {"tags": ["cell-1"]}},
        rpm_limit_type=None,
        tpm_limit_type=None,
        model_has_failures=False,
    )

    assert not [d for d in descriptors if d["key"] == "tag_per_key"]


@pytest.mark.asyncio
async def test_resolve_tag_rate_limits_from_db_reads_budget_row(monkeypatch):
    from litellm.models.budget import LiteLLM_BudgetTable
    from litellm.models.tag import LiteLLM_TagTable
    from litellm.proxy import proxy_server
    from litellm.proxy.auth import auth_checks
    from litellm.proxy.hooks.parallel_request_limiter_v3 import resolve_tag_rate_limits_from_db
    from litellm.proxy.utils import PrismaClient

    async def fake_batch(
        tag_names: Sequence[str],
        prisma_client: PrismaClient | None,
        user_api_key_cache: DualCache,
    ) -> dict[str, LiteLLM_TagTable]:
        return {
            "limited": LiteLLM_TagTable(
                tag_name="limited",
                litellm_budget_table=LiteLLM_BudgetTable(rpm_limit=3, tpm_limit=None),
            ),
            "spend-only": LiteLLM_TagTable(
                tag_name="spend-only",
                litellm_budget_table=LiteLLM_BudgetTable(max_budget=5.0),
            ),
            "bare": LiteLLM_TagTable(tag_name="bare"),
        }

    monkeypatch.setattr(proxy_server, "prisma_client", object())
    monkeypatch.setattr(auth_checks, "get_tag_objects_batch", fake_batch)

    assert dict(await resolve_tag_rate_limits_from_db(["limited", "spend-only", "bare"])) == {
        "limited": TagRateLimit(rpm_limit=3, tpm_limit=None)
    }

    monkeypatch.setattr(proxy_server, "prisma_client", None)
    assert dict(await resolve_tag_rate_limits_from_db(["limited"])) == {}


# --------------------------------------------------------------------------
# Streaming success logging mirrors x-ratelimit-* remaining values into
# standard_logging_object.hidden_params.additional_headers so Prometheus /
# logging callbacks see them for streams too (non-streaming already gets
# them via async_post_call_success_hook, which the streaming path skips).
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_log_success_event_counts_passthrough_reported_tokens(monkeypatch):
    """
    Pass-through responses are not modelled as ModelResponse/EmbeddingResponse,
    so their usage rides on ``combined_usage_object``. Without honoring it, a
    pass-through request never charged the TPM window and a team could exceed
    its shared token limit purely through pass-through traffic.
    """
    monkeypatch.setenv("LITELLM_RATE_LIMIT_WINDOW_SIZE", "60")

    _api_key = hash_token("sk-passthrough")
    fakes = FakeRateLimits()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache()),
        rate_limit_store=fakes.store(),
    )
    monkeypatch.setattr(parallel_request_handler, "get_rate_limit_type", lambda: "total")

    captured_operations = _capture_token_increments(fakes)

    await parallel_request_handler.async_log_success_event(
        kwargs={
            "standard_logging_object": {
                "metadata": {"user_api_key_hash": _api_key, "user_api_key_team_id": "team-fil"}
            },
            "combined_usage_object": Usage(total_tokens=1874),
        },
        response_obj={"response": "upstream body was never parsed"},
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    token_operations = [op for op in captured_operations if op["key"].endswith(":tokens")]
    assert token_operations, "pass-through usage should charge the TPM counter"
    assert all(op["increment_value"] == 1874 for op in token_operations)
    assert any("team-fil" in op["key"] for op in token_operations), (
        "team TPM window must be charged so pass-through shares the team's limit"
    )


@pytest.mark.parametrize("rate_limit_type", ["input", "output", "total"])
@pytest.mark.asyncio
async def test_aggregate_only_usage_charges_tpm_under_every_limit_type(monkeypatch, rate_limit_type):
    """
    A pass-through target reports one total for the whole request and cannot
    split it into prompt/completion. Reading a split out of it yields 0, which
    left the TPM window uncharged under input- or output-token limiting and let
    pass-through traffic run past a limit it is supposed to share.
    """
    monkeypatch.setenv("LITELLM_RATE_LIMIT_WINDOW_SIZE", "60")

    _api_key = hash_token("sk-aggregate-only")
    fakes = FakeRateLimits()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache()),
        rate_limit_store=fakes.store(),
    )
    monkeypatch.setattr(parallel_request_handler, "get_rate_limit_type", lambda: rate_limit_type)

    captured_operations = _capture_token_increments(fakes)

    await parallel_request_handler.async_log_success_event(
        kwargs={
            "standard_logging_object": {"metadata": {"user_api_key_hash": _api_key}},
            "combined_usage_object": Usage(total_tokens=1874),
        },
        response_obj={"response": "upstream body was never parsed"},
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    token_operations = [op for op in captured_operations if op["key"].endswith(":tokens")]
    assert token_operations, f"aggregate usage should charge TPM under {rate_limit_type}"
    assert all(op["increment_value"] == 1874 for op in token_operations)


@pytest.mark.asyncio
async def test_split_usage_still_respects_the_configured_limit_type(monkeypatch):
    """The aggregate fallback must not hijack usage that does carry a split."""
    monkeypatch.setenv("LITELLM_RATE_LIMIT_WINDOW_SIZE", "60")

    _api_key = hash_token("sk-split-usage")
    fakes = FakeRateLimits()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache()),
        rate_limit_store=fakes.store(),
    )
    monkeypatch.setattr(parallel_request_handler, "get_rate_limit_type", lambda: "output")

    captured_operations = _capture_token_increments(fakes)

    await parallel_request_handler.async_log_success_event(
        kwargs={"standard_logging_object": {"metadata": {"user_api_key_hash": _api_key}}},
        response_obj=ModelResponse(
            model="gpt-4o",
            usage=Usage(prompt_tokens=100, completion_tokens=7, total_tokens=107),
        ),
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    token_operations = [op for op in captured_operations if op["key"].endswith(":tokens")]
    assert token_operations
    assert all(op["increment_value"] == 7 for op in token_operations)


@pytest.mark.parametrize("origin", ["shadow_eval_judge", "autorouter_compaction"])
@pytest.mark.parametrize("rate_limit_type", ["input", "output", "total"])
def test_internal_call_origin_success_ops_are_skipped(origin, rate_limit_type):
    """Foreground compaction charges the same scopes as ordinary caller traffic."""
    handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(DualCache()))
    response = ModelResponse(
        id="internal-origin-tpm",
        object="chat.completion",
        created=int(datetime.now().timestamp()),
        model="gpt-4o-mini",
        usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        choices=[],
    )

    def _kwargs(metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            "standard_logging_object": {
                "metadata": {
                    "user_api_key_hash": hash_token("sk-internal-origin"),
                    "user_api_key_team_id": "compaction-team",
                    "user_api_key_project_id": "compaction-project",
                }
            },
            "litellm_params": {"metadata": metadata},
            "model": "gpt-4o-mini",
        }

    charged = handler._build_success_event_pipeline_operations(
        kwargs=_kwargs({}), response_obj=response, rate_limit_type=rate_limit_type
    )
    skipped = handler._build_success_event_pipeline_operations(
        kwargs=_kwargs({INTERNAL_CALL_ORIGIN_METADATA_KEY: origin}),
        response_obj=response,
        rate_limit_type=rate_limit_type,
    )

    assert charged
    assert skipped == (charged if origin == "autorouter_compaction" else [])


# ---------------------------------------------------------------------------
# LIT-5273: batch enqueued-token reservations in the post-call hooks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_call_success_hook_attaches_ratelimit_headers_to_dict_response():
    from litellm.proxy.hooks.parallel_request_limiter_v3 import RateLimitResponse, RateLimitStatus

    handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(DualCache()))
    get_or_create_request_stash().rate_limit_response = RateLimitResponse(
        overall_code="OK",
        statuses=[
            RateLimitStatus(
                code="OK",
                current_limit=100,
                limit_remaining=99,
                rate_limit_type="requests",
                descriptor_key="model_saturation_check",
            )
        ],
    )
    response = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [],
        "_hidden_params": {"additional_headers": {"x-litellm-attempted-retries": 0}},
    }

    await handler.async_post_call_success_hook(
        data={"model": "anthropic-haiku"},
        user_api_key_dict=UserAPIKeyAuth(api_key=hash_token("sk-dict-response")),
        response=response,
    )

    additional_headers = response["_hidden_params"]["additional_headers"]
    assert additional_headers["x-litellm-attempted-retries"] == 0
    assert additional_headers["x-ratelimit-model_saturation_check-limit-requests"] == 100
    assert additional_headers["x-ratelimit-model_saturation_check-remaining-requests"] == 99


@pytest.mark.asyncio
async def test_post_call_success_hook_leaves_raw_provider_dict_untouched():
    from litellm.proxy.hooks.parallel_request_limiter_v3 import RateLimitResponse, RateLimitStatus

    handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(DualCache()))
    get_or_create_request_stash().rate_limit_response = RateLimitResponse(
        overall_code="OK",
        statuses=[
            RateLimitStatus(
                code="OK",
                current_limit=100,
                limit_remaining=99,
                rate_limit_type="requests",
                descriptor_key="model_saturation_check",
            )
        ],
    )
    response = {"id": "msg_123", "type": "message", "role": "assistant", "content": []}

    await handler.async_post_call_success_hook(
        data={"model": "anthropic-haiku"},
        user_api_key_dict=UserAPIKeyAuth(api_key=hash_token("sk-raw-dict")),
        response=response,
    )

    assert response == {"id": "msg_123", "type": "message", "role": "assistant", "content": []}


_SCRIPT_NAMES: Final = (
    "batch_rate_limiter_script",
    "batch_counter_read_script",
    "check_and_increment_by_n_script",
    "window_guarded_token_increment_script",
    "parallel_acquire_script",
    "parallel_release_script",
    "parallel_count_script",
    "token_increment_script",
)
_CONNECTION_REFUSED: Final = "Error 61 connecting to 127.0.0.1:6379. Connection refused."


def _raising_redis(error: Exception) -> FakeRateLimits:
    async def refused(keys: list[str], args: list[object]) -> object:
        raise error

    return FakeRateLimits(**{name: refused for name in _SCRIPT_NAMES})


def _open_breaker_redis() -> FakeRateLimits:
    from litellm.caching.redis_cache import RedisCircuitBreakerOpenError

    return _raising_redis(RedisCircuitBreakerOpenError("Redis circuit breaker is open"))


def _unreachable_redis() -> FakeRateLimits:
    return _raising_redis(ConnectionError(_CONNECTION_REFUSED))


@pytest.mark.asyncio
async def test_an_open_circuit_breaker_falls_back_to_the_pipeline_without_a_warning(caplog):
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache()), rate_limit_store=_open_breaker_redis().store()
    )
    counter: Final = _counter("quiet", "key")

    with caplog.at_level(logging.DEBUG, logger="LiteLLM Proxy"):
        await handler.async_increment_tokens_with_ttl_preservation(
            pipeline_operations=[CounterIncrement(counter, 10.0, 60)]
        )

    assert _local(handler, counter) == 10.0
    assert [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING] == []


@pytest.mark.asyncio
async def test_an_open_circuit_breaker_reads_the_sliding_window_locally_without_a_warning(caplog):
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache()), rate_limit_store=_open_breaker_redis().store()
    )
    now: Final = int(time.time())

    with caplog.at_level(logging.DEBUG, logger="LiteLLM Proxy"):
        values = await handler._execute_redis_batch_rate_limiter_script(
            [_counter("quiet", "window", "requests")], now_int=now
        )

    assert values == [(str(now), 1)]
    assert [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING] == []
    assert any("circuit breaker is open" in record.getMessage() for record in caplog.records)


class _ScriptedRedis:
    def __init__(
        self,
        failing_script: str | None = None,
        failing_batch_call: int | None = None,
        stored_counter_value: int = 0,
    ):
        self.failing_script = failing_script
        self.failing_batch_call = failing_batch_call
        self.stored_counter_value = stored_counter_value
        self.released_slots: list[tuple[list[str], list[str]]] = []
        self.batch_calls = 0
        self.batch_call_keys: list[list[str]] = []
        self.batch_call_args: list[list[object]] = []
        self.guarded_increments: list[tuple[list[str], list[object]]] = []
        self.fakes: Final = FakeRateLimits(**{name: self._script(name) for name in _SCRIPT_NAMES})

    @property
    def increments(self) -> list[tuple[str, float]]:
        return [(key, amount) for key, amount, _ttl in self.fakes.redis_increments]

    def store(self) -> RateLimitStore:
        return self.fakes.store()

    def _script(self, name: str):
        async def run(keys: list[str], args: list[object]) -> object:
            if name == self.failing_script:
                raise ConnectionError(_CONNECTION_REFUSED)
            if name == "window_guarded_token_increment_script":
                self.guarded_increments.append((list(keys), list(args)))
                return [1, 0] * (len(keys) // 2)
            if name == "batch_rate_limiter_script":
                self.batch_calls += 1
                self.batch_call_keys.append(list(keys))
                self.batch_call_args.append(list(args))
                if self.batch_calls == self.failing_batch_call:
                    raise ConnectionError(_CONNECTION_REFUSED)
                return [args[0], self.batch_calls] * (len(keys) // 2)
            if name == "batch_counter_read_script":
                return [int(time.time()) if key.endswith(":window") else self.stored_counter_value for key in keys]
            if name == "parallel_count_script":
                return [0 for _ in keys]
            if name == "parallel_acquire_script":
                return [0, *[1 for _ in keys]]
            if name == "parallel_release_script":
                self.released_slots.append((list(keys), list(args)))
                return [0 for _ in keys]
            raise AssertionError(f"unexpected script: {name}")

        return run


def _handler_with_redis(
    redis: _ScriptedRedis | FakeRateLimits,
    fail_closed: bool | None = None,
    force_hash_tag_grouping: bool | None = None,
):
    internal_usage_cache = InternalUsageCache(DualCache())
    store: Final = redis.store()
    if fail_closed is None and force_hash_tag_grouping is None:
        return PROXY_MaxParallelRequestsHandler_v3(internal_usage_cache=internal_usage_cache, rate_limit_store=store)
    if fail_closed is None:
        return PROXY_MaxParallelRequestsHandler_v3(
            internal_usage_cache=internal_usage_cache,
            force_hash_tag_grouping_resolver=lambda: force_hash_tag_grouping,
            rate_limit_store=store,
        )
    if force_hash_tag_grouping is None:
        return PROXY_MaxParallelRequestsHandler_v3(
            internal_usage_cache=internal_usage_cache,
            fail_closed_resolver=lambda: fail_closed,
            rate_limit_store=store,
        )
    return PROXY_MaxParallelRequestsHandler_v3(
        internal_usage_cache=internal_usage_cache,
        fail_closed_resolver=lambda: fail_closed,
        force_hash_tag_grouping_resolver=lambda: force_hash_tag_grouping,
        rate_limit_store=store,
    )


_GROUPING_COUNTERS: Final = (
    _counter("api_key", "sk-abc", "requests"),
    _counter("team", "t1", "requests"),
    _counter("end_user", "u28551", "requests"),
)


@pytest.mark.parametrize("force_hash_tag_grouping", [True, False], ids=["forced", "default"])
@pytest.mark.asyncio
async def test_redis_batch_rate_limiter_respects_hash_tag_grouping_opt_in(force_hash_tag_grouping: bool):
    redis: Final = _ScriptedRedis()
    handler: Final = _handler_with_redis(redis, force_hash_tag_grouping=force_hash_tag_grouping)
    keys: Final = [
        "{api_key:sk-abc}:window",
        "{api_key:sk-abc}:requests",
        "{team:t1}:window",
        "{team:t1}:requests",
        "{end_user:u28551}:window",
        "{end_user:u28551}:requests",
    ]

    await handler._execute_redis_batch_rate_limiter_script(counters=_GROUPING_COUNTERS, now_int=1234)

    expected_calls: Final = (
        [
            [
                "{api_key:sk-abc}:window",
                "{api_key:sk-abc}:requests",
                "{end_user:u28551}:window",
                "{end_user:u28551}:requests",
            ],
            ["{team:t1}:window", "{team:t1}:requests"],
        ]
        if force_hash_tag_grouping
        else [keys]
    )
    assert redis.batch_call_keys == expected_calls


@pytest.mark.asyncio
async def test_redis_batch_rate_limiter_returns_values_in_caller_order():
    redis: Final = _ScriptedRedis()
    handler: Final = _handler_with_redis(redis, force_hash_tag_grouping=True)
    now: Final = 1234

    results: Final = await handler._execute_redis_batch_rate_limiter_script(counters=_GROUPING_COUNTERS, now_int=now)

    assert results == [(str(now), 1), (str(now), 2), (str(now), 1)]


async def _admit(handler, auth, data=None):
    await handler.async_pre_call_hook(
        user_api_key_dict=auth,
        cache=handler.internal_usage_cache.dual_cache,
        data=data if data is not None else {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
        call_type="acompletion",
    )


def test_a_total_timeout_never_interrupts_a_token_reservation_so_the_failure_path_can_refund_it() -> None:
    async def reserve_outlasting_the_budget(keys: list[str], args: list[object]) -> list[object]:
        await asyncio.sleep(120)
        return [0, *[value for index in range(0, len(args), 4) for value in (0, args[index + 1])]]

    async def admit_under_a_total_timeout() -> tuple[bool, int]:
        redis: Final = FakeRateLimits(check_and_increment_by_n_script=reserve_outlasting_the_budget)
        handler: Final = _handler_with_redis(redis)
        auth: Final = UserAPIKeyAuth(api_key=hash_token("sk-total-timeout-reservation"), tpm_limit=1000)
        stash: Final = get_or_create_request_stash()
        try:
            async with total_deadline(60, from_body=True):
                await _admit(handler, auth)
                await asyncio.sleep(3600)
        except TotalTimeoutExpired:
            return True, stash.reserved_tokens
        return False, stash.reserved_tokens

    loop: Final = AutojumpClockLoop()
    try:
        expired, reserved = loop.run_until_complete(admit_under_a_total_timeout())
    finally:
        loop.close()

    assert expired
    assert reserved > 0


async def _read_only_check(handler, auth):
    descriptors = handler.create_rate_limit_descriptors(
        user_api_key_dict=auth,
        data={"model": "test-model"},
        rpm_limit_type=None,
        tpm_limit_type=None,
        model_has_failures=False,
    )
    return await handler.should_rate_limit(descriptors=descriptors, read_only=True)


@pytest.mark.asyncio
async def test_fail_closed_is_a_no_op_while_redis_answers():
    handler = _handler_with_redis(_ScriptedRedis(), fail_closed=True)
    auth = UserAPIKeyAuth(api_key=hash_token("sk-fail-closed-healthy"), rpm_limit=2)

    await _admit(handler, auth)
    await _admit(handler, auth)
    with pytest.raises(ProxyRateLimitError) as exc:
        await _admit(handler, auth)

    assert exc.value.status_code == 429


@pytest.mark.asyncio
async def test_fail_closed_tpm_rejection_releases_the_parallel_slot_it_acquired():
    redis = _ScriptedRedis(failing_script="check_and_increment_by_n_script")
    handler = _handler_with_redis(redis, fail_closed=True)
    auth = UserAPIKeyAuth(api_key=hash_token("sk-fail-closed-slot"), max_parallel_requests=1, tpm_limit=1000)
    data = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}

    with pytest.raises(HTTPException) as exc:
        await _admit(handler, auth, data)
    assert exc.value.status_code == 503
    acquired = get_or_create_request_stash().parallel_slot
    assert acquired is not None

    await handler.async_post_call_failure_hook(request_data=data, original_exception=exc.value, user_api_key_dict=auth)

    assert redis.released_slots == [
        (
            [scope_key(gauge, "max_parallel_requests") for gauge in acquired["gauges"]],
            [acquired["slot_id"]],
        )
    ]
    assert get_or_create_request_stash().parallel_slot is None


@pytest.mark.parametrize("stored_counter_value, expected_code", [(1, "OK"), (2, "OVER_LIMIT"), (3, "OVER_LIMIT")])
@pytest.mark.asyncio
async def test_read_only_check_reports_the_redis_counters_without_incrementing_them(
    stored_counter_value, expected_code
):
    redis = _ScriptedRedis(stored_counter_value=stored_counter_value)
    auth = UserAPIKeyAuth(api_key=hash_token("sk-read-only-counters"), rpm_limit=2)

    response = await _read_only_check(_handler_with_redis(redis, fail_closed=True), auth)

    assert response["overall_code"] == expected_code
    assert redis.batch_calls == 0
    assert redis.increments == []


@pytest.mark.parametrize("fail_closed", [True, False], ids=["fail_closed", "fail_open"])
@pytest.mark.asyncio
async def test_batch_increment_refunds_counters_already_applied_when_a_later_cluster_slot_fails(fail_closed):
    redis = _ScriptedRedis(failing_batch_call=2)
    handler = _handler_with_redis(redis, fail_closed=fail_closed, force_hash_tag_grouping=True)
    auth = UserAPIKeyAuth(
        api_key=hash_token("sk-cluster-partial"), rpm_limit=5, user_id="cluster-user", user_rpm_limit=5
    )

    if fail_closed:
        with pytest.raises(HTTPException) as exc:
            await _admit(handler, auth)
        assert exc.value.status_code == 503
    else:
        await _admit(handler, auth)

    assert len(redis.batch_call_keys) == 2
    applied_keys = redis.batch_call_keys[0]
    assert applied_keys
    window_start_at_increment = str(redis.batch_call_args[0][0])
    expected_refunds = [
        ([applied_keys[offset], applied_keys[offset + 1]], [window_start_at_increment, -1, 0])
        for offset in range(0, len(applied_keys), 2)
    ]
    assert redis.guarded_increments == (expected_refunds if fail_closed else [])
    assert redis.increments == []


@pytest.mark.parametrize("fail_closed", [True, False], ids=["fail_closed", "fail_open"])
@pytest.mark.asyncio
async def test_batch_increment_refunds_pipelined_groups_declared_after_the_one_that_failed(fail_closed):
    from litellm.caching.redis_batch import request_redis_batch_scope

    redis = _ScriptedRedis()
    handler = _handler_with_redis(redis, fail_closed=fail_closed, force_hash_tag_grouping=True)
    now = int(time.time())
    counters = (_counter("a", "1", "requests"), _counter("b", "1", "requests"))
    groups = {"a": ["{a:1}:window", "{a:1}:requests"], "b": ["{b:1}:window", "{b:1}:requests"]}

    async def batch(keys: list[str], args: list[object]) -> list[object]:
        if keys == groups["a"]:
            raise ConnectionError(_CONNECTION_REFUSED)
        return [now, 1]

    redis.fakes.batch_rate_limiter_script = batch

    with request_redis_batch_scope():
        if fail_closed:
            with pytest.raises(HTTPException) as exc:
                await handler._execute_redis_batch_rate_limiter_script(counters=counters, now_int=now)
            assert exc.value.status_code == 503
        else:
            await handler._execute_redis_batch_rate_limiter_script(counters=counters, now_int=now)

    assert redis.guarded_increments == ([(groups["b"], [str(now), -1, 0])] if fail_closed else [])


class _DelayedCapacityRedis:
    def __init__(self):
        self.acquired = asyncio.Event()
        self.finish_admission = asyncio.Event()
        self.releasing = asyncio.Event()
        self.finish_release = asyncio.Event()
        self.fakes: Final = FakeRateLimits(parallel_acquire_script=self._acquire, parallel_release_script=self._release)

    async def _acquire(self, keys: list[str], args: list[object]) -> list[int]:
        self.acquired.set()
        await self.finish_admission.wait()
        return [0, *[1 for _ in keys]]

    async def _release(self, keys: list[str], args: list[object]) -> list[int]:
        self.releasing.set()
        await self.finish_release.wait()
        return [0 for _ in keys]


@pytest.mark.asyncio
async def test_request_capacity_finishes_admission_and_release_despite_repeated_cancel():
    cache = _DelayedCapacityRedis()
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache()), rate_limit_store=cache.fakes.store()
    )
    auth = UserAPIKeyAuth(api_key=hash_token("sk-count-cancel-admission"), max_parallel_requests=1)

    async def provider():
        async with handler.request_capacity(auth, "test-model"):
            pytest.fail("cancelled admission entered provider body")

    task = asyncio.create_task(provider())
    await asyncio.wait_for(cache.acquired.wait(), timeout=2)
    task.cancel()
    await asyncio.sleep(0)
    cache.finish_admission.set()
    await asyncio.wait_for(cache.releasing.wait(), timeout=2)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    cache.finish_release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2)
    async with handler.request_capacity(auth, "test-model"):
        pass


@pytest.mark.parametrize(
    "key_metadata, charges_team_model_pool",
    [
        ({}, True),
        ({"model_rpm_limit": {"test-model": 10}}, True),
        ({"model_tpm_limit": {"test-model": 5000}}, False),
        ({"model_tpm_limit": {"other-model": 5000}}, True),
    ],
    ids=["no_override", "rpm_only_override", "tpm_override", "tpm_override_on_other_model"],
)
def test_success_tpm_accounting_skips_team_model_pool_when_key_owns_model_tpm_limit(
    key_metadata, charges_team_model_pool
):
    handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(DualCache()))
    response = ModelResponse(
        id="team-pool-tpm",
        object="chat.completion",
        created=int(datetime.now().timestamp()),
        model="test-model",
        usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        choices=[],
    )
    kwargs = {
        "standard_logging_object": {
            "metadata": {"user_api_key_hash": hash_token("sk-pool"), "user_api_key_team_id": "t"}
        },
        "litellm_params": {
            "metadata": {
                "model_group": "test-model",
                "user_api_key_metadata": key_metadata,
                "user_api_key_team_metadata": {"model_tpm_limit": {"test-model": 500}},
            }
        },
        "model": "test-model",
    }

    ops = handler._build_success_event_pipeline_operations(
        kwargs=kwargs, response_obj=response, rate_limit_type="output"
    )

    charged_keys = {_increment_of(op).counter for op in ops}
    assert _counter("model_per_key", f"{hash_token('sk-pool')}:test-model") in charged_keys
    team_pool_key = _counter("model_per_team", "t:test-model")
    assert (team_pool_key in charged_keys) is charges_team_model_pool


@pytest.fixture(params=["Europe/Paris", "Asia/Kolkata", "America/Los_Angeles"])
def process_timezone(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    monkeypatch.setenv("TZ", request.param)
    time.tzset()
    yield request.param
    monkeypatch.undo()
    time.tzset()


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="switching the process timezone needs time.tzset()")
def test_rate_limit_error_reports_reset_time_in_utc_on_a_non_utc_proxy(process_timezone: str) -> None:
    now: Final = datetime(2026, 9, 4, 21, 53, 21, tzinfo=timezone.utc)
    handler: Final = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache()), time_provider=lambda: now
    )
    expected_reset: Final = (now + timedelta(seconds=handler.window_size)).strftime("%Y-%m-%d %H:%M:%S UTC")
    over_limit: Final[RateLimitResponse] = {
        "overall_code": "OVER_LIMIT",
        "statuses": [
            {
                "code": "OVER_LIMIT",
                "descriptor_key": "api_key",
                "limit_remaining": 0,
                "rate_limit_type": "requests",
                "current_limit": 2,
            }
        ],
    }

    with pytest.raises(ProxyRateLimitError) as exc_info:
        handler._handle_rate_limit_error(
            response=over_limit,
            descriptors=[{"key": "api_key", "value": "sk-test", "rate_limit": None}],
            requested_model="gpt-4o-mini",
        )

    assert exc_info.value.status_code == 429
    assert exc_info.value.headers == {
        "retry-after": str(handler.window_size),
        "rate_limit_type": "requests",
        "rate_limited_by": "api_key",
        "reset_at": expected_reset,
    }
    assert exc_info.value.detail == (
        "Rate limit exceeded for api_key: sk-test. Limit type: requests. "
        f"Current limit: 2, Remaining: 0. Limit resets at: {expected_reset}"
    )


def test_per_model_rate_limit_error_carries_descriptor_key() -> None:
    handler: Final = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(DualCache()))
    over_limit: Final[RateLimitResponse] = {
        "overall_code": "OVER_LIMIT",
        "statuses": [
            {
                "code": "OVER_LIMIT",
                "descriptor_key": "model_per_key",
                "descriptor_value": "gpt-4o-mini",
                "limit_remaining": 0,
                "rate_limit_type": "requests",
                "current_limit": 2,
            }
        ],
    }

    with pytest.raises(ProxyRateLimitError) as exc_info:
        handler._handle_rate_limit_error(
            response=over_limit,
            descriptors=[{"key": "model_per_key", "value": "gpt-4o-mini", "rate_limit": None}],
            requested_model="gpt-4o-mini",
        )

    assert exc_info.value.descriptor_key == "model_per_key"


def _resolve_alias_to_target(model: str) -> str | None:
    return "target" if model == "alias" else None


def test_tpm_settlement_targets_include_the_customer_levels_and_their_model_buckets() -> None:
    handler: Final = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(DualCache()))
    targets: Final = handler._collect_tpm_scope_targets(  # pyright: ignore[reportPrivateUsage]  # settlement enumerates scopes here
        standard_logging_metadata={
            "user_api_key_hash": "hash",
            "user_api_key_customer_org_id": "org-1",
            "user_api_key_customer_team_id": "team-1",
            "user_api_key_customer_member_id": "member-1",
        },
        kwargs={},
        model_group="target",
    )
    assert {
        ("customer_org", "org-1"),
        ("customer_team", "team-1"),
        ("customer_member", "member-1"),
        ("model_per_customer_org", "org-1:target"),
        ("model_per_customer_team", "team-1:target"),
        ("model_per_customer_member", "member-1:target"),
    } <= set(targets)
    without: Final = handler._collect_tpm_scope_targets(  # pyright: ignore[reportPrivateUsage]  # settlement enumerates scopes here
        standard_logging_metadata={"user_api_key_hash": "hash"}, kwargs={}, model_group="target"
    )
    assert not any(scope.startswith(("customer_", "model_per_customer_")) for scope, _ in without)


@pytest.mark.parametrize(
    "key_metadata, charges_team_model_pool",
    [({}, True), ({"model_tpm_limit": {"target": 500}}, False)],
    ids=["no_key_override", "key_owns_target_tpm_limit"],
)
def test_success_tpm_accounting_charges_the_alias_target_bucket(
    key_metadata: dict[str, object], charges_team_model_pool: bool
) -> None:
    handler: Final = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache()), model_group_resolver=_resolve_alias_to_target
    )
    response: Final = ModelResponse(
        id="alias-tpm",
        object="chat.completion",
        created=int(datetime.now().timestamp()),
        model="alias",
        usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        choices=[],
    )
    kwargs: Final = {
        "standard_logging_object": {
            "metadata": {"user_api_key_hash": hash_token("sk-alias-tpm"), "user_api_key_team_id": "t"}
        },
        "litellm_params": {
            "metadata": {
                "model_group": "alias",
                "user_api_key_metadata": key_metadata,
                "user_api_key_team_metadata": {"model_tpm_limit": {"target": 500}},
            }
        },
        "model": "alias",
    }

    ops: Final = handler._build_success_event_pipeline_operations(
        kwargs=kwargs, response_obj=response, rate_limit_type="output"
    )

    charged_keys: Final = {_increment_of(op).counter for op in ops}
    assert _counter("model_per_key", f"{hash_token('sk-alias-tpm')}:target") in charged_keys
    assert not any(":alias" in counter.scope.descriptor_value for counter in charged_keys)
    team_pool_key: Final = _counter("model_per_team", "t:target")
    assert (team_pool_key in charged_keys) is charges_team_model_pool


@pytest.mark.asyncio
async def test_success_tpm_accounting_keeps_the_admission_target_after_an_alias_reload() -> None:
    alias_map: Final[dict[str, str]] = {"alias": "target-a"}
    cache: Final = DualCache()
    handler: Final = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(cache), model_group_resolver=alias_map.get
    )
    key_metadata: Final = {"model_tpm_limit": {"target-a": 1000, "target-b": 1000}}
    auth: Final = UserAPIKeyAuth(api_key=hash_token("sk-alias-reload"), metadata=key_metadata)

    await handler.async_pre_call_hook(
        user_api_key_dict=auth,
        cache=cache,
        data={"model": "alias", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 10},
        call_type="acompletion",
    )
    stash: Final = get_request_stash()
    assert stash is not None
    assert stash.reserved_model == RateLimitedModel(requested="alias", group="target-a")
    assert stash.reserved_tokens > 0

    alias_map["alias"] = "target-b"
    response: Final = ModelResponse(
        id="alias-reload",
        object="chat.completion",
        created=int(datetime.now().timestamp()),
        model="alias",
        usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        choices=[],
    )
    kwargs: Final = {
        "standard_logging_object": {"metadata": {"user_api_key_hash": auth.api_key}},
        "litellm_params": {"metadata": {"model_group": "alias", "user_api_key_metadata": key_metadata}},
        "model": "alias",
    }

    ops: Final = handler._build_success_event_pipeline_operations(
        kwargs=kwargs, response_obj=response, rate_limit_type="total"
    )

    admission_bucket: Final = _counter("model_per_key", f"{auth.api_key}:target-a")
    charged: Final = {_increment_of(op).counter: _increment_of(op).amount for op in ops}
    assert charged[admission_bucket] == 150 - stash.reserved_tokens
    assert not any(":target-b" in counter.scope.descriptor_value for counter in charged)


# --- a team's PTU share on a shared Azure provisioned deployment ---------------------------


def _ptu_ceiling_for(team_id: str, model_group: str, tpm_limit: int, ratio: float, cached_ratio: float = 0.0):
    calls: list[tuple[str, str]] = []

    def resolve(requested_team: str, requested_group: str) -> PTUTeamCeiling | None:
        calls.append((requested_team, requested_group))
        if (requested_team, requested_group) != (team_id, model_group):
            return None
        return PTUTeamCeiling(
            model_group=model_group, tpm_limit=tpm_limit, output_to_input_ratio=ratio, cached_input_ratio=cached_ratio
        )

    return resolve, calls


def _ptu_request(model: str = "test-model") -> dict:
    return {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 300}


def _ptu_success_kwargs() -> dict:
    return {
        "standard_logging_object": {
            "metadata": {"user_api_key_hash": hash_token("sk-ptu"), "user_api_key_team_id": "t"}
        },
        "litellm_params": {
            "metadata": {"model_group": "test-model", "user_api_key_metadata": {}, "user_api_key_team_metadata": {}}
        },
        "model": "test-model",
    }


def _ptu_response(usage: Usage) -> ModelResponse:
    return ModelResponse(
        id="ptu-share",
        object="chat.completion",
        created=1_760_000_000,
        model="test-model",
        usage=usage,
        choices=[],
    )


def _ptu_increment(handler: _PROXY_MaxParallelRequestsHandler, ops: list) -> float | None:
    ptu_key = _counter("model_per_team_ptu", "t:test-model")
    return next((_increment_of(op).amount for op in ops if _increment_of(op).counter == ptu_key), None)


def test_success_accounting_settles_the_ptu_counter_in_azure_normalized_tokens():
    """100 uncached input tokens plus 50 output tokens at a 4:1 output ratio are 300
    normalized tokens, which is what the share was sized in."""
    resolve, _ = _ptu_ceiling_for("t", "test-model", tpm_limit=500, ratio=4.0)
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache()), ptu_team_ceiling_resolver=resolve
    )
    response = _ptu_response(Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150))

    ops = handler._build_success_event_pipeline_operations(
        kwargs=_ptu_success_kwargs(), response_obj=response, rate_limit_type="output"
    )

    assert _ptu_increment(handler, ops) == 300


def test_pass_through_usage_settles_the_ptu_counter_in_azure_normalized_tokens():
    """A pass-through response carries its usage beside the body rather than in it, and its 100
    input and 50 output tokens still settle at 300 normalized tokens, not a raw count."""
    resolve, _ = _ptu_ceiling_for("t", "test-model", tpm_limit=500, ratio=4.0)
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache()), ptu_team_ceiling_resolver=resolve
    )

    ops = handler._build_success_event_pipeline_operations(
        kwargs={
            **_ptu_success_kwargs(),
            "combined_usage_object": Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        },
        response_obj={"response": "upstream body was never parsed"},
        rate_limit_type="output",
    )

    assert _ptu_increment(handler, ops) == 300


def test_cached_input_is_not_charged_to_the_ptu_counter():
    resolve, _ = _ptu_ceiling_for("t", "test-model", tpm_limit=500, ratio=4.0)
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache()), ptu_team_ceiling_resolver=resolve
    )
    response = _ptu_response(
        Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=40),
        )
    )

    ops = handler._build_success_event_pipeline_operations(
        kwargs=_ptu_success_kwargs(), response_obj=response, rate_limit_type="output"
    )

    assert _ptu_increment(handler, ops) == 60 + 4 * 50


def test_cached_input_is_charged_at_the_models_cached_ratio():
    """40 of the 100 input tokens were cache reads; at a tenth each they are 4 normalized tokens
    beside the 60 uncached ones and the 200 for 50 outputs at 4:1."""
    resolve, _ = _ptu_ceiling_for("t", "test-model", tpm_limit=500, ratio=4.0, cached_ratio=0.1)
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache()), ptu_team_ceiling_resolver=resolve
    )
    response = _ptu_response(
        Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=40),
        )
    )

    ops = handler._build_success_event_pipeline_operations(
        kwargs=_ptu_success_kwargs(), response_obj=response, rate_limit_type="output"
    )

    assert _ptu_increment(handler, ops) == 60 + 4 + 4 * 50


@pytest.mark.asyncio
async def test_a_reservation_is_settled_even_after_the_teams_share_is_gone():
    """The share can be removed between admission and completion; the reserved tokens still
    come off the counter instead of standing in the window."""
    ceiling: dict[str, PTUTeamCeiling | None] = {
        "current": PTUTeamCeiling(
            model_group="test-model", tpm_limit=2000, output_to_input_ratio=4.0, cached_input_ratio=0.0
        )
    }
    cache = DualCache()
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(cache),
        ptu_team_ceiling_resolver=lambda _team, _group: ceiling["current"],
    )
    key = UserAPIKeyAuth(api_key=hash_token("sk-ptu"), team_id="t")

    await handler.async_pre_call_hook(user_api_key_dict=key, cache=cache, data=_ptu_request(), call_type="acompletion")
    stash = get_request_stash()
    assert stash is not None
    assert ("model_per_team_ptu", "t:test-model") in stash.reserved_scopes
    assert stash.ptu_reserved_tokens > stash.reserved_tokens > 150

    ceiling["current"] = None
    ops = handler._build_success_event_pipeline_operations(
        kwargs=_ptu_success_kwargs(),
        response_obj=_ptu_response(Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)),
        rate_limit_type="total",
    )

    assert _ptu_increment(handler, ops) == 300 - stash.ptu_reserved_tokens


@pytest.mark.asyncio
async def test_a_reservation_is_settled_when_the_success_metadata_lost_the_team():
    """Admission reserved on the team's PTU counter, so settlement credits that scope even when the
    success event's logging metadata carries no team id, instead of leaving the reservation standing."""
    cache = DualCache()
    resolve, _ = _ptu_ceiling_for("t", "test-model", tpm_limit=2000, ratio=4.0)
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(cache), ptu_team_ceiling_resolver=resolve
    )
    key = UserAPIKeyAuth(api_key=hash_token("sk-ptu"), team_id="t")

    await handler.async_pre_call_hook(user_api_key_dict=key, cache=cache, data=_ptu_request(), call_type="acompletion")
    stash = get_request_stash()
    assert stash is not None
    assert stash.ptu_reserved_tokens > 300

    ops = handler._build_success_event_pipeline_operations(
        kwargs={
            **_ptu_success_kwargs(),
            "standard_logging_object": {"metadata": {"user_api_key_hash": hash_token("sk-ptu")}},
        },
        response_obj=_ptu_response(Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)),
        rate_limit_type="total",
    )

    assert _ptu_increment(handler, ops) == 300 - stash.ptu_reserved_tokens


@pytest.mark.asyncio
async def test_success_settles_the_scope_the_reservation_was_taken_on():
    """Settlement credits the scope admission reserved, not one rebuilt from the name the
    request used, so a request naming the deployment by id cannot leave its reservation standing."""
    ceiling: dict[str, PTUTeamCeiling | None] = {
        "current": PTUTeamCeiling(
            model_group="test-model", tpm_limit=2000, output_to_input_ratio=4.0, cached_input_ratio=0.0
        )
    }
    cache = DualCache()
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(cache),
        ptu_team_ceiling_resolver=lambda _team, _group: ceiling["current"],
    )
    key = UserAPIKeyAuth(api_key=hash_token("sk-ptu"), team_id="t")

    await handler.async_pre_call_hook(
        user_api_key_dict=key, cache=cache, data=_ptu_request("shared-ptu"), call_type="acompletion"
    )
    stash = get_request_stash()
    assert stash is not None
    assert ("model_per_team_ptu", "t:test-model") in stash.reserved_scopes

    ceiling["current"] = None
    stash.ptu_ceiling = None
    kwargs = _ptu_success_kwargs()
    kwargs["litellm_params"]["metadata"]["model_group"] = "shared-ptu"
    ops = handler._build_success_event_pipeline_operations(
        kwargs=kwargs,
        response_obj=_ptu_response(Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)),
        rate_limit_type="total",
    )

    assert _ptu_increment(handler, ops) == 150 - stash.ptu_reserved_tokens


def test_usage_that_only_reports_a_total_charges_that_total_to_the_ptu_counter():
    resolve, _ = _ptu_ceiling_for("t", "test-model", tpm_limit=500, ratio=4.0)
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache()), ptu_team_ceiling_resolver=resolve
    )
    response = _ptu_response(Usage(prompt_tokens=0, completion_tokens=0, total_tokens=150))

    ops = handler._build_success_event_pipeline_operations(
        kwargs=_ptu_success_kwargs(), response_obj=response, rate_limit_type="output"
    )

    assert _ptu_increment(handler, ops) == 150


def test_a_socket_turns_usage_charges_the_key_but_leaves_the_team_ptu_counter_alone() -> None:
    resolve, _ = _ptu_ceiling_for("t", "test-model", tpm_limit=500, ratio=4.0)
    handler: Final = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache()), ptu_team_ceiling_resolver=resolve
    )

    ops: Final = handler._build_success_event_pipeline_operations(
        kwargs={**_ptu_success_kwargs(), "call_type": CallTypes.aresponses_websocket.value},
        response_obj=_responses_websocket_turn_events(total_tokens=150),
        rate_limit_type="total",
    )

    key_window: Final = _counter("api_key", hash_token("sk-ptu"))
    assert next(_increment_of(op).amount for op in ops if _increment_of(op).counter == key_window) == 150
    assert not _ptu_increment(handler, ops)


def test_success_accounting_charges_no_ptu_counter_without_a_ceiling():
    resolve, _ = _ptu_ceiling_for("u", "test-model", tpm_limit=500, ratio=4.0)
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache()), ptu_team_ceiling_resolver=resolve
    )
    response = _ptu_response(Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150))

    ops = handler._build_success_event_pipeline_operations(
        kwargs=_ptu_success_kwargs(), response_obj=response, rate_limit_type="output"
    )

    assert _ptu_increment(handler, ops) is None


@pytest.mark.parametrize("self_call", [False, True])
async def test_managed_invocations_enforce_actor_and_target_rate_policies(
    monkeypatch: pytest.MonkeyPatch, self_call: bool
) -> None:
    from litellm.types.agents import AgentResponse

    actor: Final = AgentResponse(
        agent_id="actor", agent_name="Actor", agent_card_params={}, rpm_limit=10, tpm_limit=1000
    )
    target: Final = AgentResponse(
        agent_id="target",
        agent_name="Target",
        agent_card_params={},
        rpm_limit=1,
        tpm_limit=1000,
        session_rpm_limit=1,
        session_tpm_limit=1000,
    )
    auth: Final = UserAPIKeyAuth(agent_id="actor")
    auth.managed_agent_policy = actor
    auth.invoked_agent_id = "actor" if self_call else "target"
    auth.invoked_agent_policy = actor if self_call else target
    cache: Final = DualCache()
    handler: Final = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(cache))
    monkeypatch.setattr(handler, "_get_agent_from_registry", lambda _: None)
    descriptors: Final = handler.create_rate_limit_descriptors(
        user_api_key_dict=auth,
        data={"model": "a2a/target", "litellm_session_id": "session"},
        rpm_limit_type=None,
        tpm_limit_type=None,
        model_has_failures=False,
    )
    limits: Final = {(item["key"], item["value"]): item["rate_limit"]["requests_per_unit"] for item in descriptors}
    assert limits == (
        {("agent", "actor"): 10}
        if self_call
        else {("agent", "actor"): 10, ("agent", "target"): 1, ("agent_session", "target:session"): 1}
    )
    assert len(descriptors) == len(limits)
    await handler.async_pre_call_hook(
        user_api_key_dict=auth,
        cache=cache,
        data={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 20,
            "litellm_session_id": "session",
        },
        call_type="acompletion",
    )
    stash: Final = get_request_stash()
    assert stash is not None and stash.reserved_tokens > 3
    response: Final = ModelResponse(usage=Usage(prompt_tokens=2, completion_tokens=1, total_tokens=3))
    operations: Final = handler._build_success_event_pipeline_operations(
        kwargs={"standard_logging_object": {"metadata": {"agent_id": auth.invoked_agent_id, "session_id": "session"}}},
        response_obj=response,
        rate_limit_type="total",
    )
    increments: Final = {_increment_of(op).counter: _increment_of(op).amount for op in operations}
    for scope in stash.reserved_scopes:
        if scope[0] in ("agent", "agent_session"):
            assert increments[_counter(*scope)] == 3 - stash.reserved_tokens


def test_project_itpm_adds_what_openai_chat_bills_on_top_of_a_pdf_files_pages_on_chat_completions_only() -> None:
    from litellm.constants import OPENAI_CHAT_PDF_FILE_TOKENS
    from tests.unit.proxy.spend_tracking.test_input_tokens import _one_page_pdf_message

    router: Final = Router(
        model_list=[
            {"model_name": name, "litellm_params": {"model": model, "api_key": "sk-test"}}
            for name, model in (("on-openai", "openai/gpt-5.6-sol"), ("on-azure", "azure/gpt-5.6-sol"))
        ]
    )
    handler: Final = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache()),
        counted_models_resolver=lambda model, team_id: models_counted_for(model, router, team_id),
    )
    data: Final = {"messages": _one_page_pdf_message()}

    assert handler._estimate_precise_input_tokens(data, "on-openai", "acompletion") == (
        handler._estimate_precise_input_tokens(data, "on-azure", "acompletion") + OPENAI_CHAT_PDF_FILE_TOKENS
    )
    assert handler._estimate_precise_input_tokens(data, "on-openai", "anthropic_messages") == (
        handler._estimate_precise_input_tokens(data, "on-azure", "anthropic_messages")
    )


_MULTILINGUAL_REQUEST: Final[dict[str, object]] = {
    "messages": [{"role": "user", "content": "日本語のテキストとहिन्दी पाठ और русский текст. " * 20}]
}


def _project_itpm_estimate(router: Router, model: str) -> int:
    handler: Final = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache()),
        counted_models_resolver=lambda model, team_id: models_counted_for(model, router, team_id),
    )
    return handler._estimate_precise_input_tokens(_MULTILINGUAL_REQUEST, model, "acompletion")


def test_project_itpm_counts_a_group_mixing_in_a_non_openai_model_by_its_name_as_written_and_its_openai_models():
    router: Final = Router(
        model_list=[
            {"model_name": "gpt-6.1-sol", "litellm_params": {"model": model, "api_key": "sk-test"}}
            for model in ("openai/gpt-6.1-sol", "cohere/command-r-plus")
        ]
    )
    messages: Final = _MULTILINGUAL_REQUEST["messages"]

    assert _project_itpm_estimate(router, "gpt-6.1-sol") == max(
        litellm.token_counter(model="my-model", messages=messages),
        litellm.token_counter(model="openai/gpt-6.1-sol", messages=messages),
    )


def test_project_itpm_counts_a_model_group_alias_as_its_target_group():
    router: Final = Router(
        model_list=[
            {"model_name": "gpt-6.1-sol", "litellm_params": {"model": "openai/gpt-6.1-sol", "api_key": "sk-test"}}
        ],
        model_group_alias={"prod": "gpt-6.1-sol"},
    )

    assert _project_itpm_estimate(router, "prod") == _project_itpm_estimate(router, "gpt-6.1-sol")


def test_media_models_resolve_to_each_deployment_base_model():
    from litellm.proxy.hooks.parallel_request_limiter_v3 import _resolve_media_models

    router: Final = Router(
        model_list=[
            {
                "model_name": "whisper",
                "litellm_params": {"model": "azure/my-whisper", "api_key": "k", "api_base": "https://x.invalid"},
                "model_info": {"id": "w1", "base_model": "azure/whisper-1"},
            }
        ]
    )
    assert _resolve_media_models(None, "whisper", None) == ("whisper",)
    assert _resolve_media_models(router, "whisper", None) == ("azure/whisper-1",)
    assert _resolve_media_models(router, "unknown", None) == ("unknown",)


@pytest.mark.asyncio
async def test_success_tpm_accounting_moves_the_charge_to_the_model_group_that_served_a_fallback() -> None:
    cache: Final = DualCache()
    handler: Final = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(cache))
    key_metadata: Final = {"model_tpm_limit": {"primary": 1000, "fallback": 1000}}
    auth: Final = UserAPIKeyAuth(api_key=hash_token("sk-fallback-settle"), metadata=key_metadata)

    await handler.async_pre_call_hook(
        user_api_key_dict=auth,
        cache=cache,
        data={"model": "primary", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 10},
        call_type="acompletion",
    )
    stash: Final = get_request_stash()
    assert stash is not None
    assert stash.reserved_model == RateLimitedModel(requested="primary", group="primary")
    assert stash.reserved_tokens > 0
    assert await handler.async_fallback_target_allowed(
        auth, "fallback", {"messages": [{"role": "user", "content": "hello"}]}
    )

    response: Final = ModelResponse(
        id="fallback-settle",
        object="chat.completion",
        created=int(datetime.now().timestamp()),
        model="fallback",
        usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        choices=[],
    )
    kwargs: Final = {
        "standard_logging_object": {"metadata": {"user_api_key_hash": auth.api_key}},
        "litellm_params": {
            "metadata": {
                "model_group": "fallback",
                "original_model_group": "primary",
                "user_api_key_metadata": key_metadata,
            }
        },
        "model": "fallback",
    }

    ops: Final = handler._build_success_event_pipeline_operations(
        kwargs=kwargs, response_obj=response, rate_limit_type="total"
    )

    charged: Final = {_increment_of(op).counter: _increment_of(op).amount for op in ops}
    assert charged[_counter("model_per_key", f"{auth.api_key}:fallback")] == 150
    assert charged[_counter("model_per_key", f"{auth.api_key}:primary")] == -stash.reserved_tokens


@pytest.mark.asyncio
async def test_success_tpm_accounting_settles_an_alias_to_target_fallback_once_per_counter(
    collision_limiter: tuple[DualCache, PROXY_MaxParallelRequestsHandler_v3, str, str],
) -> None:
    cache, handler, alias, target = collision_limiter
    key_metadata: Final = {"model_tpm_limit": {alias: 1000, target: 1000}}
    auth: Final = UserAPIKeyAuth(api_key=hash_token("sk-alias-fallback-settle"), metadata=key_metadata)

    await handler.async_pre_call_hook(
        user_api_key_dict=auth,
        cache=cache,
        data={"model": alias, "messages": [{"role": "user", "content": "hello"}], "max_tokens": 10},
        call_type="acompletion",
    )
    stash: Final = get_request_stash()
    assert stash is not None
    assert stash.reserved_model == RateLimitedModel(requested=alias, group=target)
    assert stash.reserved_tokens > 0
    assert await handler.async_fallback_target_allowed(
        auth, target, {"messages": [{"role": "user", "content": "hello"}]}
    )

    response: Final = ModelResponse(
        id="alias-fallback-settle",
        object="chat.completion",
        created=int(datetime.now().timestamp()),
        model=target,
        usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        choices=[],
    )
    kwargs: Final = {
        "standard_logging_object": {"metadata": {"user_api_key_hash": auth.api_key}},
        "litellm_params": {
            "metadata": {"model_group": target, "original_model_group": alias, "user_api_key_metadata": key_metadata}
        },
        "model": target,
    }

    ops: Final = handler._build_success_event_pipeline_operations(
        kwargs=kwargs, response_obj=response, rate_limit_type="total"
    )

    increments: Final = tuple(_increment_of(op) for op in ops)
    target_counter: Final = _counter("model_per_key", f"{auth.api_key}:{target}")
    alias_counter: Final = _counter("model_per_key", f"{auth.api_key}:{alias}")
    assert [inc.amount for inc in increments if inc.counter == target_counter] == [150 - stash.reserved_tokens]
    assert [inc.amount for inc in increments if inc.counter == alias_counter] == []


@pytest.mark.asyncio
async def test_success_tpm_accounting_ignores_a_client_supplied_model_group_without_a_router_fallback() -> None:
    cache: Final = DualCache()
    handler: Final = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(cache))
    key_metadata: Final = {"model_tpm_limit": {"primary": 1000}}
    auth: Final = UserAPIKeyAuth(api_key=hash_token("sk-client-model-group"), metadata=key_metadata)

    await handler.async_pre_call_hook(
        user_api_key_dict=auth,
        cache=cache,
        data={"model": "primary", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 10},
        call_type="acompletion",
    )
    stash: Final = get_request_stash()
    assert stash is not None
    assert stash.reserved_tokens > 0

    response: Final = ModelResponse(
        id="client-model-group",
        object="chat.completion",
        created=int(datetime.now().timestamp()),
        model="primary",
        usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        choices=[],
    )
    kwargs: Final = {
        "standard_logging_object": {"metadata": {"user_api_key_hash": auth.api_key}},
        "litellm_params": {"metadata": {"model_group": "uncapped", "user_api_key_metadata": key_metadata}},
        "model": "primary",
    }

    ops: Final = handler._build_success_event_pipeline_operations(
        kwargs=kwargs, response_obj=response, rate_limit_type="total"
    )

    charged: Final = {_increment_of(op).counter: _increment_of(op).amount for op in ops}
    assert charged[_counter("model_per_key", f"{auth.api_key}:primary")] == 150 - stash.reserved_tokens
    assert not any(":uncapped" in counter.scope.descriptor_value for counter in charged)
