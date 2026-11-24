"""
Unit Tests for the max parallel request limiter v3 for the proxy
"""

import asyncio
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
from litellm.litellm_core_utils.model_latest_aliases import get_model_latest_aliases
from litellm.litellm_core_utils.ptu_shares import PTUTeamCeiling
from litellm.litellm_core_utils.routing_common_utils import resolve_model_group_alias
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.hooks import parallel_request_limiter_v3 as parallel_request_limiter_v3_module
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    PARALLEL_REQUEST_SLOT_TTL_SECONDS,
    PROXY_MaxParallelRequestsHandler_v3,
    RateLimitDescriptor,
    RateLimitedModel,
    RateLimitResponse,
    TagRateLimit,
    _mcp_scope,  # pyright: ignore[reportPrivateUsage]  # the MCP entry point's own scope
    _request_stash,
    current_time,
    enforce_mcp_server_rate_limits,
    estimate_precise_input_tokens,
    fallback_target_allowed,
    get_or_create_request_stash,
    handle_rate_limit_error,
    run_scoped,
    token_rate_limit_type,
)
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    PROXY_MaxParallelRequestsHandler_v3 as _PROXY_MaxParallelRequestsHandler,
)
from litellm.proxy.spend_tracking.input_tokens import models_counted_for
from litellm.proxy.utils import InternalUsageCache, hash_token
from litellm.redis.stores.rate_limits import (
    CounterIncrement,
    CounterKind,
    LocalValue,
    RateLimitCounter,
    RateLimitEntry,
    RateLimitScope,
)
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
from tests._support.rust_parallel_request_limiter import (
    PROXY_MaxParallelRequestsHandler_v3 as NativeLimiter,
)
from tests._support.rust_parallel_request_limiter import (
    estimate_tokens_for_request,
    get_request_stash,
)
from tests.unit.redis.stores.fake_rate_limits import scope_key


def _set_token_rate_limit_type(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    from litellm.proxy import proxy_server

    monkeypatch.setattr(
        proxy_server, "general_settings", {**proxy_server.general_settings, "token_rate_limit_type": value}
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


def _counter(key: str, value: str, kind: CounterKind = "tokens") -> RateLimitCounter:
    return RateLimitScope(key, value).counter(kind)


def _local(handler: _PROXY_MaxParallelRequestsHandler, entry: RateLimitEntry) -> LocalValue | None:
    return handler.rate_limits.local((entry,))[0]


def _gauge_in_flight(handler: _PROXY_MaxParallelRequestsHandler, raw_value: LocalValue | None) -> int:
    """Unexpired slots in a cached gauge: a slot registry dict, or a mirrored integer count."""
    if raw_value is None:
        return 0
    if isinstance(raw_value, dict):
        cutoff: Final = current_time(handler).timestamp() - PARALLEL_REQUEST_SLOT_TTL_SECONDS
        return sum(1 for ts in raw_value.values() if ts >= cutoff)
    return max(0, int(raw_value))


def _tokens(handler: _PROXY_MaxParallelRequestsHandler, key: str, value: str) -> float:
    return float(_local(handler, _counter(key, value)) or 0)


async def _request_descriptors(
    handler: _PROXY_MaxParallelRequestsHandler, auth: UserAPIKeyAuth, data: dict[str, object]
) -> Sequence[RateLimitDescriptor]:
    scoped: Final = await run_scoped(handler, auth, data, {"kind": "request"}, {"kind": "build"})
    return scoped.descriptors


async def _mcp_descriptors(
    handler: _PROXY_MaxParallelRequestsHandler, auth: UserAPIKeyAuth, server: MCPServer
) -> Sequence[RateLimitDescriptor]:
    scoped: Final = await run_scoped(
        handler, auth, {}, {"kind": "mcp", **_mcp_scope(auth, server)}, {"kind": "build"}, lookups=False
    )
    return scoped.descriptors


async def _log_success(
    handler: _PROXY_MaxParallelRequestsHandler, kwargs: dict[str, object], response_obj: object
) -> None:
    await handler.async_log_success_event(
        kwargs=kwargs, response_obj=response_obj, start_time=datetime.now(), end_time=datetime.now()
    )


def _token_counters(handler: _PROXY_MaxParallelRequestsHandler, *scopes: tuple[str, str]) -> list[dict[str, object]]:
    counters: Final = tuple(RateLimitScope(key, value).counter("tokens") for key, value in scopes)
    values: Final = handler.rate_limits.local(counters)
    return [
        {"key": scope_key(counter.scope, "tokens"), "increment_value": value}
        for counter, value in zip(counters, values)
        if value
    ]


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
    handler: Final = NativeLimiter(
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
@pytest.mark.asyncio
async def test_api_key_descriptor_applies_budget_throttle(throttle_pct, expected_rpm, expected_tpm):
    """The api_key rate-limit descriptor scales the key's configured TPM/RPM by
    the request-scoped budget_throttle_pct, leaving the configured limits intact."""
    handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(DualCache()))
    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-throttle"),
        rpm_limit=100,
        tpm_limit=1000,
        budget_throttle_pct=throttle_pct,
    )

    descriptors = await _request_descriptors(handler, user_api_key_dict, {})

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
    parallel_request_handler = NativeLimiter(
        internal_usage_cache=InternalUsageCache(local_cache),
        time_provider=time_controller.now,
    )

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
    parallel_request_handler = NativeLimiter(
        internal_usage_cache=InternalUsageCache(local_cache),
        time_provider=time_controller.now,
    )

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
    # the reservation populates the :tokens counter with a predictable amount.
    # Estimate: input ~ 1 token (`"hi"`), max_tokens = 5 → reservation = 6,
    # which fits under the tpm_limit of 10.
    pre_call_data = {
        "model": "azure-model",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 5,
    }
    expected_reservation = estimate_tokens_for_request(parallel_request_handler, pre_call_data)
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

    # Verify the token count is tracked (populated by the reservation).
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


@pytest.mark.asyncio
@pytest.mark.parametrize("tpm_limit,rpm_limit", ((50, 5), (500, 1)))
async def test_zero_cost_subscription_request_still_consumes_tpm_and_rpm(monkeypatch, tpm_limit, rpm_limit):
    _set_token_rate_limit_type(monkeypatch, "total")
    now: Final = datetime(2026, 11, 19, 12, 0, tzinfo=timezone.utc)
    key: Final = hash_token("sk-subscription-rate-limit")
    cache: Final = DualCache()
    handler: Final = NativeLimiter(
        internal_usage_cache=InternalUsageCache(cache),
        time_provider=lambda: now,
    )
    auth: Final = UserAPIKeyAuth(api_key=key, tpm_limit=tpm_limit, rpm_limit=rpm_limit)
    data: Final = {
        "model": "subscription-test",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
    }
    await handler.async_pre_call_hook(user_api_key_dict=auth, cache=cache, data=data, call_type="acompletion")
    await handler.async_log_success_event(
        kwargs={
            "model": "subscription-test",
            "response_cost": 0.0,
            "standard_logging_object": {
                "response_cost": 0.0,
                "metadata": {
                    "user_api_key_hash": key,
                    "used_client_oauth_token": True,
                    "claude_subscription_usage": {"is_overage": False, "status": "allowed", "overage_status": None},
                },
            },
        },
        response_obj=ModelResponse(usage=Usage(prompt_tokens=20, completion_tokens=30, total_tokens=50)),
        start_time=now,
        end_time=now,
    )
    assert _tokens(handler, "api_key", key) == 50
    assert _local(handler, _counter("api_key", key, "requests")) == 1
    with pytest.raises(ProxyRateLimitError):
        await handler.async_pre_call_hook(user_api_key_dict=auth, cache=cache, data=data, call_type="acompletion")


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
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache),
    )

    def mock_get_rate_limit_type():
        return token_rate_limit_type

    _set_token_rate_limit_type(monkeypatch, mock_get_rate_limit_type())

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

    # Call the success event handler
    await parallel_request_handler.async_log_success_event(
        kwargs=mock_kwargs,
        response_obj=mock_response,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )
    captured_operations = _token_counters(parallel_request_handler, ("api_key", _api_key))

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
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache()),
    )
    _set_token_rate_limit_type(monkeypatch, "total")

    mock_kwargs = {
        "standard_logging_object": {"metadata": {"user_api_key_hash": _api_key}},
        "model": response_obj.model,
    }

    await parallel_request_handler.async_log_success_event(
        kwargs=mock_kwargs,
        response_obj=response_obj,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )
    captured_operations = _token_counters(parallel_request_handler, ("api_key", _api_key))

    tpm_operation = next((op for op in captured_operations if op["key"].endswith(":tokens")), None)
    assert tpm_operation is not None, "Should have a TPM increment operation"
    assert tpm_operation["increment_value"] == 50


# Redis Cluster Compatibility Tests


def test_missing_descriptor_fallback():
    """
    Test that the fallback works when a descriptor_key cannot be found in the descriptors list.

    This tests an edge case where somehow the descriptor_key in status doesn't match
    any descriptor key (shouldn't happen in normal operation but good for robustness).
    """
    _api_key_hash = hash_token("sk-98765")
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(DualCache()))

    with pytest.raises(HTTPException) as exc_info:
        handle_rate_limit_error(
            parallel_request_handler,
            response={
                "overall_code": "OVER_LIMIT",
                "statuses": [
                    {
                        "code": "OVER_LIMIT",
                        "current_limit": 2,
                        "limit_remaining": -1,
                        "rate_limit_type": "requests",
                        "descriptor_key": "nonexistent_key",
                    }
                ],
            },
            descriptors=[
                {
                    "key": "api_key",
                    "value": _api_key_hash,
                    "rate_limit": {"requests_per_unit": 2, "tokens_per_unit": None, "window_size": 60},
                }
            ],
            requested_model="gpt-3.5-turbo",
        )

    # Verify the exception uses fallback values
    assert exc_info.value.status_code == 429
    assert "Rate limit exceeded for nonexistent_key: unknown" in exc_info.value.detail
    assert "requests" in exc_info.value.detail
    assert "Current limit: 2" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_rate_limit_type_default_is_total(monkeypatch):
    """
    Test that token_rate_limit_type returns 'total' as the default when no setting is specified.

    This verifies the change from 'output' to 'total' as the default value.
    """
    # Mock general_settings to return empty dict (no token_rate_limit_type set)
    from litellm.proxy import proxy_server

    original_settings = getattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(proxy_server, "general_settings", {})

    try:
        result = token_rate_limit_type()
        assert result == "total", f"Default rate limit type should be 'total', got '{result}'"
    finally:
        monkeypatch.setattr(proxy_server, "general_settings", original_settings)


@pytest.mark.asyncio
async def test_get_rate_limit_type_invalid_falls_back_to_total(monkeypatch):
    """
    Test that token_rate_limit_type falls back to 'total' when an invalid value is specified.
    """
    # Mock general_settings to return an invalid token_rate_limit_type
    from litellm.proxy import proxy_server

    original_settings = getattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(proxy_server, "general_settings", {"token_rate_limit_type": "invalid_type"})

    try:
        result = token_rate_limit_type()
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
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache),
    )

    def mock_get_rate_limit_type():
        return token_rate_limit_type

    _set_token_rate_limit_type(monkeypatch, mock_get_rate_limit_type())

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

    # Call the success event handler
    await parallel_request_handler.async_log_success_event(
        kwargs=mock_kwargs,
        response_obj=mock_response,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )
    captured_operations = _token_counters(parallel_request_handler, ("api_key", _api_key))

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
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache),
    )

    def mock_get_rate_limit_type():
        return "output"

    _set_token_rate_limit_type(monkeypatch, mock_get_rate_limit_type())

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

    # Call the success event handler - should not raise exception
    await parallel_request_handler.async_log_success_event(
        kwargs=mock_kwargs,
        response_obj=mock_response,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )
    captured_operations = _token_counters(parallel_request_handler, ("api_key", _api_key))

    # When total_tokens resolves to 0 (missing fields) and there's no reservation,
    # the reconciliation delta is 0 — no TPM increment should be emitted.
    tpm_ops = [op for op in captured_operations if op["key"].endswith(":tokens")]
    assert tpm_ops == [], f"Expected no TPM ops when usage is empty, got: {tpm_ops}"


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
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache),
    )

    def mock_get_rate_limit_type():
        return "total"

    _set_token_rate_limit_type(monkeypatch, mock_get_rate_limit_type())

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

    await parallel_request_handler.async_log_success_event(
        kwargs=mock_kwargs,
        response_obj=mock_response,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )
    captured_operations = _token_counters(
        parallel_request_handler, ("agent", _agent_id), ("agent_session", f"{_agent_id}:{_session_id}")
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
    parallel_request_handler = NativeLimiter(
        internal_usage_cache=InternalUsageCache(local_cache),
        time_provider=time_controller.now,
    )

    mock_agent = AgentResponse(
        agent_id=_agent_id,
        agent_name="rate-limited-agent",
        agent_card_params={"name": "Rate Limited Agent"},
        rpm_limit=2,
    )

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


def _usage_response(usage: Usage | dict[str, object]) -> object:
    from unittest.mock import MagicMock

    from litellm.types.utils import BaseLiteLLMOpenAIResponseObject

    if isinstance(usage, Usage):
        return ModelResponse(model="gpt-4o", usage=usage)
    response: Final = MagicMock(spec=BaseLiteLLMOpenAIResponseObject)
    response.usage = usage
    return response


_CACHED_USAGE: Final = Usage(
    prompt_tokens=1000,
    completion_tokens=500,
    total_tokens=1500,
    prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=800),
)


class TestTpmChargeExcludesCachedTokens:
    """
    The TPM charge settled on success excludes cached prompt tokens.

    Issue: AWS Bedrock and similar providers exclude cache tokens from TPM calculation,
    but LiteLLM was including them, causing up to 10x difference in rate limiting.
    """

    @pytest.mark.parametrize(
        "usage, rate_limit_type, expected",
        [
            (_CACHED_USAGE, "total", 700),
            (_CACHED_USAGE, "input", 200),
            (_CACHED_USAGE, "output", 500),
            (Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500), "total", 1500),
            (
                {
                    "prompt_tokens": 1000,
                    "completion_tokens": 500,
                    "total_tokens": 1500,
                    "prompt_tokens_details": {"cached_tokens": 600},
                },
                "total",
                900,
            ),
        ],
        ids=["total", "input", "output", "no_cached_tokens", "dict_usage"],
    )
    @pytest.mark.asyncio
    async def test_charges_usage_without_cached_tokens(
        self,
        monkeypatch: pytest.MonkeyPatch,
        usage: Usage | dict[str, object],
        rate_limit_type: str,
        expected: int,
    ) -> None:
        handler: Final = PROXY_MaxParallelRequestsHandler_v3(internal_usage_cache=InternalUsageCache(DualCache()))
        _set_token_rate_limit_type(monkeypatch, rate_limit_type)
        api_key: Final = hash_token("sk-cached-tokens")

        await _log_success(
            handler,
            {"standard_logging_object": {"metadata": {"user_api_key_hash": api_key}}, "model": "gpt-4o"},
            _usage_response(usage),
        )

        assert _tokens(handler, "api_key", api_key) == expected

    @pytest.mark.asyncio
    async def test_charges_nothing_without_usage(self) -> None:
        handler: Final = PROXY_MaxParallelRequestsHandler_v3(internal_usage_cache=InternalUsageCache(DualCache()))
        api_key: Final = hash_token("sk-no-usage")

        await _log_success(
            handler, {"standard_logging_object": {"metadata": {"user_api_key_hash": api_key}}, "model": "gpt-4o"}, None
        )

        assert _tokens(handler, "api_key", api_key) == 0


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

    assert [
        (descriptor["key"], descriptor["value"])
        for descriptor in await _mcp_descriptors(handler, inherited_auth, server)
    ] == [("mcp_per_team", "team-a:math_a")]
    await enforce_mcp_server_rate_limits(handler, inherited_auth, server)

    with pytest.raises(ProxyRateLimitError, match="mcp_per_team"):
        await enforce_mcp_server_rate_limits(handler, inherited_auth, server)


@pytest.mark.asyncio
async def test_inherited_mcp_team_rate_limit_uses_source_limit_when_primary_metadata_is_missing() -> None:
    handler, _ = _make_mcp_handler()
    server: Final = MCPServer(
        server_id="math-a",
        name="math_a",
        server_name="math_a",
        transport=MCPTransport.http,
    )
    auth: Final = UserAPIKeyAuth(
        api_key=hash_token("sk-inherited-source-limit"),
        team_id="team-a",
    ).model_copy(update={"mcp_source_team_rpm_limits": {"team-a": {"math_a": 1}}})

    assert [
        (descriptor["key"], descriptor["value"], descriptor["rate_limit"])
        for descriptor in await _mcp_descriptors(handler, auth, server)
    ] == [
        (
            "mcp_per_team",
            "team-a:math_a",
            {"requests_per_unit": 1, "tokens_per_unit": None, "window_size": handler.window_size},
        )
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

    descriptors: Final = await _request_descriptors(handler, billed_auth, {})

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
    handler = NativeLimiter(internal_usage_cache=InternalUsageCache(local_cache))
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key, max_parallel_requests=5, tpm_limit=100)
    counter_key = RateLimitScope("api_key", _api_key).gauge

    with pytest.raises(HTTPException) as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1000},
            call_type="",
        )
    assert exc_info.value.status_code == 429
    assert exc_info.value.headers["rate_limit_type"] == "tokens"
    assert _gauge_in_flight(handler, _local(handler, counter_key)) == 0


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
    budget upfront on the :tokens counter.
    """
    monkeypatch.delenv("LITELLM_TPM_TOKEN_RESERVATION_ENABLED", raising=False)
    handler = NativeLimiter(internal_usage_cache=InternalUsageCache(DualCache()))
    user_api_key_dict = UserAPIKeyAuth(api_key=hash_token("sk-tpm"), tpm_limit=10_000)

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=handler.internal_usage_cache.dual_cache,
        data={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
        call_type="completion",
    )

    stash = get_request_stash()
    assert stash is not None
    assert stash.reserved_tokens > 0, "reservation must run when enabled"
    assert _tokens(handler, "api_key", user_api_key_dict.api_key) == stash.reserved_tokens


@pytest.mark.asyncio
async def test_pre_call_hook_skips_reservation_when_disabled(monkeypatch):
    """
    With reservation disabled, the pre-call hook reserves nothing and enforces
    TPM from the usage already counted, the pre-v1.82 post-call accounting
    behavior, so a request whose estimate exceeds the limit is still admitted.
    """
    monkeypatch.setenv("LITELLM_TPM_TOKEN_RESERVATION_ENABLED", "false")
    handler = NativeLimiter(internal_usage_cache=InternalUsageCache(DualCache()))
    user_api_key_dict = UserAPIKeyAuth(api_key=hash_token("sk-tpm"), tpm_limit=10)

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=handler.internal_usage_cache.dual_cache,
        data={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1000},
        call_type="completion",
    )

    stash = get_request_stash()
    assert stash is not None
    assert stash.reserved_tokens == 0, "reservation must be skipped when disabled"


@pytest.mark.asyncio
async def test_per_tag_descriptor_creation_v3():
    """
    The pre-call emits a tag_per_key descriptor carrying the configured RPM
    limit only for request tags present in the configured map.
    """
    _api_key = hash_token("sk-per-tag-desc")
    user_api_key_dict = UserAPIKeyAuth(
        api_key=_api_key,
        metadata={"tag_rpm_limit": {"cell-1": 5}},
    )
    handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(DualCache()))

    descriptors = await _request_descriptors(
        handler, user_api_key_dict, {"model": "gpt-3.5-turbo", "metadata": {"tags": ["cell-1", "cell-2"]}}
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

    descriptors = await _request_descriptors(
        handler, user_api_key_dict, {"model": "gpt-3.5-turbo", "metadata": {"tags": ["cell-1"]}}
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
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache()),
    )
    _set_token_rate_limit_type(monkeypatch, "total")

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
    captured_operations = _token_counters(parallel_request_handler, ("api_key", _api_key), ("team", "team-fil"))

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
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache()),
    )
    _set_token_rate_limit_type(monkeypatch, rate_limit_type)

    await parallel_request_handler.async_log_success_event(
        kwargs={
            "standard_logging_object": {"metadata": {"user_api_key_hash": _api_key}},
            "combined_usage_object": Usage(total_tokens=1874),
        },
        response_obj={"response": "upstream body was never parsed"},
        start_time=datetime.now(),
        end_time=datetime.now(),
    )
    captured_operations = _token_counters(parallel_request_handler, ("api_key", _api_key))

    token_operations = [op for op in captured_operations if op["key"].endswith(":tokens")]
    assert token_operations, f"aggregate usage should charge TPM under {rate_limit_type}"
    assert all(op["increment_value"] == 1874 for op in token_operations)


@pytest.mark.asyncio
async def test_split_usage_still_respects_the_configured_limit_type(monkeypatch):
    """The aggregate fallback must not hijack usage that does carry a split."""
    monkeypatch.setenv("LITELLM_RATE_LIMIT_WINDOW_SIZE", "60")

    _api_key = hash_token("sk-split-usage")
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache()),
    )
    _set_token_rate_limit_type(monkeypatch, "output")

    await parallel_request_handler.async_log_success_event(
        kwargs={"standard_logging_object": {"metadata": {"user_api_key_hash": _api_key}}},
        response_obj=ModelResponse(
            model="gpt-4o",
            usage=Usage(prompt_tokens=100, completion_tokens=7, total_tokens=107),
        ),
        start_time=datetime.now(),
        end_time=datetime.now(),
    )
    captured_operations = _token_counters(parallel_request_handler, ("api_key", _api_key))

    token_operations = [op for op in captured_operations if op["key"].endswith(":tokens")]
    assert token_operations
    assert all(op["increment_value"] == 7 for op in token_operations)


_JPEG_DATA_URI: Final = (
    "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIs"
    "IxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAI"
    "AQEAAD8AKp//2Q=="
)


_MP4_DATA_URI: Final = "data:video/mp4;base64,AAAAIGZ0eXBpc29t" + "A" * 4096

_SETTLE_PROJECT_ITPM: Final = ("model_per_project_itpm", "p-settle:titan-embed-image")


async def _settle_tpm_for_call(
    monkeypatch: pytest.MonkeyPatch,
    *,
    call_type: str,
    data: dict[str, object],
    usage: Usage,
    rate_limit_type: str = "total",
    failed_attempts: int = 0,
    counted: tuple[str, str] | None = None,
    media_models: tuple[str, ...] = ("bedrock/amazon.titan-embed-image-v1",),
) -> tuple[int, float]:
    _set_token_rate_limit_type(monkeypatch, rate_limit_type)
    api_key: Final = hash_token(f"sk-settle-{call_type}")
    cache: Final = DualCache()
    handler: Final = NativeLimiter(
        internal_usage_cache=InternalUsageCache(cache),
        media_model_resolver=lambda _model, _team_id: media_models,
    )
    await handler.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(
            api_key=api_key,
            tpm_limit=10_000,
            project_id="p-settle",
            project_metadata={"model_itpm_limit": {"titan-embed-image": 10_000}},
        ),
        cache=cache,
        data={**data, "litellm_call_id": "settle-call"},
        call_type=call_type,
    )
    stash: Final = get_request_stash()
    assert stash is not None and stash.reserved_tokens > 0
    response: Final = (
        EmbeddingResponse(model="titan-embed-image", usage=usage)
        if call_type == "aembedding"
        else ModelResponse(usage=usage)
    )
    callback_kwargs: Final = {
        "litellm_call_id": "settle-call",
        "call_type": call_type,
        "standard_logging_object": {"metadata": {"user_api_key_hash": api_key}},
    }
    for _ in range(failed_attempts):
        await handler.async_log_failure_event(
            kwargs=dict(callback_kwargs), response_obj=None, start_time=datetime.now(), end_time=datetime.now()
        )
    await _log_success(handler, dict(callback_kwargs), response)
    return stash.reserved_tokens, _tokens(handler, *(counted or ("api_key", api_key)))


@pytest.mark.asyncio
@pytest.mark.parametrize("rate_limit_type", ["total", "input"])
@pytest.mark.parametrize("embedding_input", [[_JPEG_DATA_URI], "a photo of scones"])
async def test_embedding_the_provider_reports_as_zero_tokens_keeps_its_tpm_reservation(
    monkeypatch: pytest.MonkeyPatch, embedding_input: list[str] | str, rate_limit_type: str
) -> None:
    reserved, counted = await _settle_tpm_for_call(
        monkeypatch,
        call_type="aembedding",
        data={"model": "titan-embed-image", "input": embedding_input},
        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        rate_limit_type=rate_limit_type,
    )

    assert counted == reserved


@pytest.mark.asyncio
async def test_embedding_served_after_a_failed_attempt_still_charges_its_tpm_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reserved, counted = await _settle_tpm_for_call(
        monkeypatch,
        call_type="aembedding",
        data={"model": "titan-embed-image", "input": [_JPEG_DATA_URI]},
        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        failed_attempts=1,
    )

    assert counted == reserved


@pytest.mark.asyncio
async def test_embedding_media_no_deployment_prices_refunds_its_tpm_reservation_when_reported_as_zero_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, counted = await _settle_tpm_for_call(
        monkeypatch,
        call_type="aembedding",
        data={"model": "titan-embed-image", "input": [_MP4_DATA_URI]},
        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        media_models=("bedrock/async_invoke/us.twelvelabs.marengo-embed-3-0-v1:0",),
    )

    assert counted == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("embedding_input", [["s3://notes " + "word " * 40], "gs://bucket/clip.mp4"])
async def test_embedding_text_starting_with_a_storage_link_keeps_its_tpm_charge_when_reported_as_zero_tokens(
    monkeypatch: pytest.MonkeyPatch, embedding_input: list[str] | str
) -> None:
    reserved, counted = await _settle_tpm_for_call(
        monkeypatch,
        call_type="aembedding",
        data={"model": "titan-embed-image", "input": embedding_input},
        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        media_models=("bedrock/async_invoke/us.twelvelabs.marengo-embed-3-0-v1:0",),
    )

    assert counted == reserved


@pytest.mark.asyncio
@pytest.mark.parametrize("total_tokens", [7, 0])
async def test_embedding_with_reported_tokens_settles_tpm_to_the_reported_count(
    monkeypatch: pytest.MonkeyPatch, total_tokens: int
) -> None:
    reserved, counted = await _settle_tpm_for_call(
        monkeypatch,
        call_type="aembedding",
        data={"model": "titan-embed-image", "input": [_JPEG_DATA_URI]},
        usage=Usage(prompt_tokens=7, completion_tokens=0, total_tokens=total_tokens),
    )

    assert counted == 7 != reserved


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("call_type", "data", "rate_limit_type"),
    [
        ("acompletion", {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}]}, "total"),
        ("aembedding", {"model": "titan-embed-image", "input": [_JPEG_DATA_URI]}, "output"),
    ],
)
async def test_zero_usage_still_refunds_the_tpm_reservation_outside_input_counted_embeddings(
    monkeypatch: pytest.MonkeyPatch, call_type: str, data: dict[str, object], rate_limit_type: str
) -> None:
    _, counted = await _settle_tpm_for_call(
        monkeypatch,
        call_type=call_type,
        data=data,
        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        rate_limit_type=rate_limit_type,
    )

    assert counted == 0


@pytest.mark.asyncio
async def test_embedding_the_provider_reports_as_zero_tokens_keeps_its_project_itpm_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reserved, counted = await _settle_tpm_for_call(
        monkeypatch,
        call_type="aembedding",
        data={"model": "titan-embed-image", "input": [_JPEG_DATA_URI]},
        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        rate_limit_type="output",
        counted=_SETTLE_PROJECT_ITPM,
    )

    assert counted == reserved


@pytest.mark.parametrize("origin", ["shadow_eval_judge", "autorouter_compaction"])
@pytest.mark.parametrize("rate_limit_type", ["input", "output", "total"])
@pytest.mark.asyncio
async def test_internal_call_origin_success_ops_are_skipped(monkeypatch, origin, rate_limit_type):
    """Foreground compaction charges the same scopes as ordinary caller traffic."""
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

    async def _charged(metadata: dict[str, Any]) -> list[dict[str, object]]:
        handler: Final = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(DualCache()))
        _set_token_rate_limit_type(monkeypatch, rate_limit_type)
        await _log_success(handler, _kwargs(metadata), response)
        return _token_counters(handler, ("api_key", hash_token("sk-internal-origin")), ("team", "compaction-team"))

    charged = await _charged({})
    skipped = await _charged({INTERNAL_CALL_ORIGIN_METADATA_KEY: origin})

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


_GROUPING_COUNTERS: Final = (
    _counter("api_key", "sk-abc", "requests"),
    _counter("team", "t1", "requests"),
    _counter("end_user", "u28551", "requests"),
)


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
@pytest.mark.asyncio
async def test_success_tpm_accounting_skips_team_model_pool_when_key_owns_model_tpm_limit(
    monkeypatch, key_metadata, charges_team_model_pool
):
    handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(DualCache()))
    _set_token_rate_limit_type(monkeypatch, "output")
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

    await _log_success(handler, kwargs, response)

    assert _tokens(handler, "model_per_key", f"{hash_token('sk-pool')}:test-model") == 50
    assert (_tokens(handler, "model_per_team", "t:test-model") == 50) is charges_team_model_pool


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
        handle_rate_limit_error(
            handler,
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
        handle_rate_limit_error(
            handler,
            response=over_limit,
            descriptors=[{"key": "model_per_key", "value": "gpt-4o-mini", "rate_limit": None}],
            requested_model="gpt-4o-mini",
        )

    assert exc_info.value.descriptor_key == "model_per_key"


def _resolve_alias_to_target(model: str) -> str | None:
    return "target" if model == "alias" else None


_CUSTOMER_SCOPES: Final = (
    ("customer_org", "org-1"),
    ("customer_team", "team-1"),
    ("customer_member", "member-1"),
    ("model_per_customer_org", "org-1:target"),
    ("model_per_customer_team", "team-1:target"),
    ("model_per_customer_member", "member-1:target"),
)


@pytest.mark.asyncio
async def test_tpm_settlement_targets_include_the_customer_levels_and_their_model_buckets() -> None:
    async def _settled(metadata: dict[str, object]) -> dict[tuple[str, str], float]:
        handler: Final = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(DualCache()))
        await _log_success(
            handler,
            {
                "standard_logging_object": {"metadata": {"user_api_key_hash": "hash", **metadata}},
                "litellm_params": {"metadata": {"model_group": "target"}},
                "model": "target",
            },
            ModelResponse(model="target", usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)),
        )
        return {scope: _tokens(handler, *scope) for scope in _CUSTOMER_SCOPES}

    assert await _settled(
        {
            "user_api_key_customer_org_id": "org-1",
            "user_api_key_customer_team_id": "team-1",
            "user_api_key_customer_member_id": "member-1",
        }
    ) == dict.fromkeys(_CUSTOMER_SCOPES, 150)
    assert await _settled({}) == dict.fromkeys(_CUSTOMER_SCOPES, 0)


@pytest.mark.parametrize(
    "key_metadata, charges_team_model_pool",
    [({}, True), ({"model_tpm_limit": {"target": 500}}, False)],
    ids=["no_key_override", "key_owns_target_tpm_limit"],
)
@pytest.mark.asyncio
async def test_success_tpm_accounting_charges_the_alias_target_bucket(
    monkeypatch: pytest.MonkeyPatch, key_metadata: dict[str, object], charges_team_model_pool: bool
) -> None:
    handler: Final = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache()), model_group_resolver=_resolve_alias_to_target
    )
    _set_token_rate_limit_type(monkeypatch, "output")
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

    await _log_success(handler, kwargs, response)

    assert _tokens(handler, "model_per_key", f"{hash_token('sk-alias-tpm')}:target") == 50
    assert _tokens(handler, "model_per_key", f"{hash_token('sk-alias-tpm')}:alias") == 0
    assert _tokens(handler, "model_per_team", "t:alias") == 0
    assert (_tokens(handler, "model_per_team", "t:target") == 50) is charges_team_model_pool


@pytest.mark.asyncio
async def test_success_tpm_accounting_keeps_the_admission_target_after_an_alias_reload() -> None:
    alias_map: Final[dict[str, str]] = {"alias": "target-a"}
    cache: Final = DualCache()
    handler: Final = NativeLimiter(internal_usage_cache=InternalUsageCache(cache), model_group_resolver=alias_map.get)
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

    admitted: Final = _tokens(handler, "model_per_key", f"{auth.api_key}:target-a")
    await _log_success(handler, kwargs, response)

    assert _tokens(handler, "model_per_key", f"{auth.api_key}:target-a") - admitted == 150 - stash.reserved_tokens
    assert _tokens(handler, "model_per_key", f"{auth.api_key}:target-b") == 0


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


def _ptu_tokens(handler: _PROXY_MaxParallelRequestsHandler) -> float:
    return _tokens(handler, "model_per_team_ptu", "t:test-model")


async def _ptu_settled(
    handler: _PROXY_MaxParallelRequestsHandler,
    monkeypatch: pytest.MonkeyPatch,
    response_obj: object,
    kwargs: dict | None = None,
    rate_limit_type: str = "output",
) -> float:
    _set_token_rate_limit_type(monkeypatch, rate_limit_type)
    before: Final = _ptu_tokens(handler)
    await _log_success(handler, _ptu_success_kwargs() if kwargs is None else kwargs, response_obj)
    return _ptu_tokens(handler) - before


@pytest.mark.asyncio
async def test_success_accounting_settles_the_ptu_counter_in_azure_normalized_tokens(monkeypatch: pytest.MonkeyPatch):
    """100 uncached input tokens plus 50 output tokens at a 4:1 output ratio are 300
    normalized tokens, which is what the share was sized in."""
    resolve, _ = _ptu_ceiling_for("t", "test-model", tpm_limit=500, ratio=4.0)
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache()), ptu_team_ceiling_resolver=resolve
    )
    response = _ptu_response(Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150))

    assert await _ptu_settled(handler, monkeypatch, response) == 300


@pytest.mark.asyncio
async def test_pass_through_usage_settles_the_ptu_counter_in_azure_normalized_tokens(monkeypatch: pytest.MonkeyPatch):
    """A pass-through response carries its usage beside the body rather than in it, and its 100
    input and 50 output tokens still settle at 300 normalized tokens, not a raw count."""
    resolve, _ = _ptu_ceiling_for("t", "test-model", tpm_limit=500, ratio=4.0)
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache()), ptu_team_ceiling_resolver=resolve
    )

    settled = await _ptu_settled(
        handler,
        monkeypatch,
        {"response": "upstream body was never parsed"},
        {
            **_ptu_success_kwargs(),
            "combined_usage_object": Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        },
    )

    assert settled == 300


@pytest.mark.asyncio
async def test_cached_input_is_not_charged_to_the_ptu_counter(monkeypatch: pytest.MonkeyPatch):
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

    assert await _ptu_settled(handler, monkeypatch, response) == 60 + 4 * 50


@pytest.mark.asyncio
async def test_cached_input_is_charged_at_the_models_cached_ratio(monkeypatch: pytest.MonkeyPatch):
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

    assert await _ptu_settled(handler, monkeypatch, response) == 60 + 4 + 4 * 50


@pytest.mark.asyncio
async def test_a_reservation_is_settled_even_after_the_teams_share_is_gone(monkeypatch: pytest.MonkeyPatch):
    """The share can be removed between admission and completion; the reserved tokens still
    come off the counter instead of standing in the window."""
    ceiling: dict[str, PTUTeamCeiling | None] = {
        "current": PTUTeamCeiling(
            model_group="test-model", tpm_limit=2000, output_to_input_ratio=4.0, cached_input_ratio=0.0
        )
    }
    cache = DualCache()
    handler = NativeLimiter(
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
    settled = await _ptu_settled(
        handler,
        monkeypatch,
        _ptu_response(Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)),
        rate_limit_type="total",
    )

    assert settled == 300 - stash.ptu_reserved_tokens


@pytest.mark.asyncio
async def test_embedding_the_provider_reports_as_zero_tokens_keeps_its_ptu_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolve, _ = _ptu_ceiling_for("t", "test-model", tpm_limit=2000, ratio=4.0)
    cache: Final = DualCache()
    handler: Final = NativeLimiter(internal_usage_cache=InternalUsageCache(cache), ptu_team_ceiling_resolver=resolve)

    await handler.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(api_key=hash_token("sk-ptu"), team_id="t"),
        cache=cache,
        data={"model": "test-model", "input": ["a photo of scones"]},
        call_type="aembedding",
    )
    stash: Final = get_request_stash()
    assert stash is not None and stash.ptu_reserved_tokens > 0
    await _ptu_settled(
        handler,
        monkeypatch,
        EmbeddingResponse(model="test-model", usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)),
        {**_ptu_success_kwargs(), "call_type": "aembedding"},
        rate_limit_type="total",
    )

    assert _ptu_tokens(handler) == stash.ptu_reserved_tokens


@pytest.mark.asyncio
async def test_a_reservation_is_settled_when_the_success_metadata_lost_the_team(monkeypatch: pytest.MonkeyPatch):
    """Admission reserved on the team's PTU counter, so settlement credits that scope even when the
    success event's logging metadata carries no team id, instead of leaving the reservation standing."""
    cache = DualCache()
    resolve, _ = _ptu_ceiling_for("t", "test-model", tpm_limit=2000, ratio=4.0)
    handler = NativeLimiter(internal_usage_cache=InternalUsageCache(cache), ptu_team_ceiling_resolver=resolve)
    key = UserAPIKeyAuth(api_key=hash_token("sk-ptu"), team_id="t")

    await handler.async_pre_call_hook(user_api_key_dict=key, cache=cache, data=_ptu_request(), call_type="acompletion")
    stash = get_request_stash()
    assert stash is not None
    assert stash.ptu_reserved_tokens > 300

    settled = await _ptu_settled(
        handler,
        monkeypatch,
        _ptu_response(Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)),
        {
            **_ptu_success_kwargs(),
            "standard_logging_object": {"metadata": {"user_api_key_hash": hash_token("sk-ptu")}},
        },
        "total",
    )

    assert settled == 300 - stash.ptu_reserved_tokens


@pytest.mark.asyncio
async def test_success_settles_the_scope_the_reservation_was_taken_on(monkeypatch: pytest.MonkeyPatch):
    """Settlement credits the scope admission reserved, at the ceiling held since admission, not one
    rebuilt from the name the request used, so a request naming the deployment by id cannot leave its
    reservation standing."""
    ceiling: dict[str, PTUTeamCeiling | None] = {
        "current": PTUTeamCeiling(
            model_group="test-model", tpm_limit=2000, output_to_input_ratio=4.0, cached_input_ratio=0.0
        )
    }
    cache = DualCache()
    handler = NativeLimiter(
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
    kwargs = _ptu_success_kwargs()
    kwargs["litellm_params"]["metadata"]["model_group"] = "shared-ptu"
    settled = await _ptu_settled(
        handler,
        monkeypatch,
        _ptu_response(Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)),
        kwargs,
        "total",
    )

    assert settled == 300 - stash.ptu_reserved_tokens
    assert _tokens(handler, "model_per_team_ptu", "t:shared-ptu") == 0


@pytest.mark.asyncio
async def test_usage_that_only_reports_a_total_charges_that_total_to_the_ptu_counter(monkeypatch: pytest.MonkeyPatch):
    resolve, _ = _ptu_ceiling_for("t", "test-model", tpm_limit=500, ratio=4.0)
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache()), ptu_team_ceiling_resolver=resolve
    )
    response = _ptu_response(Usage(prompt_tokens=0, completion_tokens=0, total_tokens=150))

    assert await _ptu_settled(handler, monkeypatch, response) == 150


@pytest.mark.asyncio
async def test_a_socket_turns_usage_charges_the_key_but_leaves_the_team_ptu_counter_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolve, _ = _ptu_ceiling_for("t", "test-model", tpm_limit=500, ratio=4.0)
    handler: Final = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache()), ptu_team_ceiling_resolver=resolve
    )

    settled: Final = await _ptu_settled(
        handler,
        monkeypatch,
        _responses_websocket_turn_events(total_tokens=150),
        {**_ptu_success_kwargs(), "call_type": CallTypes.aresponses_websocket.value},
        "total",
    )

    assert _tokens(handler, "api_key", hash_token("sk-ptu")) == 150
    assert settled == 0


@pytest.mark.asyncio
async def test_success_accounting_charges_no_ptu_counter_without_a_ceiling(monkeypatch: pytest.MonkeyPatch):
    resolve, _ = _ptu_ceiling_for("u", "test-model", tpm_limit=500, ratio=4.0)
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache()), ptu_team_ceiling_resolver=resolve
    )
    response = _ptu_response(Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150))

    assert await _ptu_settled(handler, monkeypatch, response) == 0


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
    handler: Final = NativeLimiter(internal_usage_cache=InternalUsageCache(cache))
    monkeypatch.setattr(parallel_request_limiter_v3_module, "registry_agent", lambda _: None)
    descriptors: Final = await _request_descriptors(
        handler, auth, {"model": "a2a/target", "litellm_session_id": "session"}
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
    agent_scopes: Final = [scope for scope in stash.reserved_scopes if scope[0] in ("agent", "agent_session")]
    admitted: Final = {scope: _tokens(handler, *scope) for scope in agent_scopes}
    _set_token_rate_limit_type(monkeypatch, "total")
    await _log_success(
        handler,
        {"standard_logging_object": {"metadata": {"agent_id": auth.invoked_agent_id, "session_id": "session"}}},
        ModelResponse(usage=Usage(prompt_tokens=2, completion_tokens=1, total_tokens=3)),
    )
    for scope in agent_scopes:
        assert _tokens(handler, *scope) - admitted[scope] == 3 - stash.reserved_tokens


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

    assert estimate_precise_input_tokens(handler, data, "on-openai", "acompletion") == (
        estimate_precise_input_tokens(handler, data, "on-azure", "acompletion") + OPENAI_CHAT_PDF_FILE_TOKENS
    )
    assert estimate_precise_input_tokens(handler, data, "on-openai", "anthropic_messages") == (
        estimate_precise_input_tokens(handler, data, "on-azure", "anthropic_messages")
    )


_MULTILINGUAL_REQUEST: Final[dict[str, object]] = {
    "messages": [{"role": "user", "content": "日本語のテキストとहिन्दी पाठ और русский текст. " * 20}]
}


def _project_itpm_estimate(router: Router, model: str) -> int:
    handler: Final = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache()),
        counted_models_resolver=lambda model, team_id: models_counted_for(model, router, team_id),
    )
    return estimate_precise_input_tokens(handler, _MULTILINGUAL_REQUEST, model, "acompletion")


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
    handler: Final = NativeLimiter(internal_usage_cache=InternalUsageCache(cache))
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
    assert await fallback_target_allowed(
        handler, auth, "fallback", {"messages": [{"role": "user", "content": "hello"}]}
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

    buckets: Final = tuple(f"{auth.api_key}:{model}" for model in ("fallback", "primary"))
    admitted: Final = {bucket: _tokens(handler, "model_per_key", bucket) for bucket in buckets}
    await _log_success(handler, kwargs, response)

    settled: Final = {bucket: _tokens(handler, "model_per_key", bucket) - admitted[bucket] for bucket in buckets}
    assert settled == {buckets[0]: 150, buckets[1]: -stash.reserved_tokens}


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
    assert await fallback_target_allowed(handler, auth, target, {"messages": [{"role": "user", "content": "hello"}]})

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

    buckets: Final = (f"{auth.api_key}:{target}", f"{auth.api_key}:{alias}")
    admitted: Final = {bucket: _tokens(handler, "model_per_key", bucket) for bucket in buckets}
    await _log_success(handler, kwargs, response)

    settled: Final = {bucket: _tokens(handler, "model_per_key", bucket) - admitted[bucket] for bucket in buckets}
    assert settled == {buckets[0]: 150 - stash.reserved_tokens, buckets[1]: 0}


@pytest.mark.asyncio
async def test_success_tpm_accounting_ignores_a_client_supplied_model_group_without_a_router_fallback() -> None:
    cache: Final = DualCache()
    handler: Final = NativeLimiter(internal_usage_cache=InternalUsageCache(cache))
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

    admitted: Final = _tokens(handler, "model_per_key", f"{auth.api_key}:primary")
    await _log_success(handler, kwargs, response)

    assert _tokens(handler, "model_per_key", f"{auth.api_key}:primary") - admitted == 150 - stash.reserved_tokens
    assert _tokens(handler, "model_per_key", f"{auth.api_key}:uncapped") == 0
