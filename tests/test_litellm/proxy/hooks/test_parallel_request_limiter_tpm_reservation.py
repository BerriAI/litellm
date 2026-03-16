"""
Test that concurrent requests don't bypass TPM limits.

Fixes: https://github.com/BerriAI/litellm/issues/18730

The root cause was a TOCTOU race condition: TPM was only updated
after requests completed (in async_log_success_event), so concurrent
requests all saw current_tpm=0 and passed the check.

The fix reserves estimated tokens (max_tokens) at pre-call time and
reconciles with actual usage on completion.
"""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm import ModelResponse
from litellm.proxy.hooks.parallel_request_limiter import (
    _PROXY_MaxParallelRequestsHandler,
)
from litellm.proxy._types import UserAPIKeyAuth


class FakeInternalUsageCache:
    """In-memory cache for testing (mimics InternalUsageCache interface)."""

    def __init__(self):
        self._cache = {}

    async def async_get_cache(self, key=None, **kwargs):
        return self._cache.get(key)

    async def async_set_cache(self, key, value, **kwargs):
        self._cache[key] = value

    async def async_batch_get_cache(self, keys=None, **kwargs):
        if keys is None:
            return None
        return [self._cache.get(k) for k in keys]

    async def async_batch_set_cache(self, cache_list=None, **kwargs):
        if cache_list:
            for key, value in cache_list:
                self._cache[key] = value

    async def async_increment_cache(self, key=None, value=1, **kwargs):
        current = self._cache.get(key, 0)
        self._cache[key] = current + value


@pytest.fixture
def handler():
    cache = FakeInternalUsageCache()
    return _PROXY_MaxParallelRequestsHandler(internal_usage_cache=cache)


@pytest.fixture
def user_api_key_dict():
    return UserAPIKeyAuth(
        api_key="test-key-123",
        max_parallel_requests=100,
        tpm_limit=100,
        rpm_limit=100,
        parent_otel_span=None,
    )


def test_estimate_request_tokens():
    """_estimate_request_tokens should extract max_tokens from request data."""
    handler_cls = _PROXY_MaxParallelRequestsHandler
    assert handler_cls._estimate_request_tokens({"max_tokens": 500}) == 500
    assert handler_cls._estimate_request_tokens({"max_completion_tokens": 300}) == 300
    assert handler_cls._estimate_request_tokens({}) == 0
    assert handler_cls._estimate_request_tokens({"max_tokens": 0}) == 0
    assert handler_cls._estimate_request_tokens({"max_tokens": -1}) == 0
    # Bug 4: float values from JSON parsing should be handled
    assert handler_cls._estimate_request_tokens({"max_tokens": 100.0}) == 100
    assert handler_cls._estimate_request_tokens({"max_tokens": 99.9}) == 99


@pytest.mark.asyncio
async def test_concurrent_requests_respect_tpm_limit(handler, user_api_key_dict):
    """
    Reproduce issue #18730: 5 concurrent requests with max_tokens=50 each
    against a TPM limit of 100. Without the fix, all 5 pass (250 tokens).
    With the fix, only 2 should pass (100 tokens reserved).
    """
    cache = MagicMock()

    async def try_request():
        data = {"model": "gpt-4", "max_tokens": 50, "metadata": {}}
        try:
            await handler.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=cache,
                data=data,
                call_type="completion",
            )
            return "allowed"
        except Exception:
            return "rejected"

    results = await asyncio.gather(*[try_request() for _ in range(5)])
    allowed_count = results.count("allowed")
    rejected_count = results.count("rejected")

    # With TPM limit of 100 and max_tokens=50 per request,
    # only 2 requests should be allowed (2 * 50 = 100 <= TPM limit)
    assert allowed_count == 2, (
        f"Expected 2 requests allowed (2 * 50 = 100 TPM), "
        f"but got {allowed_count} allowed"
    )
    assert rejected_count == 3


@pytest.mark.asyncio
async def test_reservation_cleared_after_success(handler, user_api_key_dict):
    """After a request completes, its reservation should be reconciled
    with actual usage, freeing capacity for new requests."""
    cache = MagicMock()

    # First request: reserves 50 tokens
    data = {"model": "gpt-4", "max_tokens": 50, "metadata": {}}
    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type="completion",
    )

    # Simulate successful completion with only 20 actual tokens
    response = MagicMock(spec=ModelResponse)
    response.usage = MagicMock()
    response.usage.total_tokens = 20

    kwargs = {
        "litellm_params": {
            "metadata": {
                "user_api_key": "test-key-123",
                "_reserved_estimated_tokens": 50,
            }
        }
    }
    await handler.async_log_success_event(
        kwargs=kwargs,
        response_obj=response,
        start_time=None,
        end_time=None,
    )

    # Now a second request should be able to use the freed capacity
    # (100 TPM - 20 actual = 80 remaining, so 50 should fit)
    data2 = {"model": "gpt-4", "max_tokens": 50, "metadata": {}}
    try:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data=data2,
            call_type="completion",
        )
        second_allowed = True
    except Exception:
        second_allowed = False

    assert second_allowed, "Second request should be allowed after first completed with lower actual usage"


@pytest.mark.asyncio
async def test_reservation_released_on_failure(handler, user_api_key_dict):
    """If a request fails, its reserved tokens should be released."""
    cache = MagicMock()

    # Reserve 50 tokens
    data = {"model": "gpt-4", "max_tokens": 50, "metadata": {}}
    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type="completion",
    )

    # Simulate failure — should release the 50 reserved tokens
    kwargs = {
        "litellm_params": {
            "metadata": {
                "user_api_key": "test-key-123",
                "_reserved_estimated_tokens": 50,
            }
        },
        "exception": "Connection error",
    }
    await handler.async_log_failure_event(
        kwargs=kwargs,
        response_obj=None,
        start_time=None,
        end_time=None,
    )

    # After failure release, should be able to use the full 100 TPM again
    data2 = {"model": "gpt-4", "max_tokens": 90, "metadata": {}}
    try:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data=data2,
            call_type="completion",
        )
        allowed = True
    except Exception:
        allowed = False

    assert allowed, "Request should be allowed after failed request released its reservation"


@pytest.mark.asyncio
async def test_first_request_exceeding_tpm_limit_rejected(handler, user_api_key_dict):
    """Bug 1: A single first request with max_tokens > tpm_limit should be
    rejected even when the cache has no prior entry."""
    cache = MagicMock()

    # tpm_limit is 100, but request wants 200 tokens
    data = {"model": "gpt-4", "max_tokens": 200, "metadata": {}}
    with pytest.raises(Exception) as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data=data,
            call_type="completion",
        )
    assert exc_info.value.status_code == 429
