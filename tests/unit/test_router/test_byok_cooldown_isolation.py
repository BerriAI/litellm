"""Regression tests for Issue #45312:
BYOK rate-limit (429) cooldown isolation and response-header retry-after cooldown capping.
"""

from unittest.mock import patch

import httpx
import pytest

import litellm
from litellm import Router
from litellm.constants import DEFAULT_MAX_COOLDOWN_TIME_SECONDS
from litellm.types.router import AllowedFailsPolicy


@pytest.fixture
def mock_httpx_429():
    """Mock httpx.AsyncClient.send to always return a 429 with a large retry-after."""

    async def _mock_send(self, request, **kw):
        return httpx.Response(
            429,
            headers={"retry-after": "86400"},
            request=request,
            json={"type": "error", "error": {"type": "rate_limit_error", "message": "usage limit"}},
        )

    with patch.object(httpx.AsyncClient, "send", new=_mock_send):
        yield


@pytest.mark.asyncio
async def test_byok_rate_limit_with_extra_headers_does_not_cool_down_shared_deployment(mock_httpx_429):
    """Client providing x-api-key via extra_headers (BYOK) hitting 429 must not cool down the shared deployment."""
    router = Router(
        num_retries=0,
        allowed_fails_policy=AllowedFailsPolicy(RateLimitErrorAllowedFails=1),
        model_list=[
            {
                "model_name": "claude",
                "model_info": {"id": "claude-dep-shared"},
                "litellm_params": {"model": "anthropic/claude-haiku-4-5", "api_key": "sk-ant-deployment-key"},
            }
        ],
    )

    # Client 1 makes requests that hit 429 twice (exceeding allowed fails)
    for _ in range(2):
        with pytest.raises(litellm.RateLimitError):
            await router.acompletion(
                model="claude",
                messages=[{"role": "user", "content": "hi"}],
                extra_headers={"x-api-key": "sk-client-tenant-1"},
            )

    # The shared deployment must remain uncooled
    active = await router.cooldown_cache.async_get_active_cooldowns(
        model_ids=["claude-dep-shared"], parent_otel_span=None
    )
    assert len(active) == 0, f"Shared deployment was cooled down: {active}"


@pytest.mark.asyncio
async def test_byok_rate_limit_with_client_api_key_does_not_cool_down_shared_deployment(mock_httpx_429):
    """Client providing dynamic api_key hitting 429 must not cool down the shared deployment."""
    router = Router(
        num_retries=0,
        allowed_fails_policy=AllowedFailsPolicy(RateLimitErrorAllowedFails=1),
        model_list=[
            {
                "model_name": "claude",
                "model_info": {"id": "claude-dep-shared"},
                "litellm_params": {"model": "anthropic/claude-haiku-4-5", "api_key": "sk-ant-deployment-key"},
            }
        ],
    )

    for _ in range(2):
        with pytest.raises(litellm.RateLimitError):
            await router.acompletion(
                model="claude",
                messages=[{"role": "user", "content": "hi"}],
                api_key="sk-client-tenant-1",
            )

    active = await router.cooldown_cache.async_get_active_cooldowns(
        model_ids=["claude-dep-shared"], parent_otel_span=None
    )
    assert len(active) == 0


@pytest.mark.asyncio
async def test_byok_rate_limit_with_authorization_header_does_not_cool_down_shared_deployment(mock_httpx_429):
    """Client providing OAuth/Bearer Authorization header hitting 429 must not cool down the shared deployment."""
    router = Router(
        num_retries=0,
        allowed_fails_policy=AllowedFailsPolicy(RateLimitErrorAllowedFails=1),
        model_list=[
            {
                "model_name": "claude",
                "model_info": {"id": "claude-dep-shared"},
                "litellm_params": {"model": "anthropic/claude-haiku-4-5", "api_key": "sk-ant-deployment-key"},
            }
        ],
    )

    for _ in range(2):
        with pytest.raises(litellm.RateLimitError):
            await router.acompletion(
                model="claude",
                messages=[{"role": "user", "content": "hi"}],
                extra_headers={"Authorization": "Bearer sk-ant-oauth-token-123"},
            )

    active = await router.cooldown_cache.async_get_active_cooldowns(
        model_ids=["claude-dep-shared"], parent_otel_span=None
    )
    assert len(active) == 0


@pytest.mark.asyncio
async def test_max_cooldown_time_caps_response_header_retry_after(mock_httpx_429):
    """When max_cooldown_time is configured, a response retry-after header must be capped to that value."""
    router = Router(
        num_retries=0,
        allowed_fails_policy=AllowedFailsPolicy(RateLimitErrorAllowedFails=1),
        max_cooldown_time=120,
        model_list=[
            {
                "model_name": "claude",
                "model_info": {"id": "claude-dep-shared"},
                "litellm_params": {"model": "anthropic/claude-haiku-4-5", "api_key": "sk-ant-deployment-key"},
            }
        ],
    )

    for _ in range(2):
        with pytest.raises(litellm.RateLimitError):
            await router.acompletion(
                model="claude",
                messages=[{"role": "user", "content": "hi"}],
            )

    active = await router.cooldown_cache.async_get_active_cooldowns(
        model_ids=["claude-dep-shared"], parent_otel_span=None
    )
    assert len(active) == 1
    assert active[0][1]["cooldown_time"] == 120


@pytest.mark.asyncio
async def test_default_max_cooldown_time_caps_excessive_retry_after(mock_httpx_429):
    """When max_cooldown_time is not set, a huge retry-after (e.g. 86400s) must be capped to DEFAULT_MAX_COOLDOWN_TIME_SECONDS."""
    router = Router(
        num_retries=0,
        allowed_fails_policy=AllowedFailsPolicy(RateLimitErrorAllowedFails=1),
        model_list=[
            {
                "model_name": "claude",
                "model_info": {"id": "claude-dep-shared"},
                "litellm_params": {"model": "anthropic/claude-haiku-4-5", "api_key": "sk-ant-deployment-key"},
            }
        ],
    )

    for _ in range(2):
        with pytest.raises(litellm.RateLimitError):
            await router.acompletion(
                model="claude",
                messages=[{"role": "user", "content": "hi"}],
            )

    active = await router.cooldown_cache.async_get_active_cooldowns(
        model_ids=["claude-dep-shared"], parent_otel_span=None
    )
    assert len(active) == 1
    assert active[0][1]["cooldown_time"] == DEFAULT_MAX_COOLDOWN_TIME_SECONDS
