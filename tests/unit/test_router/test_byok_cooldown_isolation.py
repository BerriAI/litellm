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


def test_is_client_credential_rate_limit_edge_cases():
    """Unit tests for is_client_credential_rate_limit covering all branch conditions."""
    from litellm.router_utils.cooldown_handlers import is_client_credential_rate_limit

    # Non-429 errors must not be treated as client credential rate limits
    assert not is_client_credential_rate_limit({}, exception_status=500)
    assert not is_client_credential_rate_limit({}, exception_status="502")

    # RateLimitError with no client credentials returns False
    rate_err = litellm.RateLimitError(message="rate limited", model="claude", llm_provider="anthropic")
    assert not is_client_credential_rate_limit({}, exception_status=429, exception=rate_err)

    # Direct model_call_details flags
    assert is_client_credential_rate_limit({"is_clientside_credential": True}, exception_status=429)
    assert is_client_credential_rate_limit({"has_forwarded_client_auth": True}, exception_status=429)

    # Flags inside litellm_params
    assert is_client_credential_rate_limit(
        {"litellm_params": {"is_clientside_credential": True}}, exception_status=429
    )
    assert is_client_credential_rate_limit(
        {"litellm_params": {"has_forwarded_client_auth": True}}, exception_status=429
    )
    assert is_client_credential_rate_limit(
        {"litellm_params": {"model_info": {"is_clientside_credential": True}}}, exception_status=429
    )
    assert is_client_credential_rate_limit(
        {"litellm_params": {"model_info": {"original_model_id": "dep-original"}}}, exception_status=429
    )
    assert is_client_credential_rate_limit(
        {"litellm_params": {"metadata": {"is_clientside_credential": True}}}, exception_status=429
    )
    assert is_client_credential_rate_limit(
        {"litellm_params": {"proxy_server_request": {"credential_fields": ["api_key"]}}},
        exception_status="429",
    )


def test_is_clientside_credential_and_dynamic_params():
    """Unit tests for is_clientside_credential and get_dynamic_litellm_params."""
    from litellm.router_utils.clientside_credential_handler import (
        get_dynamic_litellm_params,
        is_clientside_credential,
    )

    # Non-mapping input
    assert not is_clientside_credential(None)  # type: ignore[arg-type]
    assert not is_clientside_credential("invalid")  # type: ignore[arg-type]

    # Explicit flag
    assert is_clientside_credential({"has_forwarded_client_auth": True})

    # Header variations
    assert is_clientside_credential({"headers": {"Authorization": "Bearer token"}})
    assert is_clientside_credential({"extra_headers": {"X-Goog-Api-Key": "key"}})
    assert is_clientside_credential({"default_headers": {"ocp-apim-subscription-key": "sub-key"}})
    assert not is_clientside_credential({"headers": {"User-Agent": "litellm-client"}})

    # get_dynamic_litellm_params forwards client credential headers
    base_params = {"model": "claude"}
    updated = get_dynamic_litellm_params(
        base_params, {"extra_headers": {"x-api-key": "secret", "unrelated": "val"}}
    )
    assert updated.get("extra_headers", {}).get("x-api-key") == "secret"


def test_fallback_handler_cooldown_capping_and_bypass():
    """Unit tests for fallback handler retry-after capping and BYOK cooldown bypass."""
    from unittest.mock import MagicMock, patch

    from litellm.router_utils.fallback_event_handlers import _trigger_cooldown_for_failed_deployment

    mock_router = MagicMock()
    mock_router.max_cooldown_time = 60
    mock_router.get_model_info.return_value = {"litellm_params": {}}

    rate_err = litellm.RateLimitError(message="rate limited", model="claude", llm_provider="anthropic")
    setattr(rate_err, "failed_deployment_id", "dep-1")
    setattr(rate_err, "response", MagicMock(headers={"retry-after": "100000"}))

    # BYOK 429 must return early without setting cooldown
    _trigger_cooldown_for_failed_deployment(
        litellm_router=mock_router,
        kwargs={"original_exception": rate_err},
        exception=rate_err,
        model_call_details={"is_clientside_credential": True},
    )
    mock_router.set_deployment_cooldown.assert_not_called()

    # Shared failure with excessive retry-after should cap to max_cooldown_time
    with patch(
        "litellm.router_utils.fallback_event_handlers.set_cooldown_deployments"
    ) as mock_set_cooldown:
        _trigger_cooldown_for_failed_deployment(
            litellm_router=mock_router,
            kwargs={"original_exception": rate_err},
            exception=rate_err,
            model_call_details={},
        )
        assert mock_set_cooldown.called
        call_kwargs = mock_set_cooldown.call_args.kwargs
        assert call_kwargs.get("time_to_cooldown") == 60

