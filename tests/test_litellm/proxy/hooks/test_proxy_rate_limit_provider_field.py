"""
Regression tests for the "provider field missing" bug on proxy-side
rate-limit errors.

Background
----------
The proxy's internal rate-limit hooks (parallel_request_limiter,
parallel_request_limiter_v3, dynamic_rate_limiter, dynamic_rate_limiter_v3,
batch_rate_limiter, max_budget_limiter, max_iterations_limiter,
max_budget_per_session_limiter) all fire from ``async_pre_call_hook`` —
*before* :func:`litellm.get_llm_provider` runs anywhere else in the request
lifecycle.

Until now, those hooks raised a bare ``HTTPException(429, ...)`` which carries
no ``llm_provider`` / ``model`` attribute. Downstream:

- The Prometheus ``litellm_proxy_failed_requests_metric`` reads
  ``exception.llm_provider`` via ``_get_exception_class_name`` — it came back
  empty, so dashboards showed ``exception_class="HTTPException"`` with no
  provider attribution.
- Observability callbacks that ``isinstance(e, RateLimitError)`` for
  category routing missed these entirely.

The fix wraps every internal raise site in
:class:`ProxyRateLimitError` (an ``HTTPException`` *and* a
``litellm.RateLimitError``), and resolves ``model`` / ``llm_provider`` from
``data["model"]`` via :func:`get_llm_provider`. When the model is missing or
unparseable we fall back to ``llm_provider="litellm_proxy"`` so we never break
the request path with a second exception.

These tests pin both the happy path (provider correctly resolved) and the
fallback path (unknown model, missing model) for every limiter.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

import litellm
from litellm.caching.caching import DualCache
from litellm.exceptions import RateLimitError
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.batch_rate_limiter import (
    BatchFileUsage,
    _PROXY_BatchRateLimiter,
)
from litellm.proxy.hooks.dynamic_rate_limiter import _PROXY_DynamicRateLimitHandler
from litellm.proxy.hooks.dynamic_rate_limiter_v3 import (
    _PROXY_DynamicRateLimitHandlerV3,
)
from litellm.proxy.hooks.max_budget_limiter import _PROXY_MaxBudgetLimiter
from litellm.proxy.hooks.max_budget_per_session_limiter import (
    _PROXY_MaxBudgetPerSessionHandler,
)
from litellm.proxy.hooks.max_iterations_limiter import _PROXY_MaxIterationsHandler
from litellm.proxy.hooks.parallel_request_limiter import (
    _PROXY_MaxParallelRequestsHandler,
)
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    _PROXY_MaxParallelRequestsHandler_v3,
)
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.hooks.rate_limiter_utils import (
    PROXY_LLM_PROVIDER_FALLBACK,
    resolve_llm_provider_for_rate_limit,
)
from litellm.proxy.utils import InternalUsageCache
from litellm.types.agents import AgentResponse


# ---------------------------------------------------------------------------
# Helper class itself
# ---------------------------------------------------------------------------


class TestProxyRateLimitErrorClass:
    """Pin the dual ``HTTPException`` + ``RateLimitError`` shape."""

    def test_is_both_http_exception_and_rate_limit_error(self):
        e = ProxyRateLimitError(
            detail="boom",
            model="gpt-4o-mini",
            llm_provider="openai",
        )
        # FastAPI handler keys off HTTPException to render the 429.
        assert isinstance(e, HTTPException)
        # Prometheus / observability key off RateLimitError + .llm_provider.
        assert isinstance(e, RateLimitError)
        assert e.status_code == 429
        assert e.model == "gpt-4o-mini"
        assert e.llm_provider == "openai"
        # ProxyRateLimitError prefixes message via RateLimitError.__init__.
        assert "boom" in e.message
        assert e.detail == "boom"

    def test_dict_detail_is_stringified_for_message(self):
        # Some hooks pass a dict detail (e.g. dynamic_rate_limiter v1) — the
        # `message` attr (read by RateLimitError.__str__ and observability
        # callbacks) must still be a string.
        e = ProxyRateLimitError(
            detail={"error": "over rpm"},
            model="claude-3-5-sonnet",
            llm_provider="anthropic",
        )
        assert isinstance(e.message, str)
        assert "over rpm" in e.message

    def test_defaults_to_litellm_proxy_provider(self):
        e = ProxyRateLimitError(detail="x")
        assert e.llm_provider == PROXY_LLM_PROVIDER_FALLBACK
        assert e.model == ""

    def test_none_provider_normalized_to_fallback(self):
        e = ProxyRateLimitError(
            detail="x",
            model=None,
            llm_provider=None,
        )
        assert e.llm_provider == PROXY_LLM_PROVIDER_FALLBACK
        assert e.model == ""


class TestResolveLLMProviderForRateLimit:
    @pytest.mark.parametrize(
        "model, expected_provider",
        [
            ("gpt-4o-mini", "openai"),
            ("anthropic/claude-3-5-sonnet", "anthropic"),
            ("bedrock/meta.llama3-1-70b-instruct-v1:0", "bedrock"),
        ],
    )
    def test_known_models_resolve_provider(self, model, expected_provider):
        resolved_model, provider = resolve_llm_provider_for_rate_limit(model)
        assert provider == expected_provider
        assert resolved_model  # non-empty

    @pytest.mark.parametrize("model", [None, "", "totally-not-a-real-model-name"])
    def test_missing_or_unknown_model_falls_back(self, model):
        # Must never raise — the resolver wraps `get_llm_provider` defensively
        # because raising here would mask the rate-limit error we're trying
        # to surface to the user.
        # Pin llm_router to None so the alias-fallback path doesn't pick up
        # a router left behind by another test in the session.
        with patch("litellm.proxy.proxy_server.llm_router", None):
            resolved_model, provider = resolve_llm_provider_for_rate_limit(model)
        assert provider == PROXY_LLM_PROVIDER_FALLBACK
        # Resolver returns the input model verbatim on the unknown branch so
        # the `.model` attribute is never silently swapped to a different one.
        if not model:
            assert resolved_model == ""
        else:
            assert resolved_model == model

    def test_get_llm_provider_raising_is_swallowed(self):
        # If get_llm_provider itself blows up (unexpected error), we still
        # fall back rather than letting the secondary exception escape.
        # No router is registered in this test, so the alias-fallback path
        # also yields None and we land at PROXY_LLM_PROVIDER_FALLBACK.
        with patch.object(
            litellm,
            "get_llm_provider",
            side_effect=RuntimeError("boom"),
        ):
            with patch(
                "litellm.proxy.proxy_server.llm_router",
                None,
            ):
                resolved_model, provider = resolve_llm_provider_for_rate_limit(
                    "anything"
                )
        assert provider == PROXY_LLM_PROVIDER_FALLBACK
        assert resolved_model == "anything"

    def test_router_alias_resolves_to_underlying_provider(self):
        """
        Nearly every real LiteLLM proxy deployment uses router aliases:

            model_list:
              - model_name: tpm-locked
                litellm_params:
                  model: openai/gpt-4o-mini
                  ...

        ``litellm.get_llm_provider("tpm-locked")`` doesn't know about
        router aliases and raises. Before this fix the resolver fell
        through to ``"litellm_proxy"``, defeating the whole point of the
        ``llm_provider`` field on the rate-limit error. The alias path
        must look the deployment up in the router's ``model_list`` and
        resolve from its ``litellm_params.model``.
        """

        class _FakeRouter:
            model_list = [
                {
                    "model_name": "tpm-locked",
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini",
                        "api_key": "fake",
                    },
                }
            ]

        with patch(
            "litellm.proxy.proxy_server.llm_router",
            _FakeRouter(),
        ):
            resolved_model, provider = resolve_llm_provider_for_rate_limit("tpm-locked")
        assert provider == "openai", (
            f"Router-alias path must resolve through litellm_params.model, "
            f"not fall through to {PROXY_LLM_PROVIDER_FALLBACK!r}. Got "
            f"provider={provider!r}, model={resolved_model!r}."
        )
        # The resolved model should point at the underlying deployment so
        # downstream Prometheus labels / failure callbacks attribute the
        # 429 to the real upstream, not the alias.
        assert resolved_model == "gpt-4o-mini"

    def test_router_alias_with_multiple_deployments_uses_first(self):
        """
        When an alias maps to multiple deployments (the load-balancing
        case), the rate-limit error fired at the *alias* level is
        deployment-agnostic — we have no way of knowing which one would
        have been picked. Use the first deployment's underlying provider:
        every deployment under one alias should agree on provider in any
        sensible config, and 'first' is deterministic so the Prometheus
        label is stable.
        """

        class _FakeRouter:
            model_list = [
                {
                    "model_name": "claude-pool",
                    "litellm_params": {"model": "anthropic/claude-3-5-sonnet"},
                },
                {
                    "model_name": "claude-pool",
                    "litellm_params": {"model": "anthropic/claude-3-5-haiku"},
                },
            ]

        with patch(
            "litellm.proxy.proxy_server.llm_router",
            _FakeRouter(),
        ):
            _, provider = resolve_llm_provider_for_rate_limit("claude-pool")
        assert provider == "anthropic"

    def test_router_alias_unknown_falls_back(self):
        """
        Alias not in the router model_list — both lookups fail, so we
        land at the defensive ``litellm_proxy`` fallback rather than
        raising.
        """

        class _FakeRouter:
            model_list = [
                {
                    "model_name": "tpm-locked",
                    "litellm_params": {"model": "openai/gpt-4o-mini"},
                }
            ]

        with patch(
            "litellm.proxy.proxy_server.llm_router",
            _FakeRouter(),
        ):
            resolved_model, provider = resolve_llm_provider_for_rate_limit(
                "not-an-alias"
            )
        assert provider == PROXY_LLM_PROVIDER_FALLBACK
        assert resolved_model == "not-an-alias"

    def test_router_alias_with_malformed_deployment_falls_back(self):
        """
        A deployment in the router model_list with no usable
        ``litellm_params.model`` (or where ``get_llm_provider`` on the
        underlying string also raises) must not crash the resolver —
        fall through to the defensive fallback.
        """

        class _FakeRouter:
            model_list = [
                {"model_name": "broken", "litellm_params": {}},
                {"model_name": "broken", "litellm_params": {"model": ""}},
                {
                    "model_name": "broken",
                    "litellm_params": {"model": "nonsense-no-provider"},
                },
            ]

        with patch(
            "litellm.proxy.proxy_server.llm_router",
            _FakeRouter(),
        ):
            resolved_model, provider = resolve_llm_provider_for_rate_limit("broken")
        assert provider == PROXY_LLM_PROVIDER_FALLBACK
        assert resolved_model == "broken"


# ---------------------------------------------------------------------------
# parallel_request_limiter v1
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parallel_request_limiter_v1_populates_provider_when_at_rpm_limit():
    """
    Trip the per-key RPM cap and assert the raised exception carries
    ``model`` / ``llm_provider`` resolved from ``data["model"]``.
    """
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache())
    )
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-rl-test",
        max_parallel_requests=10,
        rpm_limit=1,
        tpm_limit=10,
    )
    data = {"model": "gpt-4o-mini"}

    # First request consumes the budget.
    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=DualCache(),
        data=data,
        call_type="completion",
    )

    with pytest.raises(HTTPException) as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=DualCache(),
            data=data,
            call_type="completion",
        )

    exc = exc_info.value
    assert exc.status_code == 429
    assert isinstance(exc, RateLimitError)
    assert exc.llm_provider == "openai"
    assert exc.model == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_parallel_request_limiter_v1_zero_limit_path_populates_provider():
    """
    When tpm_limit / rpm_limit is 0 the limiter takes the
    ``raise_rate_limit_error`` path. That path receives ``requested_model``
    via the call-site change and must pass it through.
    """
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache())
    )
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-rl-zero",
        max_parallel_requests=0,
        rpm_limit=10,
        tpm_limit=10,
    )

    with pytest.raises(HTTPException) as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=DualCache(),
            data={"model": "anthropic/claude-3-5-sonnet"},
            call_type="completion",
        )

    exc = exc_info.value
    assert exc.status_code == 429
    assert isinstance(exc, RateLimitError)
    assert exc.llm_provider == "anthropic"
    assert exc.model == "claude-3-5-sonnet"


@pytest.mark.asyncio
async def test_parallel_request_limiter_v1_global_limit_populates_provider():
    """global_max_parallel_requests path also threads the model through."""
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache())
    )
    user_api_key_dict = UserAPIKeyAuth(api_key="sk-global")

    # Pre-fill the global counter so the next call exceeds it.
    await handler.internal_usage_cache.async_set_cache(
        key="global_max_parallel_requests",
        value=5,
        local_only=True,
        litellm_parent_otel_span=None,
    )

    with pytest.raises(HTTPException) as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=DualCache(),
            data={
                "model": "bedrock/meta.llama3-1-70b-instruct-v1:0",
                "metadata": {"global_max_parallel_requests": 1},
            },
            call_type="completion",
        )

    exc = exc_info.value
    assert exc.status_code == 429
    assert exc.llm_provider == "bedrock"
    assert exc.model == "meta.llama3-1-70b-instruct-v1:0"


@pytest.mark.asyncio
async def test_parallel_request_limiter_v1_unknown_model_falls_back():
    """
    When ``data["model"]`` is unparseable, the resolver falls back to
    ``litellm_proxy`` — and crucially does *not* leak a secondary exception.
    """
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache())
    )
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-rl-unknown",
        max_parallel_requests=10,
        rpm_limit=1,
        tpm_limit=10,
    )
    data = {"model": "totally-not-a-real-model"}

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=DualCache(),
        data=data,
        call_type="completion",
    )

    with pytest.raises(HTTPException) as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=DualCache(),
            data=data,
            call_type="completion",
        )

    exc = exc_info.value
    assert exc.status_code == 429
    assert exc.llm_provider == PROXY_LLM_PROVIDER_FALLBACK
    # Resolver returns the input verbatim so we don't silently relabel the
    # model in the user-facing 429 detail.
    assert exc.model == "totally-not-a-real-model"


@pytest.mark.asyncio
async def test_parallel_request_limiter_v1_missing_model_falls_back():
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache())
    )
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-rl-no-model",
        max_parallel_requests=10,
        rpm_limit=1,
        tpm_limit=10,
    )

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=DualCache(),
        data={},
        call_type="completion",
    )

    with pytest.raises(HTTPException) as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=DualCache(),
            data={},
            call_type="completion",
        )

    exc = exc_info.value
    assert exc.llm_provider == PROXY_LLM_PROVIDER_FALLBACK
    assert exc.model == ""


# ---------------------------------------------------------------------------
# parallel_request_limiter v3
# ---------------------------------------------------------------------------


def _v3_over_limit_response(rate_limit_type: str = "requests") -> dict:
    return {
        "overall_code": "OVER_LIMIT",
        "statuses": [
            {
                "code": "OVER_LIMIT",
                "descriptor_key": "key",
                "current_limit": 1,
                "limit_remaining": -1,
                "rate_limit_type": rate_limit_type,
            }
        ],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model, expected_provider",
    [
        ("gpt-4o-mini", "openai"),
        ("anthropic/claude-3-5-sonnet", "anthropic"),
    ],
)
async def test_parallel_request_limiter_v3_populates_provider(model, expected_provider):
    handler = _PROXY_MaxParallelRequestsHandler_v3(
        internal_usage_cache=InternalUsageCache(DualCache())
    )

    descriptors = [{"key": "key", "value": "v", "rate_limit": {"requests_per_unit": 1}}]
    over = _v3_over_limit_response()

    with pytest.raises(HTTPException) as exc_info:
        handler._handle_rate_limit_error(
            response=over,
            descriptors=descriptors,
            requested_model=model,
        )

    exc = exc_info.value
    assert exc.status_code == 429
    assert isinstance(exc, RateLimitError)
    assert exc.llm_provider == expected_provider
    # v3 may strip the "anthropic/" prefix in the resolved model — accept
    # either; we only care that the provider field is correct and the model
    # is non-empty.
    assert exc.model


@pytest.mark.asyncio
async def test_parallel_request_limiter_v3_unknown_model_falls_back():
    handler = _PROXY_MaxParallelRequestsHandler_v3(
        internal_usage_cache=InternalUsageCache(DualCache())
    )
    descriptors = [{"key": "key", "value": "v", "rate_limit": {"requests_per_unit": 1}}]

    with pytest.raises(HTTPException) as exc_info:
        handler._handle_rate_limit_error(
            response=_v3_over_limit_response(),
            descriptors=descriptors,
            requested_model="totally-bogus",
        )

    assert exc_info.value.llm_provider == PROXY_LLM_PROVIDER_FALLBACK
    assert exc_info.value.model == "totally-bogus"


@pytest.mark.asyncio
async def test_parallel_request_limiter_v3_missing_model_falls_back():
    handler = _PROXY_MaxParallelRequestsHandler_v3(
        internal_usage_cache=InternalUsageCache(DualCache())
    )
    descriptors = [{"key": "key", "value": "v", "rate_limit": {"requests_per_unit": 1}}]

    with pytest.raises(HTTPException) as exc_info:
        handler._handle_rate_limit_error(
            response=_v3_over_limit_response(),
            descriptors=descriptors,
            requested_model=None,
        )

    assert exc_info.value.llm_provider == PROXY_LLM_PROVIDER_FALLBACK
    assert exc_info.value.model == ""


# ---------------------------------------------------------------------------
# dynamic_rate_limiter v1
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dynamic_rate_limiter_v1_tpm_zero_populates_provider():
    handler = _PROXY_DynamicRateLimitHandler(internal_usage_cache=DualCache())
    handler.check_available_usage = AsyncMock(return_value=(0, 5, 100, 5, 1))

    user_api_key_dict = UserAPIKeyAuth(api_key="sk-dyn")
    user_api_key_dict.metadata = {}

    with pytest.raises(HTTPException) as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=DualCache(),
            data={"model": "gpt-4o-mini"},
            call_type="completion",
        )

    exc = exc_info.value
    assert exc.status_code == 429
    assert isinstance(exc, RateLimitError)
    assert exc.llm_provider == "openai"
    assert exc.model == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_dynamic_rate_limiter_v1_rpm_zero_populates_provider():
    handler = _PROXY_DynamicRateLimitHandler(internal_usage_cache=DualCache())
    handler.check_available_usage = AsyncMock(return_value=(5, 0, 5, 100, 1))

    user_api_key_dict = UserAPIKeyAuth(api_key="sk-dyn")
    user_api_key_dict.metadata = {}

    with pytest.raises(HTTPException) as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=DualCache(),
            data={"model": "anthropic/claude-3-5-sonnet"},
            call_type="completion",
        )

    exc = exc_info.value
    assert exc.llm_provider == "anthropic"
    assert exc.model == "claude-3-5-sonnet"


@pytest.mark.asyncio
async def test_dynamic_rate_limiter_v1_unknown_model_falls_back():
    handler = _PROXY_DynamicRateLimitHandler(internal_usage_cache=DualCache())
    handler.check_available_usage = AsyncMock(return_value=(0, 5, 100, 5, 1))

    user_api_key_dict = UserAPIKeyAuth(api_key="sk-dyn")
    user_api_key_dict.metadata = {}

    with pytest.raises(HTTPException) as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=DualCache(),
            data={"model": "no-such-model"},
            call_type="completion",
        )

    assert exc_info.value.llm_provider == PROXY_LLM_PROVIDER_FALLBACK
    assert exc_info.value.model == "no-such-model"


# ---------------------------------------------------------------------------
# dynamic_rate_limiter v3 — exercise just the raise path via the helper, not
# the full Redis/Lua stack.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dynamic_rate_limiter_v3_model_capacity_path_populates_provider():
    """
    The v3 dynamic limiter has three raise sites: model_saturation_check,
    priority_model, and the fail-closed unknown-descriptor branch. We patch
    the atomic increment to short-circuit straight into the model_saturation
    path — that's the most common production trip — and confirm the
    raised exception carries provider info.
    """
    from litellm.types.router import ModelGroupInfo

    handler = _PROXY_DynamicRateLimitHandlerV3(internal_usage_cache=DualCache())
    handler.v3_limiter.atomic_check_and_increment_by_n = AsyncMock(
        return_value={
            "overall_code": "OVER_LIMIT",
            "statuses": [
                {
                    "code": "OVER_LIMIT",
                    "descriptor_key": "model_saturation_check",
                    "current_limit": 100,
                    "limit_remaining": 0,
                    "rate_limit_type": "requests",
                }
            ],
        }
    )
    handler._create_priority_based_descriptors = MagicMock(return_value=[])
    handler._create_model_tracking_descriptor = MagicMock(
        return_value={
            "key": "model_saturation_check",
            "value": "gpt-4o-mini",
            "rate_limit": {"requests_per_unit": 100},
        }
    )

    user_api_key_dict = UserAPIKeyAuth(api_key="sk-dyn-v3")
    user_api_key_dict.metadata = {}
    model_info = ModelGroupInfo(model_group="gpt-4o-mini", providers=["openai"])

    with pytest.raises(HTTPException) as exc_info:
        await handler._check_rate_limits(
            model="gpt-4o-mini",
            model_group_info=model_info,
            user_api_key_dict=user_api_key_dict,
            priority="default",
            saturation=1.0,
            data={"model": "gpt-4o-mini"},
        )

    exc = exc_info.value
    assert exc.status_code == 429
    assert isinstance(exc, RateLimitError)
    assert exc.llm_provider == "openai"
    assert exc.model == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_dynamic_rate_limiter_v3_unknown_descriptor_path_populates_provider():
    """Fail-closed unknown-descriptor branch must still attribute provider."""
    from litellm.types.router import ModelGroupInfo

    handler = _PROXY_DynamicRateLimitHandlerV3(internal_usage_cache=DualCache())
    handler.v3_limiter.atomic_check_and_increment_by_n = AsyncMock(
        return_value={
            "overall_code": "OVER_LIMIT",
            "statuses": [
                {
                    "code": "OVER_LIMIT",
                    "descriptor_key": "something_we_dont_handle",
                    "current_limit": 1,
                    "limit_remaining": 0,
                    "rate_limit_type": "requests",
                }
            ],
        }
    )
    handler._create_priority_based_descriptors = MagicMock(return_value=[])
    handler._create_model_tracking_descriptor = MagicMock(
        return_value={
            "key": "model_saturation_check",
            "value": "gpt-4o-mini",
            "rate_limit": {"requests_per_unit": 1},
        }
    )

    user_api_key_dict = UserAPIKeyAuth(api_key="sk-dyn-v3-unknown")
    user_api_key_dict.metadata = {}
    model_info = ModelGroupInfo(model_group="gpt-4o-mini", providers=["openai"])

    with pytest.raises(HTTPException) as exc_info:
        await handler._check_rate_limits(
            model="gpt-4o-mini",
            model_group_info=model_info,
            user_api_key_dict=user_api_key_dict,
            priority="default",
            saturation=1.0,
            data={"model": "gpt-4o-mini"},
        )

    assert exc_info.value.llm_provider == "openai"


# ---------------------------------------------------------------------------
# batch_rate_limiter
# ---------------------------------------------------------------------------


def _batch_over_limit_response() -> dict:
    return {
        "overall_code": "OVER_LIMIT",
        "statuses": [
            {
                "code": "OVER_LIMIT",
                "descriptor_key": "key",
                "current_limit": 10,
                "limit_remaining": -5,
                "rate_limit_type": "requests",
            }
        ],
    }


@pytest.mark.asyncio
async def test_batch_rate_limiter_populates_provider():
    """
    batch_rate_limiter trips when the file's request/token count exceeds the
    remaining window. The raise must thread `data["model"]` through the
    helper.
    """
    parallel_limiter = MagicMock()
    parallel_limiter.window_size = 60
    parallel_limiter._create_rate_limit_descriptors = MagicMock(
        return_value=[
            {"key": "key", "value": "v", "rate_limit": {"requests_per_unit": 10}}
        ]
    )
    parallel_limiter.atomic_check_and_increment_by_n = AsyncMock(
        return_value=_batch_over_limit_response()
    )

    handler = _PROXY_BatchRateLimiter(
        internal_usage_cache=InternalUsageCache(DualCache()),
        parallel_request_limiter=parallel_limiter,
    )

    with pytest.raises(HTTPException) as exc_info:
        await handler._check_and_increment_batch_counters(
            user_api_key_dict=UserAPIKeyAuth(api_key="sk-batch"),
            data={"model": "gpt-4o-mini"},
            batch_usage=BatchFileUsage(total_tokens=100, request_count=15),
        )

    exc = exc_info.value
    assert exc.status_code == 429
    assert isinstance(exc, RateLimitError)
    assert exc.llm_provider == "openai"
    assert exc.model == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_batch_rate_limiter_unknown_model_falls_back():
    parallel_limiter = MagicMock()
    parallel_limiter.window_size = 60
    parallel_limiter._create_rate_limit_descriptors = MagicMock(
        return_value=[
            {"key": "key", "value": "v", "rate_limit": {"requests_per_unit": 10}}
        ]
    )
    parallel_limiter.atomic_check_and_increment_by_n = AsyncMock(
        return_value=_batch_over_limit_response()
    )

    handler = _PROXY_BatchRateLimiter(
        internal_usage_cache=InternalUsageCache(DualCache()),
        parallel_request_limiter=parallel_limiter,
    )

    with pytest.raises(HTTPException) as exc_info:
        await handler._check_and_increment_batch_counters(
            user_api_key_dict=UserAPIKeyAuth(api_key="sk-batch"),
            data={"model": "fake-model-xyz"},
            batch_usage=BatchFileUsage(total_tokens=100, request_count=15),
        )

    assert exc_info.value.llm_provider == PROXY_LLM_PROVIDER_FALLBACK


# ---------------------------------------------------------------------------
# max_budget_limiter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_budget_limiter_populates_provider():
    handler = _PROXY_MaxBudgetLimiter()
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-budget",
        user_id="user-1",
        user_max_budget=10.0,
    )

    with patch(
        "litellm.proxy.proxy_server.get_current_spend",
        new=AsyncMock(return_value=10.0),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await handler.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=DualCache(),
                data={"model": "gpt-4o-mini"},
                call_type="completion",
            )

    exc = exc_info.value
    assert exc.status_code == 429
    assert isinstance(exc, RateLimitError)
    assert exc.llm_provider == "openai"
    assert exc.model == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_max_budget_limiter_no_model_falls_back():
    handler = _PROXY_MaxBudgetLimiter()
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-budget",
        user_id="user-1",
        user_max_budget=10.0,
    )

    with patch(
        "litellm.proxy.proxy_server.get_current_spend",
        new=AsyncMock(return_value=10.0),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await handler.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=DualCache(),
                data={},
                call_type="completion",
            )

    assert exc_info.value.llm_provider == PROXY_LLM_PROVIDER_FALLBACK
    assert exc_info.value.model == ""


# ---------------------------------------------------------------------------
# max_iterations_limiter
# ---------------------------------------------------------------------------


def _make_iter_agent(max_iterations: int) -> AgentResponse:
    return AgentResponse(
        agent_id="agent-iter",
        agent_name="iter-agent",
        litellm_params={"max_iterations": max_iterations},
        agent_card_params={"name": "iter-agent", "version": "1.0.0"},
    )


@pytest.mark.asyncio
async def test_max_iterations_limiter_populates_provider():
    local_cache = DualCache()
    handler = _PROXY_MaxIterationsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )
    user_api_key_dict = UserAPIKeyAuth(api_key="sk-iter", agent_id="agent-iter")

    with patch(
        "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry"
    ) as mock_registry:
        mock_registry.get_agent_by_id.return_value = _make_iter_agent(max_iterations=1)

        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={
                "model": "gpt-4o-mini",
                "metadata": {"session_id": "session-iter-1"},
            },
            call_type="completion",
        )

        with pytest.raises(HTTPException) as exc_info:
            await handler.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=local_cache,
                data={
                    "model": "gpt-4o-mini",
                    "metadata": {"session_id": "session-iter-1"},
                },
                call_type="completion",
            )

    exc = exc_info.value
    assert exc.status_code == 429
    assert isinstance(exc, RateLimitError)
    assert exc.llm_provider == "openai"
    assert exc.model == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_max_iterations_limiter_unknown_model_falls_back():
    local_cache = DualCache()
    handler = _PROXY_MaxIterationsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )
    user_api_key_dict = UserAPIKeyAuth(api_key="sk-iter", agent_id="agent-iter")

    with patch(
        "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry"
    ) as mock_registry:
        mock_registry.get_agent_by_id.return_value = _make_iter_agent(max_iterations=1)

        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={
                "model": "no-such-model",
                "metadata": {"session_id": "session-iter-2"},
            },
            call_type="completion",
        )

        with pytest.raises(HTTPException) as exc_info:
            await handler.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=local_cache,
                data={
                    "model": "no-such-model",
                    "metadata": {"session_id": "session-iter-2"},
                },
                call_type="completion",
            )

    assert exc_info.value.llm_provider == PROXY_LLM_PROVIDER_FALLBACK


# ---------------------------------------------------------------------------
# max_budget_per_session_limiter
# ---------------------------------------------------------------------------


def _make_session_budget_agent(max_budget: float) -> AgentResponse:
    return AgentResponse(
        agent_id="agent-session-budget",
        agent_name="session-budget-agent",
        litellm_params={"max_budget_per_session": max_budget},
        agent_card_params={"name": "session-budget-agent", "version": "1.0.0"},
    )


@pytest.mark.asyncio
async def test_max_budget_per_session_limiter_populates_provider():
    handler = _PROXY_MaxBudgetPerSessionHandler(
        internal_usage_cache=InternalUsageCache(DualCache())
    )
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-session-budget", agent_id="agent-session-budget"
    )

    with patch(
        "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry"
    ) as mock_registry:
        mock_registry.get_agent_by_id.return_value = _make_session_budget_agent(
            max_budget=1.0
        )
        with patch.object(
            handler, "_get_current_spend", new=AsyncMock(return_value=5.0)
        ):
            with pytest.raises(HTTPException) as exc_info:
                await handler.async_pre_call_hook(
                    user_api_key_dict=user_api_key_dict,
                    cache=DualCache(),
                    data={
                        "model": "anthropic/claude-3-5-sonnet",
                        "metadata": {"session_id": "session-budget-1"},
                    },
                    call_type="completion",
                )

    exc = exc_info.value
    assert exc.status_code == 429
    assert isinstance(exc, RateLimitError)
    assert exc.llm_provider == "anthropic"


@pytest.mark.asyncio
async def test_max_budget_per_session_limiter_unknown_model_falls_back():
    handler = _PROXY_MaxBudgetPerSessionHandler(
        internal_usage_cache=InternalUsageCache(DualCache())
    )
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-session-budget", agent_id="agent-session-budget"
    )

    with patch(
        "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry"
    ) as mock_registry:
        mock_registry.get_agent_by_id.return_value = _make_session_budget_agent(
            max_budget=1.0
        )
        with patch.object(
            handler, "_get_current_spend", new=AsyncMock(return_value=5.0)
        ):
            with pytest.raises(HTTPException) as exc_info:
                await handler.async_pre_call_hook(
                    user_api_key_dict=user_api_key_dict,
                    cache=DualCache(),
                    data={
                        "model": "no-such-model",
                        "metadata": {"session_id": "session-budget-2"},
                    },
                    call_type="completion",
                )

    assert exc_info.value.llm_provider == PROXY_LLM_PROVIDER_FALLBACK


# ---------------------------------------------------------------------------
# Prometheus integration: failure metric reads exception.llm_provider
# via _get_exception_class_name. With the fix, this returns
# "Openai.RateLimitError" instead of plain "HTTPException" for proxy-side
# 429s on a known model. Pin that contract — that's what dashboards see.
# ---------------------------------------------------------------------------


def test_prometheus_exception_class_name_back_compat_for_proxy_rate_limit_error():
    """
    `_get_exception_class_name` deliberately returns the literal string
    ``"HTTPException"`` for every ``ProxyRateLimitError`` instance so that
    pre-existing dashboards / alerts (which key off the historical value)
    keep working after the unified rate-limit error class landed in #27687.

    Provider attribution is now surfaced separately via the
    ``rate_limit_category`` / ``rate_limit_type`` labels — this test pins
    the back-compat shim itself.
    """
    from litellm.integrations.prometheus import PrometheusLogger

    exc = ProxyRateLimitError(
        detail="over limit",
        model="gpt-4o-mini",
        llm_provider="openai",
    )
    assert PrometheusLogger._get_exception_class_name(exc) == "HTTPException"

    # Same back-compat path even when the resolver fell back to litellm_proxy.
    exc_no_model = ProxyRateLimitError(detail="over limit")
    assert PrometheusLogger._get_exception_class_name(exc_no_model) == "HTTPException"


def test_prometheus_exception_class_name_back_compat_for_budget_exceeded_error():
    """
    The unified rate-limit work also attached ``.llm_provider`` to
    ``BudgetExceededError`` so callbacks get provider attribution from
    ``StandardLoggingPayload``. Without a back-compat short-circuit the
    provider-prefix step in ``_get_exception_class_name`` would silently
    flip the label from ``"BudgetExceededError"`` to e.g.
    ``"Openai.BudgetExceededError"`` and break dashboards keyed on the
    historical value. Pin the literal label here.
    """
    from litellm.integrations.prometheus import PrometheusLogger

    err = litellm.BudgetExceededError(
        current_cost=1.0,
        max_budget=0.5,
        llm_provider="openai",
    )
    assert PrometheusLogger._get_exception_class_name(err) == "BudgetExceededError"

    # Default (empty llm_provider) path — same literal label.
    err_no_provider = litellm.BudgetExceededError(current_cost=1.0, max_budget=0.5)
    assert (
        PrometheusLogger._get_exception_class_name(err_no_provider)
        == "BudgetExceededError"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-vv", "-x"]))
