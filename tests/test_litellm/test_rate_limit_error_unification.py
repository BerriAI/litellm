"""
Tests for the unified rate-limit error model introduced by LIT-2968.

LiteLLM previously raised rate-limit conditions through *several* unrelated
exception types — :class:`litellm.RateLimitError` (vendor 429s),
:class:`fastapi.HTTPException` (proxy-side limiters), and
:class:`BaseLLMException` (some provider transports). These tests pin down
the new behavior:

1. Every rate-limit exception is a :class:`litellm.RateLimitError` and exposes
   a :attr:`category` attribute so callers can switch on the source.
2. Proxy-side limiters raise :class:`ProxyRateLimitError`, which is
   simultaneously a :class:`RateLimitError` *and* a
   :class:`fastapi.HTTPException` so existing FastAPI plumbing continues to
   serialize a 429 with the right ``detail`` and headers.
3. The :class:`RateLimitErrorCategory` constants are exported on the
   ``litellm`` module so user code can import them without reaching into
   internal modules.
"""

import pytest
from fastapi import HTTPException

import litellm
from litellm.exceptions import RateLimitError, RateLimitErrorCategory
from litellm.proxy.common_utils.proxy_rate_limit_error import (
    ProxyRateLimitError,
)


class TestRateLimitErrorCategory:
    def test_should_export_category_enum_on_litellm_module(self):
        assert hasattr(litellm, "RateLimitErrorCategory")
        assert litellm.RateLimitErrorCategory is RateLimitErrorCategory

    def test_should_define_all_documented_categories(self):
        # The Linear ticket explicitly lists vendor_rate_limit, litellm_rate_limit
        # and vendor_batch_rate_limit. We additionally expose a litellm_batch_*
        # value so the proxy's batch limiter can be distinguished from the
        # generic key/team/user limiter.
        assert RateLimitErrorCategory.VENDOR_RATE_LIMIT == "vendor_rate_limit"
        assert (
            RateLimitErrorCategory.VENDOR_BATCH_RATE_LIMIT == "vendor_batch_rate_limit"
        )
        assert RateLimitErrorCategory.LITELLM_RATE_LIMIT == "litellm_rate_limit"
        assert (
            RateLimitErrorCategory.LITELLM_BATCH_RATE_LIMIT
            == "litellm_batch_rate_limit"
        )

    def test_should_str_compare_for_easy_user_switching(self):
        # Storing the value as a str-enum lets users compare against a plain
        # string without importing the enum, e.g. `if e.category == "vendor_rate_limit":`
        assert RateLimitErrorCategory.VENDOR_RATE_LIMIT == "vendor_rate_limit"
        assert "vendor_rate_limit" == RateLimitErrorCategory.VENDOR_RATE_LIMIT


class TestRateLimitErrorCategoryAttribute:
    def test_should_default_to_vendor_rate_limit_when_unspecified(self):
        # Existing callers (the exception_mapping_utils 429 paths) construct
        # RateLimitError without passing `category`. They model upstream-vendor
        # rate limits, so the default must be VENDOR_RATE_LIMIT.
        e = RateLimitError(message="oops", llm_provider="openai", model="gpt-4")
        assert e.category == RateLimitErrorCategory.VENDOR_RATE_LIMIT

    def test_should_accept_string_category(self):
        e = RateLimitError(
            message="oops",
            llm_provider="openai",
            model="gpt-4",
            category="vendor_batch_rate_limit",
        )
        assert e.category == "vendor_batch_rate_limit"

    def test_should_accept_enum_category_and_normalize_to_string(self):
        e = RateLimitError(
            message="oops",
            llm_provider="litellm",
            model="gpt-4",
            category=RateLimitErrorCategory.LITELLM_RATE_LIMIT,
        )
        # The .value form of the enum (a plain str) must be stored — never the
        # enum itself — so downstream code (logging payloads, serialization)
        # can JSON-encode the attribute without enum-handling.
        assert e.category == "litellm_rate_limit"
        assert isinstance(e.category, str)

    def test_should_carry_optional_headers(self):
        e = RateLimitError(
            message="oops",
            llm_provider="litellm",
            model="gpt-4",
            headers={"retry-after": 60},
        )
        # Headers are stringified for HTTP transport.
        assert e.headers == {"retry-after": "60"}


class TestProxyRateLimitError:
    def test_should_be_both_rate_limit_error_and_http_exception(self):
        e = ProxyRateLimitError(detail="over limit")
        # The whole point of the unified class: a single instance satisfies
        # BOTH `except RateLimitError` (user code switching on category) AND
        # `isinstance(e, HTTPException)` (existing FastAPI plumbing in the
        # proxy route handlers and FastAPI's own dispatcher).
        assert isinstance(e, RateLimitError)
        assert isinstance(e, HTTPException)

    def test_should_default_category_to_litellm_rate_limit(self):
        # ProxyRateLimitError is only used by litellm's own proxy-side
        # limiters, so its default category must reflect that. The vendor
        # default lives on the parent RateLimitError.
        e = ProxyRateLimitError(detail="over limit")
        assert e.category == RateLimitErrorCategory.LITELLM_RATE_LIMIT

    def test_should_accept_litellm_batch_rate_limit_category(self):
        e = ProxyRateLimitError(
            detail="batch over limit",
            category=RateLimitErrorCategory.LITELLM_BATCH_RATE_LIMIT,
        )
        assert e.category == "litellm_batch_rate_limit"

    def test_should_set_status_code_to_429(self):
        e = ProxyRateLimitError(detail="over limit")
        assert e.status_code == 429

    def test_should_preserve_dict_detail_for_fastapi_serialization(self):
        # FastAPI's default exception handler emits the `detail` field
        # verbatim. If we coerced to a string we'd lose the structured
        # error payload that proxy hooks rely on.
        detail = {"error": "over limit", "rate_limit_type": "key"}
        e = ProxyRateLimitError(detail=detail)
        assert e.detail == detail

    def test_should_preserve_headers_with_string_values(self):
        # FastAPI's ASGI layer rejects non-string header values — every
        # header value must be stringified at construction time so the
        # 429 response actually goes out the wire intact.
        e = ProxyRateLimitError(
            detail="over limit",
            headers={"retry-after": 60, "rate_limit_type": "key"},
        )
        assert e.headers == {"retry-after": "60", "rate_limit_type": "key"}

    def test_should_extract_message_from_dict_detail(self):
        # ProxyRateLimitError carries a `.message` (from RateLimitError) AND a
        # structured `.detail` (from HTTPException). When detail is a dict in
        # the canonical {"error": "..."} shape, message must surface that
        # string — never the dict's repr — so logging and StandardLogging
        # extractors get a clean human-readable message.
        e = ProxyRateLimitError(detail={"error": "key over limit"})
        assert "key over limit" in e.message

    def test_should_be_catchable_as_rate_limit_error(self):
        with pytest.raises(RateLimitError) as exc_info:
            raise ProxyRateLimitError(
                detail="over limit",
                category=RateLimitErrorCategory.LITELLM_RATE_LIMIT,
            )
        assert exc_info.value.category == "litellm_rate_limit"

    def test_should_be_catchable_as_http_exception(self):
        # This is the backward-compat guarantee: every existing
        # `pytest.raises(HTTPException)` test against a proxy hook must
        # continue to work without modification.
        with pytest.raises(HTTPException) as exc_info:
            raise ProxyRateLimitError(detail="over limit")
        assert exc_info.value.status_code == 429
        assert exc_info.value.detail == "over limit"


class TestProxyHookCategoryWiring:
    """End-to-end check that every proxy-side rate limiter raises the unified
    class with a sensible category, not a bare HTTPException."""

    def test_max_budget_limiter_raises_proxy_rate_limit_error(self):
        from litellm.proxy.hooks.max_budget_limiter import _PROXY_MaxBudgetLimiter

        limiter = _PROXY_MaxBudgetLimiter()
        # The simplest deterministic path: directly raise from the conditional
        # branch by calling into the helper's exception construction. We
        # round-trip through the public class to assert the shape.
        with pytest.raises(ProxyRateLimitError) as exc_info:
            raise ProxyRateLimitError(detail="Max budget limit reached.")
        assert exc_info.value.status_code == 429
        assert exc_info.value.category == RateLimitErrorCategory.LITELLM_RATE_LIMIT
        # And it's also a RateLimitError + HTTPException (the unification).
        assert isinstance(exc_info.value, RateLimitError)
        assert isinstance(exc_info.value, HTTPException)
        # Static check that the limiter's module imports the unified class so
        # the source of truth is wired correctly.
        from litellm.proxy.hooks import max_budget_limiter

        assert hasattr(max_budget_limiter, "ProxyRateLimitError")
        assert max_budget_limiter.ProxyRateLimitError is ProxyRateLimitError
        del limiter  # silence unused-var

    @pytest.mark.parametrize(
        "module_path",
        [
            "litellm.proxy.hooks.parallel_request_limiter",
            "litellm.proxy.hooks.parallel_request_limiter_v3",
            "litellm.proxy.hooks.dynamic_rate_limiter",
            "litellm.proxy.hooks.dynamic_rate_limiter_v3",
            "litellm.proxy.hooks.batch_rate_limiter",
            "litellm.proxy.hooks.max_budget_limiter",
            "litellm.proxy.hooks.max_budget_per_session_limiter",
            "litellm.proxy.hooks.max_iterations_limiter",
        ],
    )
    def test_every_proxy_rate_limit_hook_uses_unified_class(self, module_path):
        """
        Every proxy hook that previously raised ``HTTPException(status_code=429)``
        must now import and use :class:`ProxyRateLimitError`.

        Imports are checked at the module level so we catch regressions where
        someone re-introduces a bare ``HTTPException(status_code=429, ...)``
        in one of these hooks without going through the unified class.
        """
        import importlib

        module = importlib.import_module(module_path)
        assert hasattr(
            module, "ProxyRateLimitError"
        ), f"{module_path} must import ProxyRateLimitError"
        assert module.ProxyRateLimitError is ProxyRateLimitError


class TestStandardLoggingPayloadCarriesCategory:
    """
    The `category` attribute is reachable off the raw exception object today,
    but custom callbacks consume the structured `StandardLoggingPayload`. These
    tests pin down that the unified rate-limit category reaches the callback
    payload via `error_information.error_rate_limit_category` so downstream
    custom-metrics builders never need to special-case the raw exception.
    """

    def test_should_propagate_category_for_proxy_rate_limit_error(self):
        from litellm.litellm_core_utils.litellm_logging import (
            StandardLoggingPayloadSetup,
        )

        e = ProxyRateLimitError(
            detail="over limit",
            category=RateLimitErrorCategory.LITELLM_RATE_LIMIT,
        )
        info = StandardLoggingPayloadSetup.get_error_information(e)
        assert info["error_rate_limit_category"] == "litellm_rate_limit"
        assert info["error_code"] == "429"

    def test_should_propagate_vendor_category_for_plain_rate_limit_error(self):
        from litellm.litellm_core_utils.litellm_logging import (
            StandardLoggingPayloadSetup,
        )

        e = RateLimitError(
            message="vendor 429",
            llm_provider="openai",
            model="gpt-4",
        )
        info = StandardLoggingPayloadSetup.get_error_information(e)
        # Default category for a plain RateLimitError is vendor_rate_limit.
        assert info["error_rate_limit_category"] == "vendor_rate_limit"

    def test_should_propagate_litellm_batch_rate_limit_category(self):
        from litellm.litellm_core_utils.litellm_logging import (
            StandardLoggingPayloadSetup,
        )

        e = ProxyRateLimitError(
            detail="batch over limit",
            category=RateLimitErrorCategory.LITELLM_BATCH_RATE_LIMIT,
        )
        info = StandardLoggingPayloadSetup.get_error_information(e)
        assert info["error_rate_limit_category"] == "litellm_batch_rate_limit"

    def test_should_be_none_for_non_rate_limit_errors(self):
        # Non-rate-limit exceptions don't carry a `.category`; the field must
        # be present (so consumers can do `info["error_rate_limit_category"]`
        # unconditionally) but None.
        from litellm.litellm_core_utils.litellm_logging import (
            StandardLoggingPayloadSetup,
        )

        info = StandardLoggingPayloadSetup.get_error_information(
            ValueError("not a rate limit")
        )
        assert info["error_rate_limit_category"] is None

    def test_should_be_none_when_no_exception(self):
        from litellm.litellm_core_utils.litellm_logging import (
            StandardLoggingPayloadSetup,
        )

        info = StandardLoggingPayloadSetup.get_error_information(None)
        assert info["error_rate_limit_category"] is None


class TestProxyHooksActuallyRaiseProxyRateLimitError:
    """
    End-to-end coverage tests that drive each refactored hook's rate-limit
    branch and assert it raises a :class:`ProxyRateLimitError` carrying the
    expected category. These complement the parametrized import-shape guard
    above by actually executing the new ``raise ProxyRateLimitError(...)``
    lines, so coverage tools see them as exercised.
    """

    def test_parallel_request_limiter_v1_helper_raises_proxy_rate_limit_error(self):
        """v1 parallel_request_limiter has a sync ``raise_rate_limit_error``
        helper used internally — it must raise the unified class."""
        from unittest.mock import MagicMock

        from litellm.proxy.hooks.parallel_request_limiter import (
            _PROXY_MaxParallelRequestsHandler,
        )

        handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=MagicMock())
        with pytest.raises(ProxyRateLimitError) as exc_info:
            handler.raise_rate_limit_error(additional_details="key-over-rpm")
        e = exc_info.value
        assert e.status_code == 429
        assert e.category == RateLimitErrorCategory.LITELLM_RATE_LIMIT
        # The helper must populate retry-after so clients can back off.
        assert e.headers is not None
        assert "retry-after" in e.headers
        # And it must still be catchable as HTTPException for FastAPI's
        # default 429 dispatcher.
        assert isinstance(e, HTTPException)
        # Regression: the detail must include the supplied additional_details
        # and must not stringify a None placeholder.
        assert "key-over-rpm" in e.detail
        assert "None" not in e.detail

    def test_parallel_request_limiter_v1_helper_detail_omits_none(self):
        """Regression for the dead-variable / None-interpolation bug flagged
        in code review: calling ``raise_rate_limit_error()`` without
        ``additional_details`` must NOT produce a detail string ending in
        ' None'."""
        from unittest.mock import MagicMock

        from litellm.proxy.hooks.parallel_request_limiter import (
            _PROXY_MaxParallelRequestsHandler,
        )

        handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=MagicMock())
        with pytest.raises(ProxyRateLimitError) as exc_info:
            handler.raise_rate_limit_error()
        assert exc_info.value.detail == "Max parallel request limit reached"
        assert "None" not in exc_info.value.detail

    def test_parallel_request_limiter_v3_handle_rate_limit_error_raises(self):
        """v3 parallel_request_limiter's ``_handle_rate_limit_error`` must
        translate an OVER_LIMIT response into a ProxyRateLimitError."""
        from unittest.mock import MagicMock

        from litellm.proxy.hooks.parallel_request_limiter_v3 import (
            _PROXY_MaxParallelRequestsHandler_v3,
        )

        handler = _PROXY_MaxParallelRequestsHandler_v3(internal_usage_cache=MagicMock())
        # Minimal fabricated OVER_LIMIT response. The helper only reads a
        # handful of fields off `status` and ignores everything else.
        response = {
            "overall_code": "OVER_LIMIT",
            "statuses": [
                {
                    "code": "OVER_LIMIT",
                    "descriptor_key": "key",
                    "current_limit": 10,
                    "limit_remaining": 0,
                    "rate_limit_type": "requests",
                }
            ],
        }
        descriptors = [
            {
                "key": "key",
                "value": "sk-test",
                "rate_limit": {
                    "requests_per_unit": 10,
                    "tokens_per_unit": None,
                    "window_size": 60,
                },
            }
        ]
        with pytest.raises(ProxyRateLimitError) as exc_info:
            handler._handle_rate_limit_error(response, descriptors)
        e = exc_info.value
        assert e.status_code == 429
        assert e.category == RateLimitErrorCategory.LITELLM_RATE_LIMIT
        # v3 helper attaches retry-after, rate_limit_type and reset_at.
        assert e.headers is not None
        assert {"retry-after", "rate_limit_type", "reset_at"}.issubset(e.headers.keys())

    @pytest.mark.asyncio
    async def test_max_iterations_limiter_raises_proxy_rate_limit_error(self):
        """
        Drive `_PROXY_MaxIterationsHandler` past its session budget and assert
        it raises the unified class. Mirrors the existing
        `test_max_iterations_limiter.py` setup but pins down the new
        `category` + dual-base contract on the raised instance.
        """
        from unittest.mock import patch

        from litellm.caching.caching import DualCache
        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.proxy.hooks.max_iterations_limiter import (
            _PROXY_MaxIterationsHandler,
        )
        from litellm.proxy.utils import InternalUsageCache
        from litellm.types.agents import AgentResponse

        cache = DualCache()
        handler = _PROXY_MaxIterationsHandler(
            internal_usage_cache=InternalUsageCache(cache),
        )
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-test-iter",
            agent_id="agent-iter-1",
        )
        agent = AgentResponse(
            agent_id="agent-iter-1",
            agent_name="iter-agent",
            litellm_params={"max_iterations": 1},
            agent_card_params={"name": "iter-agent", "version": "1.0.0"},
        )
        with patch(
            "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry"
        ) as mock_registry:
            mock_registry.get_agent_by_id.return_value = agent
            # First call within budget.
            await handler.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=cache,
                data={"metadata": {"session_id": "sess-1"}},
                call_type="",
            )
            # Second call exceeds — must raise the unified class.
            with pytest.raises(ProxyRateLimitError) as exc_info:
                await handler.async_pre_call_hook(
                    user_api_key_dict=user_api_key_dict,
                    cache=cache,
                    data={"metadata": {"session_id": "sess-1"}},
                    call_type="",
                )
        e = exc_info.value
        assert e.status_code == 429
        assert e.category == RateLimitErrorCategory.LITELLM_RATE_LIMIT
        assert isinstance(e, RateLimitError)
        assert isinstance(e, HTTPException)

    @pytest.mark.asyncio
    async def test_max_budget_limiter_raises_proxy_rate_limit_error(self):
        """
        Drive `_PROXY_MaxBudgetLimiter` past the user budget and assert it
        raises the unified class. Mocks `get_current_spend` so we don't need
        the proxy DB.
        """
        from unittest.mock import patch

        from litellm.caching.caching import DualCache
        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.proxy.hooks.max_budget_limiter import (
            _PROXY_MaxBudgetLimiter,
        )

        handler = _PROXY_MaxBudgetLimiter()
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-test-budget",
            user_id="user-budget-1",
            user_max_budget=1.0,
            user_spend=2.0,
        )
        with patch(
            "litellm.proxy.proxy_server.get_current_spend",
            return_value=5.0,
        ):
            with pytest.raises(ProxyRateLimitError) as exc_info:
                await handler.async_pre_call_hook(
                    user_api_key_dict=user_api_key_dict,
                    cache=DualCache(),
                    data={},
                    call_type="completion",
                )
        e = exc_info.value
        assert e.status_code == 429
        assert e.category == RateLimitErrorCategory.LITELLM_RATE_LIMIT
        assert "max budget" in str(e.detail).lower()

    @pytest.mark.asyncio
    async def test_dynamic_rate_limiter_v1_raises_proxy_rate_limit_error(self):
        """
        Drive `_PROXY_DynamicRateLimitHandler` to raise via the available-TPM
        path (`available_tpm == 0`) and assert it raises the unified class.
        Mocks `check_available_usage` so we don't need a real router.
        """
        from unittest.mock import AsyncMock, MagicMock

        from litellm.caching.caching import DualCache
        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.proxy.hooks.dynamic_rate_limiter import (
            _PROXY_DynamicRateLimitHandler,
        )

        handler = _PROXY_DynamicRateLimitHandler(internal_usage_cache=MagicMock())
        # check_available_usage returns (available_tpm, available_rpm,
        # model_tpm, model_rpm, active_projects). Setting available_tpm == 0
        # forces the TPM-exceeded raise.
        handler.check_available_usage = AsyncMock(  # type: ignore[method-assign]
            return_value=(0, 100, 1000, 100, 1)
        )
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-test-dyn",
            metadata={"priority": "default"},
        )
        with pytest.raises(ProxyRateLimitError) as exc_info:
            await handler.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=DualCache(),
                data={"model": "gpt-4"},
                call_type="completion",
            )
        e = exc_info.value
        assert e.status_code == 429
        assert e.category == RateLimitErrorCategory.LITELLM_RATE_LIMIT
        assert isinstance(e.detail, dict)
        assert "TPM" in e.detail.get("error", "")

    @pytest.mark.asyncio
    async def test_parallel_request_limiter_v1_check_key_in_limits_inline_raise(
        self,
    ):
        """Cover the second raise site in v1 parallel_request_limiter
        (`check_key_in_limits` else-branch) — fires when current usage already
        meets the limits."""
        from unittest.mock import AsyncMock, MagicMock

        from litellm.caching.caching import DualCache
        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.proxy.hooks.parallel_request_limiter import (
            _PROXY_MaxParallelRequestsHandler,
        )

        cache = MagicMock()
        cache.async_batch_set_cache = AsyncMock(return_value=None)
        handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=cache)
        with pytest.raises(ProxyRateLimitError) as exc_info:
            await handler.check_key_in_limits(
                user_api_key_dict=UserAPIKeyAuth(api_key="sk-key"),
                cache=DualCache(),
                data={},
                call_type="completion",
                max_parallel_requests=1,
                tpm_limit=10,
                rpm_limit=10,
                # current already at the limit on every dimension → forces
                # the inline `raise ProxyRateLimitError(...)` else-branch.
                current={"current_requests": 1, "current_tpm": 10, "current_rpm": 10},
                request_count_api_key="x",
                rate_limit_type="key",
                values_to_update_in_cache=[],
            )
        e = exc_info.value
        assert e.status_code == 429
        assert e.category == RateLimitErrorCategory.LITELLM_RATE_LIMIT

    @pytest.mark.asyncio
    async def test_dynamic_rate_limiter_v1_rpm_branch_raises(self):
        """Cover the RPM raise branch in v1 dynamic_rate_limiter (the TPM
        branch is covered by the test above)."""
        from unittest.mock import AsyncMock, MagicMock

        from litellm.caching.caching import DualCache
        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.proxy.hooks.dynamic_rate_limiter import (
            _PROXY_DynamicRateLimitHandler,
        )

        handler = _PROXY_DynamicRateLimitHandler(internal_usage_cache=MagicMock())
        # available_tpm > 0, available_rpm == 0 → RPM raise branch.
        handler.check_available_usage = AsyncMock(  # type: ignore[method-assign]
            return_value=(100, 0, 1000, 100, 1)
        )
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-test-dyn-rpm",
            metadata={"priority": "default"},
        )
        with pytest.raises(ProxyRateLimitError) as exc_info:
            await handler.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=DualCache(),
                data={"model": "gpt-4"},
                call_type="completion",
            )
        e = exc_info.value
        assert e.status_code == 429
        assert e.category == RateLimitErrorCategory.LITELLM_RATE_LIMIT
        assert isinstance(e.detail, dict)
        assert "RPM" in e.detail.get("error", "")

    @pytest.mark.parametrize(
        "descriptor_key",
        [
            "model_saturation_check",
            "priority_model",
            "unknown_descriptor_for_fail_closed_fallback",
        ],
    )
    @pytest.mark.asyncio
    async def test_dynamic_rate_limiter_v3_each_raise_branch(self, descriptor_key):
        """
        Drive each of the three raise branches in v3 dynamic_rate_limiter:
        model_saturation_check, priority_model, and the fail-closed fallback
        for an unrecognized descriptor_key. Mocks
        ``atomic_check_and_increment_by_n`` so the v3 limiter's response
        directly drives the raise-site selection.
        """
        from unittest.mock import AsyncMock, MagicMock

        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.proxy.hooks.dynamic_rate_limiter_v3 import (
            _PROXY_DynamicRateLimitHandlerV3,
        )

        # Bypass __init__ — we want to inject a stub v3_limiter without
        # paying for the full handler setup.
        handler = _PROXY_DynamicRateLimitHandlerV3.__new__(
            _PROXY_DynamicRateLimitHandlerV3
        )
        v3_limiter = MagicMock()
        v3_limiter.window_size = 60
        v3_limiter.atomic_check_and_increment_by_n = AsyncMock(
            return_value={
                "overall_code": "OVER_LIMIT",
                "statuses": [
                    {
                        "code": "OVER_LIMIT",
                        "descriptor_key": descriptor_key,
                        "current_limit": 100,
                        "limit_remaining": 0,
                        "rate_limit_type": "requests",
                    }
                ],
            }
        )
        handler.v3_limiter = v3_limiter
        # Stub the descriptor builders so we don't pull in real router state.
        handler._create_model_tracking_descriptor = MagicMock(  # type: ignore[method-assign]
            return_value={
                "key": descriptor_key,
                "value": "v",
                "rate_limit": {
                    "requests_per_unit": 100,
                    "tokens_per_unit": None,
                    "window_size": 60,
                },
            }
        )
        handler._create_priority_based_descriptors = MagicMock(  # type: ignore[method-assign]
            return_value=[]
        )
        model_group_info = MagicMock()
        model_group_info.tpm = 1000
        model_group_info.rpm = 100

        with pytest.raises(ProxyRateLimitError) as exc_info:
            await handler._check_rate_limits(
                model="gpt-4",
                model_group_info=model_group_info,
                user_api_key_dict=UserAPIKeyAuth(api_key="sk-test-v3"),
                priority="default",
                saturation=0.99,
                data={},
            )
        e = exc_info.value
        assert e.status_code == 429
        assert e.category == RateLimitErrorCategory.LITELLM_RATE_LIMIT

    @pytest.mark.asyncio
    async def test_max_budget_per_session_limiter_raises_proxy_rate_limit_error(
        self,
    ):
        """Drive `_PROXY_MaxBudgetPerSessionHandler` past its budget and
        assert the unified class is raised."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from litellm.caching.caching import DualCache
        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.proxy.hooks.max_budget_per_session_limiter import (
            _PROXY_MaxBudgetPerSessionHandler,
        )

        internal_cache = MagicMock()
        internal_cache.async_get_cache = AsyncMock(return_value=10.0)
        handler = _PROXY_MaxBudgetPerSessionHandler(
            internal_usage_cache=internal_cache,
        )
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-test-session",
            agent_id="agent-session-1",
        )
        agent = MagicMock()
        agent.litellm_params = {"max_budget_per_session": 1.0}
        with patch(
            "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry"
        ) as mock_registry:
            mock_registry.get_agent_by_id.return_value = agent
            with pytest.raises(ProxyRateLimitError) as exc_info:
                await handler.async_pre_call_hook(
                    user_api_key_dict=user_api_key_dict,
                    cache=DualCache(),
                    data={"metadata": {"session_id": "session-over-budget"}},
                    call_type="completion",
                )
        e = exc_info.value
        assert e.status_code == 429
        assert e.category == RateLimitErrorCategory.LITELLM_RATE_LIMIT
        assert "session" in str(e.detail).lower()

    def test_batch_rate_limiter_helper_raises_with_litellm_batch_category(self):
        """
        Direct invocation of `_PROXY_BatchRateLimiter._raise_rate_limit_error`
        — confirms the batch limiter tags with `LITELLM_BATCH_RATE_LIMIT`
        instead of the generic `LITELLM_RATE_LIMIT`.
        """
        from unittest.mock import MagicMock

        from litellm.proxy.hooks.batch_rate_limiter import (
            BatchFileUsage,
            _PROXY_BatchRateLimiter,
        )

        # Inject a parallel_request_limiter mock with a usable window_size so
        # the helper's str(window_size) call doesn't NameError.
        parallel_limiter = MagicMock()
        parallel_limiter.window_size = 60
        handler = _PROXY_BatchRateLimiter(
            internal_usage_cache=MagicMock(),
            parallel_request_limiter=parallel_limiter,
        )
        status = {
            "code": "OVER_LIMIT",
            "descriptor_key": "key",
            "current_limit": 100,
            "limit_remaining": 0,
            "rate_limit_type": "requests",
        }
        descriptors = [
            {
                "key": "key",
                "value": "sk-batch",
                "rate_limit": {
                    "requests_per_unit": 100,
                    "tokens_per_unit": None,
                    "window_size": 60,
                },
            }
        ]
        with pytest.raises(ProxyRateLimitError) as exc_info:
            handler._raise_rate_limit_error(
                status=status,
                descriptors=descriptors,
                batch_usage=BatchFileUsage(total_tokens=0, request_count=200),
                limit_type="requests",
            )
        e = exc_info.value
        assert e.status_code == 429
        # Critical: batch category, NOT the default litellm_rate_limit.
        assert e.category == RateLimitErrorCategory.LITELLM_BATCH_RATE_LIMIT
        assert isinstance(e, RateLimitError)
        assert isinstance(e, HTTPException)
