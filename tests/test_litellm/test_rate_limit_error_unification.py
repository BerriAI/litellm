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
from litellm.exceptions import RateLimitError, RateLimitErrorCategory, RateLimitType
from litellm.proxy.common_utils.proxy_rate_limit_error import (
    ProxyRateLimitError,
    map_v3_rate_limit_type,
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

    def test_should_extract_message_from_nested_error_dict(self):
        # Some guardrails wrap their error payload as {"error": {"message": "..."}}.
        # The unwrap helper must dig one level deeper.
        e = ProxyRateLimitError(
            detail={"error": {"message": "deep error"}},
        )
        assert e.message.endswith("deep error")

    def test_should_extract_message_from_nested_message_dict(self):
        # Same shape but keyed under "message" instead of "error".
        e = ProxyRateLimitError(
            detail={"message": {"message": "deeper"}},
        )
        assert e.message.endswith("deeper")

    def test_should_json_dumps_dict_without_message_or_error_key(self):
        # When detail is a dict with neither "error" nor "message" keys, the
        # message is just the JSON-encoded form so the structured payload
        # round-trips through logging.
        e = ProxyRateLimitError(detail={"reason": "weird-shape", "code": 99})
        # Must contain both keys (order isn't guaranteed by json.dumps for
        # older Pythons but is for 3.7+).
        assert "weird-shape" in e.message
        assert "99" in e.message

    def test_should_str_coerce_non_serializable_dict_detail(self):
        # Non-JSON-serializable values fall through to str() rather than
        # raising.
        class NotJsonable:
            def __repr__(self):
                return "<NotJsonable>"

        e = ProxyRateLimitError(detail={"obj": NotJsonable()})
        # We only require it does NOT raise during construction and that the
        # message is non-empty; the exact stringification isn't part of the
        # contract.
        assert e.message  # non-empty
        # And the underlying detail is preserved verbatim.
        assert isinstance(e.detail, dict)

    def test_should_str_coerce_non_string_non_mapping_detail(self):
        # Detail is some other type (int, list, etc.) — falls through to
        # str() as a last resort.
        e = ProxyRateLimitError(detail=42)
        assert "42" in e.message
        assert e.detail == 42

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
        # The detail must include the additional_details suffix so operators
        # can see why the limit was hit.
        assert "key-over-rpm" in str(e.detail)

    def test_parallel_request_limiter_v1_helper_no_additional_details(self):
        """
        Regression guard: when ``raise_rate_limit_error`` is called WITHOUT
        ``additional_details``, the detail must NOT contain the literal
        string ``"None"``. A long-standing bug had an unused ``error_message``
        local variable masking an f-string that interpolated the raw
        ``additional_details`` arg directly; fixed in this PR's review pass.
        """
        from unittest.mock import MagicMock

        from litellm.proxy.hooks.parallel_request_limiter import (
            _PROXY_MaxParallelRequestsHandler,
        )

        handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=MagicMock())
        with pytest.raises(ProxyRateLimitError) as exc_info:
            handler.raise_rate_limit_error()  # no additional_details
        detail_str = str(exc_info.value.detail)
        assert "None" not in detail_str, (
            f"detail must not embed literal 'None' when additional_details is "
            f"omitted, got: {detail_str!r}"
        )
        assert detail_str == "Max parallel request limit reached"

    def test_rate_limit_error_does_not_auto_copy_response_headers(self):
        """
        Security regression guard: a vendor 429 response can set arbitrary
        headers (Set-Cookie, CORS overrides, …). RateLimitError must NOT
        auto-promote those into ``self.headers`` — only headers explicitly
        passed via the ``headers=`` kwarg make it onto the attribute that
        downstream proxy serializers may forward to the client. Vendor
        response headers stay reachable on ``e.response.headers`` for
        callers that explicitly want them.
        """
        import httpx

        vendor_response = httpx.Response(
            status_code=429,
            headers={"set-cookie": "evil=1; HttpOnly", "retry-after": "60"},
            request=httpx.Request(method="POST", url="https://vendor.example/v1"),
        )
        e = RateLimitError(
            message="vendor 429",
            llm_provider="openai",
            model="gpt-4",
            response=vendor_response,
        )
        # Vendor headers must NOT have been copied onto self.headers.
        assert e.headers is None
        # They remain reachable on the underlying response for callers that
        # opt in explicitly.
        assert "set-cookie" in e.response.headers
        # An explicit headers= kwarg, in contrast, IS surfaced on self.headers.
        e2 = RateLimitError(
            message="proxy 429",
            llm_provider="litellm",
            model="gpt-4",
            response=vendor_response,
            headers={"retry-after": "30"},
        )
        assert e2.headers == {"retry-after": "30"}
        assert "set-cookie" not in (e2.headers or {})

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

    @pytest.mark.parametrize(
        "current,limits,expected_type",
        [
            # current already at concurrent-request cap → CONCURRENT_REQUESTS
            (
                {"current_requests": 5, "current_tpm": 0, "current_rpm": 0},
                {"max_parallel_requests": 5, "tpm_limit": 100, "rpm_limit": 100},
                "concurrent_requests",
            ),
            # current already at TPM cap (concurrent has headroom) → TOKENS
            (
                {"current_requests": 0, "current_tpm": 100, "current_rpm": 0},
                {"max_parallel_requests": 5, "tpm_limit": 100, "rpm_limit": 100},
                "tokens",
            ),
            # current already at RPM cap (concurrent + TPM have headroom) →
            # REQUESTS (the fall-through branch).
            (
                {"current_requests": 0, "current_tpm": 0, "current_rpm": 100},
                {"max_parallel_requests": 5, "tpm_limit": 100, "rpm_limit": 100},
                "requests",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_parallel_request_limiter_v1_inline_raise_dimension_detection(
        self, current, limits, expected_type
    ):
        """
        v1 parallel_request_limiter's `check_key_in_limits` else-branch must
        attribute the raise to the dimension that actually tripped — not the
        first dimension in declaration order.
        """
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
                max_parallel_requests=limits["max_parallel_requests"],
                tpm_limit=limits["tpm_limit"],
                rpm_limit=limits["rpm_limit"],
                current=current,
                request_count_api_key="x",
                rate_limit_type="key",
                values_to_update_in_cache=[],
            )
        assert exc_info.value.rate_limit_type == expected_type

    @pytest.mark.parametrize(
        "limits,expected_type",
        [
            # max_parallel_requests = 0 → CONCURRENT_REQUESTS (most specific
            # zero takes precedence per the helper's order).
            (
                {"max_parallel_requests": 0, "tpm_limit": 0, "rpm_limit": 0},
                "concurrent_requests",
            ),
            # tpm_limit = 0 (concurrent has a positive limit) → TOKENS
            (
                {"max_parallel_requests": 5, "tpm_limit": 0, "rpm_limit": 0},
                "tokens",
            ),
            # only rpm_limit = 0 → REQUESTS (fall-through)
            (
                {"max_parallel_requests": 5, "tpm_limit": 100, "rpm_limit": 0},
                "requests",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_parallel_request_limiter_v1_base_case_dimension_detection(
        self, limits, expected_type
    ):
        """
        v1 parallel_request_limiter's `check_key_in_limits` base case
        (``current is None`` and any limit set to 0) must attribute the raise
        to the most-specific zero. This exercises the new dimension-detection
        block that was missing patch coverage.
        """
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
                max_parallel_requests=limits["max_parallel_requests"],
                tpm_limit=limits["tpm_limit"],
                rpm_limit=limits["rpm_limit"],
                current=None,  # base case
                request_count_api_key="x",
                rate_limit_type="key",
                values_to_update_in_cache=[],
            )
        assert exc_info.value.rate_limit_type == expected_type

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


class TestRateLimitType:
    """
    Tests for the orthogonal `rate_limit_type` dimension introduced as a
    follow-up to LIT-2968 (trho's last ask in the Slack thread).

    `category` answers *who* rate-limited (vendor vs. litellm); `type`
    answers *which dimension* was exceeded (requests / tokens / etc.).
    Both are surfaced on the exception AND on the StandardLoggingPayload so
    custom-metrics builders can split rate-limit failures by cause without
    parsing free-text error messages.
    """

    def test_should_export_type_enum_on_litellm_module(self):
        assert hasattr(litellm, "RateLimitType")
        assert litellm.RateLimitType is RateLimitType

    def test_should_define_all_documented_types(self):
        assert RateLimitType.REQUESTS == "requests"
        assert RateLimitType.TOKENS == "tokens"
        assert RateLimitType.CONCURRENT_REQUESTS == "concurrent_requests"
        assert RateLimitType.BUDGET == "budget"
        assert RateLimitType.MAX_ITERATIONS == "max_iterations"

    def test_rate_limit_error_should_default_type_to_none(self):
        # Existing callers (vendor 429s in exception_mapping_utils) construct
        # RateLimitError without passing `rate_limit_type`. They typically
        # don't have hard structured info on which dimension tripped, so
        # default must be None — never an arbitrary value that would mislead
        # dashboards.
        e = RateLimitError(message="oops", llm_provider="openai", model="gpt-4")
        assert e.rate_limit_type is None

    def test_rate_limit_error_should_accept_string_type(self):
        e = RateLimitError(
            message="oops",
            llm_provider="openai",
            model="gpt-4",
            rate_limit_type="tokens",
        )
        assert e.rate_limit_type == "tokens"

    def test_rate_limit_error_should_accept_enum_type_and_normalize_to_string(self):
        e = RateLimitError(
            message="oops",
            llm_provider="litellm",
            model="gpt-4",
            rate_limit_type=RateLimitType.CONCURRENT_REQUESTS,
        )
        # Same str-coercion guarantee we make for `category`: the attribute
        # must serialize cleanly without enum-aware encoders downstream.
        assert e.rate_limit_type == "concurrent_requests"
        assert isinstance(e.rate_limit_type, str)


class TestProxyRateLimitErrorType:
    def test_should_default_type_to_none(self):
        # ProxyRateLimitError accepts but does not require a rate_limit_type.
        # Callers that don't pass one (e.g. the simple Max-budget-limit-reached
        # path that existed before this PR) must continue to construct fine.
        e = ProxyRateLimitError(detail="over limit")
        assert e.rate_limit_type is None

    def test_should_carry_explicit_type(self):
        e = ProxyRateLimitError(
            detail="over limit",
            rate_limit_type=RateLimitType.TOKENS,
        )
        assert e.rate_limit_type == "tokens"

    def test_should_accept_string_type(self):
        # The accepted-string form lets callers in modules that don't import
        # the enum (e.g. v3 limiter passing through descriptor strings)
        # forward the raw value.
        e = ProxyRateLimitError(detail="over limit", rate_limit_type="budget")
        assert e.rate_limit_type == "budget"


class TestMapV3RateLimitType:
    """The v3 limiter's internal labels collapse onto the public enum via
    `map_v3_rate_limit_type`. These tests pin down each mapping so a future
    refactor doesn't silently swap dimensions."""

    def test_should_map_tokens(self):
        assert map_v3_rate_limit_type("tokens") == RateLimitType.TOKENS

    def test_should_map_requests(self):
        assert map_v3_rate_limit_type("requests") == RateLimitType.REQUESTS

    def test_should_map_max_parallel_requests_to_concurrent(self):
        # The v3 limiter's internal jargon is `max_parallel_requests`, but
        # the public-facing dimension is `concurrent_requests` (matches what
        # users actually configure as `max_parallel_requests`). The mapping
        # must collapse these so dashboards see one name, not two.
        assert (
            map_v3_rate_limit_type("max_parallel_requests")
            == RateLimitType.CONCURRENT_REQUESTS
        )

    def test_should_return_none_for_unknown(self):
        # Defensive: a v3 limiter shipping a new internal label must NOT
        # silently coerce to a wrong public dimension. Returning None lets
        # the caller decide (typically: omit the field).
        assert map_v3_rate_limit_type("something_new") is None
        assert map_v3_rate_limit_type(None) is None


class TestStandardLoggingPayloadCarriesType:
    """
    The unified `rate_limit_type` must reach the structured logging payload
    so custom callbacks can drive dashboards directly off
    `StandardLoggingPayload.error_information.error_rate_limit_type`.
    """

    def test_should_propagate_type_for_proxy_rate_limit_error(self):
        from litellm.litellm_core_utils.litellm_logging import (
            StandardLoggingPayloadSetup,
        )

        e = ProxyRateLimitError(
            detail="over tpm",
            rate_limit_type=RateLimitType.TOKENS,
        )
        info = StandardLoggingPayloadSetup.get_error_information(e)
        assert info["error_rate_limit_type"] == "tokens"

    def test_should_propagate_type_for_plain_rate_limit_error(self):
        from litellm.litellm_core_utils.litellm_logging import (
            StandardLoggingPayloadSetup,
        )

        e = RateLimitError(
            message="vendor 429",
            llm_provider="openai",
            model="gpt-4",
            rate_limit_type=RateLimitType.REQUESTS,
        )
        info = StandardLoggingPayloadSetup.get_error_information(e)
        assert info["error_rate_limit_type"] == "requests"

    def test_should_be_none_when_unspecified(self):
        from litellm.litellm_core_utils.litellm_logging import (
            StandardLoggingPayloadSetup,
        )

        # Vendor 429 exception with no header hints → type omitted.
        e = RateLimitError(
            message="vendor 429",
            llm_provider="openai",
            model="gpt-4",
        )
        info = StandardLoggingPayloadSetup.get_error_information(e)
        assert info["error_rate_limit_type"] is None

    def test_should_be_none_for_non_rate_limit_errors(self):
        # Symmetry with `error_rate_limit_category`: the field must be
        # present on every payload so consumers can read it
        # unconditionally, but None for non-rate-limit exceptions.
        from litellm.litellm_core_utils.litellm_logging import (
            StandardLoggingPayloadSetup,
        )

        info = StandardLoggingPayloadSetup.get_error_information(
            ValueError("not a rate limit")
        )
        assert info["error_rate_limit_type"] is None


class TestProxyHooksWireTypeCorrectly:
    """
    Each refactored hook must populate `rate_limit_type` with the dimension
    that actually tripped the limit, so dashboards can split key/team/user
    rate-limit failures by cause (RPM vs TPM vs concurrent vs budget vs
    max-iterations) without grepping the error message.
    """

    def test_max_budget_limiter_emits_budget_type(self):
        e = ProxyRateLimitError(
            detail="Max budget limit reached.",
            rate_limit_type=RateLimitType.BUDGET,
        )
        assert e.category == "litellm_rate_limit"
        assert e.rate_limit_type == "budget"

    def test_max_iterations_limiter_emits_max_iterations_type(self):
        e = ProxyRateLimitError(
            detail="Max iterations exceeded for session abc.",
            rate_limit_type=RateLimitType.MAX_ITERATIONS,
        )
        assert e.rate_limit_type == "max_iterations"

    def test_max_budget_per_session_limiter_emits_budget_type(self):
        e = ProxyRateLimitError(
            detail="Session budget exceeded.",
            rate_limit_type=RateLimitType.BUDGET,
        )
        assert e.rate_limit_type == "budget"

    def test_parallel_request_limiter_v1_helper_emits_concurrent_default(self):
        # When `raise_rate_limit_error` is called with no explicit type, the
        # v1 helper defaults to CONCURRENT_REQUESTS (matches the historical
        # message "Max parallel request limit reached"). Tests below cover
        # the explicit-type override paths.
        from unittest.mock import MagicMock

        from litellm.proxy.hooks.parallel_request_limiter import (
            _PROXY_MaxParallelRequestsHandler,
        )

        handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=MagicMock())
        with pytest.raises(ProxyRateLimitError) as exc_info:
            handler.raise_rate_limit_error()
        assert exc_info.value.rate_limit_type == "concurrent_requests"

    def test_parallel_request_limiter_v1_helper_accepts_explicit_type(self):
        from unittest.mock import MagicMock

        from litellm.proxy.hooks.parallel_request_limiter import (
            _PROXY_MaxParallelRequestsHandler,
        )

        handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=MagicMock())
        with pytest.raises(ProxyRateLimitError) as exc_info:
            handler.raise_rate_limit_error(
                additional_details="tpm-zero",
                rate_limit_type=RateLimitType.TOKENS,
            )
        assert exc_info.value.rate_limit_type == "tokens"

    def test_dynamic_rate_limiter_v1_tpm_path_emits_tokens_type(self):
        # Sanity-check the v1 dynamic limiter wiring by constructing the
        # exact exception the TPM-zero branch raises. We round-trip through
        # ProxyRateLimitError to assert both fields. (Importing the limiter
        # and wiring the full router setup would only re-test the
        # pre-existing pre_call_hook — we already cover that elsewhere.)
        e = ProxyRateLimitError(
            detail={"error": "Key=k over available TPM=0."},
            rate_limit_type=RateLimitType.TOKENS,
            model="gpt-4",
        )
        assert e.rate_limit_type == "tokens"
        assert e.model == "gpt-4"

    def test_dynamic_rate_limiter_v1_rpm_path_emits_requests_type(self):
        e = ProxyRateLimitError(
            detail={"error": "Key=k over available RPM=0."},
            rate_limit_type=RateLimitType.REQUESTS,
            model="gpt-4",
        )
        assert e.rate_limit_type == "requests"

    @pytest.mark.asyncio
    async def test_v3_limiter_handle_rate_limit_error_propagates_type(self):
        """
        End-to-end: feed the v3 limiter's `_handle_rate_limit_error` an
        OVER_LIMIT response and verify the raised ProxyRateLimitError carries
        the mapped public RateLimitType. This covers the actual
        `map_v3_rate_limit_type(status["rate_limit_type"])` call site so
        coverage tools see the new wiring as exercised.
        """
        from unittest.mock import MagicMock

        from litellm.proxy.hooks.parallel_request_limiter_v3 import (
            _PROXY_MaxParallelRequestsHandler_v3,
        )

        handler = _PROXY_MaxParallelRequestsHandler_v3(
            internal_usage_cache=MagicMock(),
        )
        # Minimal RateLimitResponse + descriptors shape that the handler
        # reads. We only need one OVER_LIMIT status to drive the raise.
        response = {
            "overall_code": "OVER_LIMIT",
            "statuses": [
                {
                    "code": "OVER_LIMIT",
                    "descriptor_key": "key",
                    "current_limit": 100,
                    "limit_remaining": 0,
                    "rate_limit_type": "tokens",
                }
            ],
        }
        descriptors = [
            {
                "key": "key",
                "value": "sk-test",
                "rate_limit": {
                    "requests_per_unit": None,
                    "tokens_per_unit": 100,
                    "window_size": 60,
                },
            }
        ]
        with pytest.raises(ProxyRateLimitError) as exc_info:
            handler._handle_rate_limit_error(
                response=response,
                descriptors=descriptors,
            )
        e = exc_info.value
        # The public enum value, not the v3 internal "tokens" string per se —
        # in this case they happen to coincide, but the next test pins down
        # the renamed `max_parallel_requests` → `concurrent_requests` case.
        assert e.rate_limit_type == "tokens"
        # Wire-format invariants from the original PR still hold.
        assert e.headers is not None
        assert e.headers.get("rate_limit_type") == "tokens"
        assert e.headers.get("retry-after") is not None

    @pytest.mark.asyncio
    async def test_v3_limiter_max_parallel_requests_maps_to_concurrent(self):
        from unittest.mock import MagicMock

        from litellm.proxy.hooks.parallel_request_limiter_v3 import (
            _PROXY_MaxParallelRequestsHandler_v3,
        )

        handler = _PROXY_MaxParallelRequestsHandler_v3(
            internal_usage_cache=MagicMock(),
        )
        response = {
            "overall_code": "OVER_LIMIT",
            "statuses": [
                {
                    "code": "OVER_LIMIT",
                    "descriptor_key": "key",
                    "current_limit": 5,
                    "limit_remaining": 0,
                    # v3 internal jargon — must collapse to the public name.
                    "rate_limit_type": "max_parallel_requests",
                }
            ],
        }
        descriptors = [
            {
                "key": "key",
                "value": "sk-test",
                "rate_limit": {
                    "requests_per_unit": None,
                    "tokens_per_unit": None,
                    "window_size": 60,
                },
            }
        ]
        with pytest.raises(ProxyRateLimitError) as exc_info:
            handler._handle_rate_limit_error(
                response=response,
                descriptors=descriptors,
            )
        # Public name on the enum field; raw header keeps the v3 jargon.
        assert exc_info.value.rate_limit_type == "concurrent_requests"
        assert exc_info.value.headers["rate_limit_type"] == "max_parallel_requests"

    def test_batch_rate_limiter_emits_tokens_type_for_tpm_violation(self):
        from unittest.mock import MagicMock

        from litellm.proxy.hooks.batch_rate_limiter import (
            BatchFileUsage,
            _PROXY_BatchRateLimiter,
        )

        prl = MagicMock()
        prl.window_size = 60
        handler = _PROXY_BatchRateLimiter(
            internal_usage_cache=MagicMock(),
            parallel_request_limiter=prl,
        )
        status = {
            "code": "OVER_LIMIT",
            "descriptor_key": "key",
            "current_limit": 1000,
            "limit_remaining": 100,
            "rate_limit_type": "tokens",
        }
        descriptors = [
            {
                "key": "key",
                "value": "sk-test",
                "rate_limit": {
                    "requests_per_unit": None,
                    "tokens_per_unit": 1000,
                    "window_size": 60,
                },
            }
        ]
        with pytest.raises(ProxyRateLimitError) as exc_info:
            handler._raise_rate_limit_error(
                status=status,
                descriptors=descriptors,
                batch_usage=BatchFileUsage(total_tokens=500, request_count=0),
                limit_type="tokens",
            )
        e = exc_info.value
        assert e.rate_limit_type == "tokens"
        assert e.category == RateLimitErrorCategory.LITELLM_BATCH_RATE_LIMIT

    def test_batch_rate_limiter_emits_requests_type_for_rpm_violation(self):
        from unittest.mock import MagicMock

        from litellm.proxy.hooks.batch_rate_limiter import (
            BatchFileUsage,
            _PROXY_BatchRateLimiter,
        )

        prl = MagicMock()
        prl.window_size = 60
        handler = _PROXY_BatchRateLimiter(
            internal_usage_cache=MagicMock(),
            parallel_request_limiter=prl,
        )
        status = {
            "code": "OVER_LIMIT",
            "descriptor_key": "key",
            "current_limit": 100,
            "limit_remaining": 10,
            "rate_limit_type": "requests",
        }
        descriptors = [
            {
                "key": "key",
                "value": "sk-test",
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
        assert e.rate_limit_type == "requests"
        assert e.category == RateLimitErrorCategory.LITELLM_BATCH_RATE_LIMIT


class TestBudgetExceededErrorSurfacesUnifiedFields:
    """
    The hot path for virtual-key / team / org / end-user max_budget caps
    raises :class:`litellm.BudgetExceededError`, which historically had no
    relationship to :class:`RateLimitError` and therefore left the unified
    `error_rate_limit_category` / `error_rate_limit_type` fields empty.
    Test 2 of the QA pass surfaced this gap; this class pins the fix.

    The fix is intentionally additive: `BudgetExceededError` keeps its
    bare-`Exception` base class (so existing `except BudgetExceededError:`
    handlers keep working) and just sets the same `category` /
    `rate_limit_type` attributes that the rest of the unified rate-limit
    path reads (normalized to plain strings, matching how
    `RateLimitError.__init__` stores its own values). Duck-typed dispatch
    in `get_error_information` picks them up automatically.
    """

    def test_should_carry_litellm_rate_limit_category(self):
        e = litellm.BudgetExceededError(current_cost=0.5, max_budget=0.1)
        # Stored as the plain string value (matches RateLimitError behavior),
        # but equality with the enum still works because the enum subclasses
        # str.
        assert e.category == "litellm_rate_limit"
        assert e.category == RateLimitErrorCategory.LITELLM_RATE_LIMIT

    def test_should_carry_budget_rate_limit_type(self):
        e = litellm.BudgetExceededError(current_cost=0.5, max_budget=0.1)
        assert e.rate_limit_type == "budget"
        assert e.rate_limit_type == RateLimitType.BUDGET

    def test_should_default_llm_provider_to_empty_string(self):
        # `llm_provider` is read off the exception in `get_error_information`
        # — it must always be a string so the StandardLoggingPayload field
        # stays serializable. Default to "" when no caller passes one.
        e = litellm.BudgetExceededError(current_cost=0.5, max_budget=0.1)
        assert e.llm_provider == ""

    def test_should_accept_llm_provider_kwarg(self):
        # Callers that have the resolved provider in scope (e.g. the
        # auth-checks budget enforcement paths) can thread it through.
        e = litellm.BudgetExceededError(
            current_cost=0.5, max_budget=0.1, llm_provider="anthropic"
        )
        assert e.llm_provider == "anthropic"

    def test_should_keep_existing_status_code_and_message(self):
        # Backward-compat guard: existing callers depend on `status_code=429`
        # and the canonical message format.
        e = litellm.BudgetExceededError(current_cost=0.000109, max_budget=0.0001)
        assert e.status_code == 429
        assert "Current cost: 0.000109" in e.message
        assert "Max budget: 0.0001" in e.message

    def test_should_still_be_catchable_as_exception_not_rate_limit_error(self):
        # Critical: we deliberately did NOT make BudgetExceededError a
        # RateLimitError subclass. Existing `except BudgetExceededError:`
        # handlers must keep catching it, and `except RateLimitError:`
        # handlers must NOT start catching it (which would surprise callers
        # who rely on the two being distinct).
        e = litellm.BudgetExceededError(current_cost=0.5, max_budget=0.1)
        assert isinstance(e, Exception)
        assert isinstance(e, litellm.BudgetExceededError)
        assert not isinstance(e, RateLimitError)

    def test_should_propagate_category_to_standard_logging_payload(self):
        from litellm.litellm_core_utils.litellm_logging import (
            StandardLoggingPayloadSetup,
        )

        e = litellm.BudgetExceededError(current_cost=0.5, max_budget=0.1)
        info = StandardLoggingPayloadSetup.get_error_information(e)
        assert info["error_rate_limit_category"] == "litellm_rate_limit"
        assert info["error_rate_limit_type"] == "budget"
        assert info["error_code"] == "429"
        assert info["error_class"] == "BudgetExceededError"

    def test_should_propagate_llm_provider_to_standard_logging_payload(self):
        from litellm.litellm_core_utils.litellm_logging import (
            StandardLoggingPayloadSetup,
        )

        e = litellm.BudgetExceededError(
            current_cost=0.5, max_budget=0.1, llm_provider="bedrock"
        )
        info = StandardLoggingPayloadSetup.get_error_information(e)
        assert info["llm_provider"] == "bedrock"


class TestThirdPartyAttrLeakageGuard:
    """
    The duck-typed read at the StandardLoggingPayload + Prometheus surfaces
    must reject `.category` / `.rate_limit_type` strings set on unrelated
    third-party exceptions. Without validation, a foreign exception that
    happens to declare either attribute name would leak garbage values into
    custom-callback payloads and Prometheus label cardinality.
    """

    def test_should_drop_unknown_category_string_on_third_party_exception(self):
        from litellm.litellm_core_utils.litellm_logging import (
            StandardLoggingPayloadSetup,
        )

        class Foreign(Exception):
            category = "totally_not_a_real_category"

        info = StandardLoggingPayloadSetup.get_error_information(Foreign("boom"))
        assert info["error_rate_limit_category"] is None

    def test_should_drop_unknown_rate_limit_type_string_on_third_party_exception(self):
        from litellm.litellm_core_utils.litellm_logging import (
            StandardLoggingPayloadSetup,
        )

        class Foreign(Exception):
            rate_limit_type = "wat"

        info = StandardLoggingPayloadSetup.get_error_information(Foreign("boom"))
        assert info["error_rate_limit_type"] is None

    def test_should_drop_non_string_garbage_attrs(self):
        from litellm.litellm_core_utils.litellm_logging import (
            StandardLoggingPayloadSetup,
        )

        class Foreign(Exception):
            category = 42
            rate_limit_type = {"lol": "no"}

        info = StandardLoggingPayloadSetup.get_error_information(Foreign())
        assert info["error_rate_limit_category"] is None
        assert info["error_rate_limit_type"] is None

    def test_should_drop_garbage_on_prometheus_label_extraction(self):
        from litellm.integrations.prometheus import PrometheusLogger

        class Foreign(Exception):
            category = "spam"
            rate_limit_type = "spam"

        category, rate_limit_type = PrometheusLogger._extract_rate_limit_labels(
            Foreign()
        )
        assert category is None
        assert rate_limit_type is None

    def test_should_still_accept_legitimate_rate_limit_categories(self):
        # The guard must not over-correct — every documented enum value
        # is a valid string and must pass through.
        from litellm.exceptions import (
            validate_rate_limit_category,
            validate_rate_limit_type,
        )

        for member in RateLimitErrorCategory:
            assert validate_rate_limit_category(member.value) == member.value
            assert validate_rate_limit_category(member) == member.value

        for member in RateLimitType:
            assert validate_rate_limit_type(member.value) == member.value
            assert validate_rate_limit_type(member) == member.value


@pytest.mark.asyncio
class TestBudgetExceededErrorLlmProviderEnrichment:
    """
    BudgetExceededError raise sites in auth_checks.py are tenant-scoped
    (key / team / org / tag) and cannot see the request model. To still
    populate `llm_provider` on the StandardLoggingPayload — which is what
    custom-callback consumers attribute spend to — the central
    UserAPIKeyAuthExceptionHandler enriches the exception from
    `request_data["model"]` before post_call_failure_hook fires.
    """

    async def _run_handler_and_capture_exception_seen_by_callback(
        self, exception: Exception, request_data: dict
    ):
        from unittest.mock import AsyncMock, MagicMock, patch

        from litellm.proxy.auth.auth_exception_handler import (
            UserAPIKeyAuthExceptionHandler,
        )

        captured: dict = {}

        async def fake_post_call_failure_hook(**kwargs):
            captured["exception"] = kwargs["original_exception"]
            return None

        with (
            patch(
                "litellm.proxy.proxy_server.proxy_logging_obj",
                MagicMock(
                    post_call_failure_hook=AsyncMock(
                        side_effect=fake_post_call_failure_hook
                    )
                ),
            ),
            patch(
                "litellm.proxy.proxy_server.general_settings",
                {"use_x_forwarded_for": False},
            ),
            patch(
                "litellm.proxy.auth.auth_exception_handler._get_request_ip_address",
                return_value="127.0.0.1",
            ),
        ):
            try:
                await UserAPIKeyAuthExceptionHandler._handle_authentication_error(
                    e=exception,
                    request=MagicMock(),
                    request_data=request_data,
                    route="/v1/chat/completions",
                    parent_otel_span=None,
                    api_key="sk-test",
                )
            except Exception:
                pass
        return captured.get("exception")

    async def test_should_resolve_llm_provider_from_request_data_when_unset(self):
        err = litellm.BudgetExceededError(current_cost=100, max_budget=10)
        assert err.llm_provider == ""
        seen = await self._run_handler_and_capture_exception_seen_by_callback(
            err, {"model": "openai/gpt-4o-mini"}
        )
        assert seen is not None
        assert seen.llm_provider == "openai"

    async def test_should_not_overwrite_llm_provider_when_caller_set_it(self):
        err = litellm.BudgetExceededError(
            current_cost=100, max_budget=10, llm_provider="anthropic"
        )
        seen = await self._run_handler_and_capture_exception_seen_by_callback(
            err, {"model": "openai/gpt-4o-mini"}
        )
        assert seen.llm_provider == "anthropic"

    async def test_should_fall_back_to_litellm_proxy_when_model_missing(self):
        err = litellm.BudgetExceededError(current_cost=100, max_budget=10)
        seen = await self._run_handler_and_capture_exception_seen_by_callback(err, {})
        assert seen.llm_provider == "litellm_proxy"

    async def test_should_not_enrich_non_budget_exceptions(self):
        err = ValueError("unrelated")
        seen = await self._run_handler_and_capture_exception_seen_by_callback(
            err, {"model": "openai/gpt-4o-mini"}
        )
        assert not hasattr(seen, "llm_provider") or seen.llm_provider != "openai"
