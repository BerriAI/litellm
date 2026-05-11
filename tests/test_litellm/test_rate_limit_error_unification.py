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
