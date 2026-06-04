"""
Tests for the router-side rate-limit category fix.

Background
----------
PR #27687 added :class:`RateLimitErrorCategory` and the ``category`` /
``rate_limit_type`` kwargs on :class:`litellm.RateLimitError`, with
``category=VENDOR_RATE_LIMIT`` as the default. That default silently
mislabeled every router-side TPM/RPM throttle (in
``router_strategy.lowest_tpm_rpm_v2``,
``router_utils.pre_call_checks.model_rate_limit_check``, etc.) as a vendor
error: those callsites construct ``RateLimitError`` without passing
``category=`` and so inherit the vendor default. The fix:

1. The :class:`RateLimitErrorCategory` enum now exposes
   ``UNKNOWN_RATE_LIMIT`` (an "unknown" sentinel) and
   :meth:`RateLimitError.__init__` defaults to it. Future omissions surface in
   dashboards as ``unknown_rate_limit`` instead of a confidently-wrong vendor
   label.
2. Every router-side raise was updated to pass an explicit
   ``category=LITELLM_RATE_LIMIT`` (and ``rate_limit_type=`` when the
   dimension is determinable from the throttle that fired).
3. Every vendor-mapping raise in
   :mod:`litellm.litellm_core_utils.exception_mapping_utils` (and the few
   vendor mocks under ``llms/``) was updated to keep passing
   ``category=VENDOR_RATE_LIMIT`` explicitly so the new "unknown" default
   doesn't change vendor-side behavior.

These tests pin both halves of that contract.
"""

from unittest.mock import patch

import httpx
import pytest

import litellm
from litellm.exceptions import (
    RateLimitError,
    RateLimitErrorCategory,
    RateLimitType,
)


# ---------------------------------------------------------------------------
# Step 1: enum + default behavior
# ---------------------------------------------------------------------------


class TestUnknownRateLimitCategory:
    def test_should_expose_unknown_rate_limit_value(self):
        # Sanity: the new enum value exists and round-trips through the str
        # protocol so dashboards / log aggregators can compare against the
        # plain string without importing the enum.
        assert RateLimitErrorCategory.UNKNOWN_RATE_LIMIT == "unknown_rate_limit"
        assert "unknown_rate_limit" == RateLimitErrorCategory.UNKNOWN_RATE_LIMIT

    def test_should_export_unknown_value_on_litellm_module(self):
        assert (
            litellm.RateLimitErrorCategory.UNKNOWN_RATE_LIMIT
            == RateLimitErrorCategory.UNKNOWN_RATE_LIMIT
        )

    def test_should_default_category_to_unknown_when_unspecified(self):
        # Constructing RateLimitError without an explicit category yields the
        # honest "unknown" sentinel, NOT a vendor assumption. This is the
        # whole point of the fix: silent omissions become visible in
        # dashboards as ``unknown_rate_limit``.
        e = RateLimitError(message="oops", llm_provider="openai", model="gpt-4")
        assert e.category == RateLimitErrorCategory.UNKNOWN_RATE_LIMIT
        assert e.category == "unknown_rate_limit"

    def test_should_still_accept_explicit_vendor_category(self):
        # Vendor callsites still set the right value when they pass one.
        e = RateLimitError(
            message="oops",
            llm_provider="openai",
            model="gpt-4",
            category=RateLimitErrorCategory.VENDOR_RATE_LIMIT,
        )
        assert e.category == RateLimitErrorCategory.VENDOR_RATE_LIMIT

    def test_should_still_accept_explicit_litellm_category(self):
        e = RateLimitError(
            message="oops",
            llm_provider="openai",
            model="gpt-4",
            category=RateLimitErrorCategory.LITELLM_RATE_LIMIT,
            rate_limit_type=RateLimitType.REQUESTS,
        )
        assert e.category == RateLimitErrorCategory.LITELLM_RATE_LIMIT
        assert e.rate_limit_type == RateLimitType.REQUESTS


# ---------------------------------------------------------------------------
# Step 3: regression tests for router-side raises
# ---------------------------------------------------------------------------


class TestLowestTpmRpmV2RouterSideCategory:
    """
    The five raise sites in ``router_strategy.lowest_tpm_rpm_v2`` (sync
    pre-check, async pre-check, both their redis-overrun branches, and the
    no-deployments-available raise) must all carry
    ``category=LITELLM_RATE_LIMIT``. The four pre-check sites also carry
    ``rate_limit_type=REQUESTS``; the terminal no-deployments-available raise
    intentionally leaves ``rate_limit_type`` unset because the filtering could
    have been driven by TPM, RPM, or both. They were silently labeled as
    ``vendor_rate_limit`` before this fix.
    """

    def _build_handler(self):
        from litellm.caching.caching import DualCache
        from litellm.router_strategy.lowest_tpm_rpm_v2 import (
            LowestTPMLoggingHandler_v2,
        )

        cache = DualCache()
        return LowestTPMLoggingHandler_v2(router_cache=cache)

    def test_should_label_local_rpm_overrun_as_litellm_requests(self):
        handler = self._build_handler()
        deployment = {
            "litellm_params": {"model": "gpt-4", "rpm": 1},
            "model_info": {"id": "abc-123"},
            "model_name": "gpt-4",
            "rpm": 1,
        }
        # Force the local cache lookup to come back already at the limit so
        # the sync branch raises immediately.
        with patch.object(handler.router_cache, "get_cache", return_value=5):
            with pytest.raises(litellm.RateLimitError) as exc_info:
                handler.pre_call_check(deployment)

        assert exc_info.value.category == RateLimitErrorCategory.LITELLM_RATE_LIMIT
        assert exc_info.value.rate_limit_type == RateLimitType.REQUESTS

    @pytest.mark.asyncio
    async def test_should_label_async_local_rpm_overrun_as_litellm_requests(self):
        handler = self._build_handler()
        deployment = {
            "litellm_params": {"model": "gpt-4", "rpm": 1},
            "model_info": {"id": "abc-123"},
            "model_name": "gpt-4",
            "rpm": 1,
        }

        async def _stub_get(*args, **kwargs):
            return 5

        with patch.object(
            handler.router_cache, "async_get_cache", side_effect=_stub_get
        ):
            with pytest.raises(litellm.RateLimitError) as exc_info:
                await handler.async_pre_call_check(deployment, parent_otel_span=None)

        assert exc_info.value.category == RateLimitErrorCategory.LITELLM_RATE_LIMIT
        assert exc_info.value.rate_limit_type == RateLimitType.REQUESTS

    @pytest.mark.asyncio
    async def test_should_label_no_deployments_available_as_litellm_rate_limit(self):
        # The terminal "no deployments available" raise — fired from
        # ``async_get_available_deployments`` after every healthy deployment
        # was filtered out — must carry the litellm category. The
        # ``rate_limit_type`` dimension is intentionally left unset because
        # the filtering could have been driven by TPM, RPM, or both.
        handler = self._build_handler()
        with pytest.raises(litellm.RateLimitError) as exc_info:
            await handler.async_get_available_deployments(
                model_group="gpt-4",
                healthy_deployments=[],
            )

        assert exc_info.value.category == RateLimitErrorCategory.LITELLM_RATE_LIMIT
        assert exc_info.value.rate_limit_type is None


class TestModelRateLimitCheckRouterSideCategory:
    """
    The four raise sites in
    ``router_utils.pre_call_checks.model_rate_limit_check`` split between
    TPM (``rate_limit_type=TOKENS``) and RPM
    (``rate_limit_type=REQUESTS``); both carry
    ``category=LITELLM_RATE_LIMIT``.
    """

    def _build_check(self):
        from litellm.caching.dual_cache import DualCache
        from litellm.router_utils.pre_call_checks.model_rate_limit_check import (
            ModelRateLimitingCheck,
        )

        return ModelRateLimitingCheck(dual_cache=DualCache())

    def test_should_label_sync_tpm_overrun_as_litellm_tokens(self):
        check = self._build_check()
        deployment = {
            "litellm_params": {"model": "gpt-4", "tpm": 10},
            "model_info": {"id": "abc-123", "tpm": 10},
            "model_name": "gpt-4",
        }
        with patch.object(check.dual_cache, "get_cache", return_value=999):
            with pytest.raises(litellm.RateLimitError) as exc_info:
                check.pre_call_check(deployment)

        assert exc_info.value.category == RateLimitErrorCategory.LITELLM_RATE_LIMIT
        assert exc_info.value.rate_limit_type == RateLimitType.TOKENS

    def test_should_label_sync_rpm_overrun_as_litellm_requests(self):
        check = self._build_check()
        deployment = {
            "litellm_params": {"model": "gpt-4", "rpm": 1},
            "model_info": {"id": "abc-123", "rpm": 1},
            "model_name": "gpt-4",
        }
        # No TPM limit, so the TPM branch is skipped; RPM increment returns a
        # value above the limit, triggering the RPM raise.
        with patch.object(check.dual_cache, "increment_cache", return_value=42):
            with pytest.raises(litellm.RateLimitError) as exc_info:
                check.pre_call_check(deployment)

        assert exc_info.value.category == RateLimitErrorCategory.LITELLM_RATE_LIMIT
        assert exc_info.value.rate_limit_type == RateLimitType.REQUESTS

    @pytest.mark.asyncio
    async def test_should_label_async_tpm_overrun_as_litellm_tokens(self):
        check = self._build_check()
        deployment = {
            "litellm_params": {"model": "gpt-4", "tpm": 10},
            "model_info": {"id": "abc-123", "tpm": 10},
            "model_name": "gpt-4",
        }

        async def _stub(*args, **kwargs):
            return 999

        with patch.object(check.dual_cache, "async_get_cache", side_effect=_stub):
            with pytest.raises(litellm.RateLimitError) as exc_info:
                await check.async_pre_call_check(deployment)

        assert exc_info.value.category == RateLimitErrorCategory.LITELLM_RATE_LIMIT
        assert exc_info.value.rate_limit_type == RateLimitType.TOKENS

    @pytest.mark.asyncio
    async def test_should_label_async_rpm_overrun_as_litellm_requests(self):
        check = self._build_check()
        deployment = {
            "litellm_params": {"model": "gpt-4", "rpm": 1},
            "model_info": {"id": "abc-123", "rpm": 1},
            "model_name": "gpt-4",
        }

        async def _stub_inc(*args, **kwargs):
            return 42

        with patch.object(
            check.dual_cache,
            "async_increment_cache",
            side_effect=_stub_inc,
        ):
            with pytest.raises(litellm.RateLimitError) as exc_info:
                await check.async_pre_call_check(deployment)

        assert exc_info.value.category == RateLimitErrorCategory.LITELLM_RATE_LIMIT
        assert exc_info.value.rate_limit_type == RateLimitType.REQUESTS


class TestAnthropicMockVendorCategory:
    """
    The Anthropic experimental-pass-through mock raise simulates an upstream
    vendor 429 — it must carry ``category=VENDOR_RATE_LIMIT`` so callers that
    use the mock to test vendor-side behavior see the right category.
    """

    def test_should_label_anthropic_mock_rate_limit_as_vendor(self):
        from litellm.llms.anthropic.experimental_pass_through.messages.utils import (
            mock_response as anthropic_mock_response,
        )

        with pytest.raises(litellm.RateLimitError) as exc_info:
            anthropic_mock_response(
                model="claude-3-opus",
                messages=[],
                max_tokens=10,
                mock_response="litellm.RateLimitError",
            )

        assert exc_info.value.category == RateLimitErrorCategory.VENDOR_RATE_LIMIT


# ---------------------------------------------------------------------------
# Step 2 regression: vendor mappings in exception_mapping_utils still vendor
# ---------------------------------------------------------------------------


class TestExceptionMappingVendorRegression:
    """
    Step 2 added explicit ``category=VENDOR_RATE_LIMIT`` at every raise site
    in ``exception_mapping_utils.py``. These tests exercise a few
    representative branches end-to-end through ``exception_type`` to prove
    none were missed — without this explicit kwarg, the new
    ``UNKNOWN_RATE_LIMIT`` default would silently leak into vendor flows.
    """

    def _make_upstream_429(self):
        # Build a synthetic httpx-flavored 429 response that
        # ``exception_type`` will see when it inspects ``original_exception``.
        request = httpx.Request("POST", "https://example.test/v1/chat")
        return httpx.Response(status_code=429, request=request)

    def test_should_label_anthropic_status_429_as_vendor(self):
        from litellm.litellm_core_utils.exception_mapping_utils import (
            exception_type,
        )

        original = Exception("anthropic ratelimit")
        original.status_code = 429  # type: ignore[attr-defined]
        original.message = "ratelimit"  # type: ignore[attr-defined]
        original.response = self._make_upstream_429()  # type: ignore[attr-defined]

        with pytest.raises(litellm.RateLimitError) as exc_info:
            exception_type(
                model="claude-3-opus-20240229",
                original_exception=original,
                custom_llm_provider="anthropic",
            )

        assert exc_info.value.category == RateLimitErrorCategory.VENDOR_RATE_LIMIT

    def test_should_label_string_match_429_as_vendor(self):
        # The first generic raise (string-match path with
        # ExceptionCheckers.is_error_str_rate_limit) covers the
        # "rate limit" substring pattern used by many providers' SDKs.
        from litellm.litellm_core_utils.exception_mapping_utils import (
            exception_type,
        )

        original = Exception("something Rate limit reached for model")
        original.response = self._make_upstream_429()  # type: ignore[attr-defined]

        with pytest.raises(litellm.RateLimitError) as exc_info:
            exception_type(
                model="gpt-4",
                original_exception=original,
                custom_llm_provider="openai",
            )

        assert exc_info.value.category == RateLimitErrorCategory.VENDOR_RATE_LIMIT

    def test_should_label_replicate_status_429_as_vendor(self):
        from litellm.litellm_core_utils.exception_mapping_utils import (
            exception_type,
        )

        original = Exception("replicate ratelimit")
        original.status_code = 429  # type: ignore[attr-defined]
        original.message = "Rate limit reached"  # type: ignore[attr-defined]
        original.response = self._make_upstream_429()  # type: ignore[attr-defined]

        with pytest.raises(litellm.RateLimitError) as exc_info:
            exception_type(
                model="meta/llama-2-70b-chat",
                original_exception=original,
                custom_llm_provider="replicate",
            )

        assert exc_info.value.category == RateLimitErrorCategory.VENDOR_RATE_LIMIT
