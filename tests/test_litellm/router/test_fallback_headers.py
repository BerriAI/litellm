"""
Tests for x-litellm-fallback-model-used header exposure.

Covers issue: https://github.com/BerriAI/litellm/issues/25503

When a fallback model is used instead of the primary model, the response should
include an x-litellm-fallback-model-used header so callers can tell which model
actually served the request.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.router_utils.add_retry_fallback_headers import (
    add_fallback_headers_to_response,
)
from litellm.router_utils.fallback_event_handlers import run_async_fallback


# ---------------------------------------------------------------------------
# Unit tests for add_fallback_headers_to_response
# ---------------------------------------------------------------------------


class TestAddFallbackHeadersToResponse:
    """Unit tests for add_fallback_headers_to_response."""

    def _make_response(self) -> MagicMock:
        """Return a minimal pydantic-like response mock."""
        from pydantic import BaseModel

        class _FakeResponse(BaseModel):
            model: str = "gpt-4"
            _hidden_params: dict = {}

        resp = _FakeResponse()
        return resp

    def test_fallback_model_header_set_when_fallback_occurred(self):
        """x-litellm-fallback-model-used is present when a fallback model is provided."""
        resp = self._make_response()
        result = add_fallback_headers_to_response(
            response=resp,
            attempted_fallbacks=1,
            fallback_model="claude-3-haiku",
        )
        headers = result._hidden_params.get("additional_headers", {})
        assert headers.get("x-litellm-fallback-model-used") == "claude-3-haiku"

    def test_fallback_model_header_absent_when_no_fallback(self):
        """x-litellm-fallback-model-used is NOT set when primary model succeeded."""
        resp = self._make_response()
        result = add_fallback_headers_to_response(
            response=resp,
            attempted_fallbacks=0,
            fallback_model=None,
        )
        headers = result._hidden_params.get("additional_headers", {})
        assert "x-litellm-fallback-model-used" not in headers

    def test_attempted_fallbacks_header_always_set(self):
        """x-litellm-attempted-fallbacks is always present regardless of fallback_model."""
        resp = self._make_response()
        result = add_fallback_headers_to_response(
            response=resp,
            attempted_fallbacks=2,
        )
        headers = result._hidden_params.get("additional_headers", {})
        assert headers.get("x-litellm-attempted-fallbacks") == 2

    def test_fallback_model_default_is_none(self):
        """Calling add_fallback_headers_to_response without fallback_model does not error."""
        resp = self._make_response()
        # Should not raise
        result = add_fallback_headers_to_response(
            response=resp,
            attempted_fallbacks=0,
        )
        headers = result._hidden_params.get("additional_headers", {})
        assert "x-litellm-fallback-model-used" not in headers

    def test_returns_none_unchanged(self):
        """If response is None, it is returned unchanged without error."""
        result = add_fallback_headers_to_response(
            response=None,
            attempted_fallbacks=1,
            fallback_model="gpt-3.5-turbo",
        )
        assert result is None

    def test_fallback_model_header_with_multiple_fallback_depths(self):
        """Header captures the model used even when multiple fallback depths occurred."""
        resp = self._make_response()
        result = add_fallback_headers_to_response(
            response=resp,
            attempted_fallbacks=3,
            fallback_model="gpt-3.5-turbo",
        )
        headers = result._hidden_params.get("additional_headers", {})
        assert headers.get("x-litellm-fallback-model-used") == "gpt-3.5-turbo"
        assert headers.get("x-litellm-attempted-fallbacks") == 3


# ---------------------------------------------------------------------------
# Unit tests for run_async_fallback (verifies fallback_model propagation)
# ---------------------------------------------------------------------------


class TestRunAsyncFallbackHeaderPropagation:
    """Tests that run_async_fallback stamps x-litellm-fallback-model-used on success."""

    @pytest.mark.asyncio
    async def test_fallback_model_header_stamped_on_successful_string_fallback(self):
        """
        When a string fallback model succeeds, x-litellm-fallback-model-used
        should be set to that model's name.
        """
        from pydantic import BaseModel

        class _FakeResponse(BaseModel):
            model: str = "claude-3-haiku"
            _hidden_params: dict = {}

        fake_response = _FakeResponse()

        mock_router = MagicMock()
        mock_router.log_retry = MagicMock(side_effect=lambda kwargs, e: kwargs)
        mock_router.async_function_with_fallbacks = AsyncMock(
            return_value=fake_response
        )

        result = await run_async_fallback(
            litellm_router=mock_router,
            fallback_model_group=["claude-3-haiku"],
            original_model_group="gpt-4",
            original_exception=Exception("primary failed"),
            max_fallbacks=3,
            fallback_depth=0,
            model="gpt-4",
        )

        headers = result._hidden_params.get("additional_headers", {})
        assert headers.get("x-litellm-fallback-model-used") == "claude-3-haiku"

    @pytest.mark.asyncio
    async def test_fallback_model_header_absent_when_primary_succeeds_via_router(self):
        """
        When run_async_fallback skips all candidates because they equal the
        original_model_group, no fallback fires and the response should NOT
        carry x-litellm-fallback-model-used.

        This tests the router path: fallback_model_group contains only the
        original model so the loop body is never entered and error_from_fallbacks
        (the original exception) is raised — which means the caller never gets a
        response with the fallback header at all.  We verify that by confirming
        the exception propagates rather than a header-bearing response being returned.
        """
        mock_router = MagicMock()
        mock_router.log_retry = MagicMock(side_effect=lambda kwargs, e: kwargs)

        original_exc = Exception("primary failed")
        with pytest.raises(Exception, match="primary failed"):
            await run_async_fallback(
                litellm_router=mock_router,
                # All candidates are the same as original — loop body is skipped entirely
                fallback_model_group=["gpt-4"],
                original_model_group="gpt-4",
                original_exception=original_exc,
                max_fallbacks=3,
                fallback_depth=0,
                model="gpt-4",
            )
        # async_function_with_fallbacks was never called — no fallback header stamped
        mock_router.async_function_with_fallbacks.assert_not_called()

    @pytest.mark.asyncio
    async def test_all_fallbacks_fail_raises_exception(self):
        """When all fallback models fail, the last exception is re-raised."""
        mock_router = MagicMock()
        mock_router.log_retry = MagicMock(side_effect=lambda kwargs, e: kwargs)
        mock_router.async_function_with_fallbacks = AsyncMock(
            side_effect=Exception("fallback also failed")
        )

        with pytest.raises(Exception, match="fallback also failed"):
            await run_async_fallback(
                litellm_router=mock_router,
                fallback_model_group=["claude-3-haiku"],
                original_model_group="gpt-4",
                original_exception=Exception("primary failed"),
                max_fallbacks=3,
                fallback_depth=0,
                model="gpt-4",
            )

    @pytest.mark.asyncio
    async def test_max_fallback_depth_raises_original_exception(self):
        """When max_fallbacks is reached, the original exception is re-raised."""
        original_exc = Exception("original failure")
        mock_router = MagicMock()

        with pytest.raises(Exception, match="original failure"):
            await run_async_fallback(
                litellm_router=mock_router,
                fallback_model_group=["claude-3-haiku"],
                original_model_group="gpt-4",
                original_exception=original_exc,
                max_fallbacks=3,
                fallback_depth=3,  # already at max
                model="gpt-4",
            )
