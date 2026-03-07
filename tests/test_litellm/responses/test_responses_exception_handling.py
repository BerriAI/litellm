"""
Tests for Responses API exception handling.

Verifies that exceptions already mapped by litellm.completion() are not
double-mapped when they propagate through the Responses API bridge.

Regression tests for https://github.com/BerriAI/litellm/issues/22121
"""

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.exceptions import (
    BadRequestError,
    NotFoundError,
    RateLimitError,
)


class TestResponsesExceptionPreservation:
    """
    Tests that the Responses API bridge re-raises litellm exceptions
    directly instead of double-mapping them through exception_type().
    """

    @pytest.mark.asyncio
    async def test_aresponses_preserves_bad_request_error(self):
        """
        When the completion bridge raises a BadRequestError, aresponses()
        should re-raise it as-is instead of collapsing it into
        APIConnectionError.
        """
        original_error = BadRequestError(
            message="Invalid model parameter",
            model="test-model",
            llm_provider="openai",
        )

        with patch(
            "litellm.responses.main.ProviderConfigManager.get_provider_responses_api_config",
            return_value=None,
        ), patch(
            "litellm.responses.main.litellm_completion_transformation_handler.response_api_handler",
            side_effect=original_error,
        ):
            with pytest.raises(BadRequestError) as exc_info:
                await litellm.aresponses(
                    model="openai/gpt-4o",
                    input="test",
                )
            assert exc_info.value is original_error

    @pytest.mark.asyncio
    async def test_aresponses_preserves_rate_limit_error(self):
        """
        When the completion bridge raises a RateLimitError, aresponses()
        should re-raise it as-is.
        """
        original_error = RateLimitError(
            message="Rate limit exceeded",
            model="test-model",
            llm_provider="openai",
        )

        with patch(
            "litellm.responses.main.ProviderConfigManager.get_provider_responses_api_config",
            return_value=None,
        ), patch(
            "litellm.responses.main.litellm_completion_transformation_handler.response_api_handler",
            side_effect=original_error,
        ):
            with pytest.raises(RateLimitError) as exc_info:
                await litellm.aresponses(
                    model="openai/gpt-4o",
                    input="test",
                )
            assert exc_info.value is original_error

    @pytest.mark.asyncio
    async def test_aresponses_preserves_not_found_error(self):
        """
        When the completion bridge raises a NotFoundError, aresponses()
        should re-raise it as-is.
        """
        original_error = NotFoundError(
            message="Model not found",
            model="test-model",
            llm_provider="openai",
        )

        with patch(
            "litellm.responses.main.ProviderConfigManager.get_provider_responses_api_config",
            return_value=None,
        ), patch(
            "litellm.responses.main.litellm_completion_transformation_handler.response_api_handler",
            side_effect=original_error,
        ):
            with pytest.raises(NotFoundError) as exc_info:
                await litellm.aresponses(
                    model="openai/gpt-4o",
                    input="test",
                )
            assert exc_info.value is original_error

    @pytest.mark.asyncio
    async def test_aresponses_still_maps_non_litellm_exceptions(self):
        """
        Non-litellm exceptions (e.g. ValueError from transformation code)
        should still be mapped through exception_type(). This ensures we
        only skip mapping for already-mapped exceptions.
        """
        with patch(
            "litellm.responses.main.ProviderConfigManager.get_provider_responses_api_config",
            return_value=None,
        ), patch(
            "litellm.responses.main.litellm_completion_transformation_handler.response_api_handler",
            side_effect=ValueError("unexpected transformation error"),
        ):
            with pytest.raises(Exception) as exc_info:
                await litellm.aresponses(
                    model="openai/gpt-4o",
                    input="test",
                )
            # Should NOT be a ValueError -- it should be mapped to a litellm type
            assert not isinstance(exc_info.value, ValueError)

    def test_responses_preserves_bad_request_error_sync(self):
        """
        Sync variant: when the completion bridge raises a BadRequestError,
        responses() should re-raise it as-is.
        """
        original_error = BadRequestError(
            message="Invalid model parameter",
            model="test-model",
            llm_provider="openai",
        )

        with patch(
            "litellm.responses.main.ProviderConfigManager.get_provider_responses_api_config",
            return_value=None,
        ), patch(
            "litellm.responses.main.litellm_completion_transformation_handler.response_api_handler",
            side_effect=original_error,
        ):
            with pytest.raises(BadRequestError) as exc_info:
                litellm.responses(
                    model="openai/gpt-4o",
                    input="test",
                )
            assert exc_info.value is original_error
