"""
Unit test for https://github.com/BerriAI/litellm/issues/22285

Verifies that extra_headers passed to image_generation() are forwarded
to the OpenAI SDK on the openai/litellm_proxy/openai_compatible_providers
code paths.
"""

from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm.images.main import image_generation
from litellm.litellm_core_utils.litellm_logging import use_custom_pricing_for_model


class TestImageGenerationExtraHeaders:
    """Test that extra_headers are forwarded on the OpenAI code path."""

    @patch("litellm.images.main.openai_chat_completions")
    def test_extra_headers_forwarded_to_openai_image_generation(
        self, mock_openai_chat_completions
    ):
        """
        extra_headers passed to image_generation() should appear in
        optional_params["extra_headers"] when the provider is openai.
        """
        mock_image_response = litellm.utils.ImageResponse(
            created=1234567890,
            data=[{"url": "https://example.com/image.png"}],
        )
        mock_openai_chat_completions.image_generation.return_value = mock_image_response

        extra_headers = {"traceparent": "00-abc123-def456-01", "X-Custom": "value"}

        image_generation(
            model="openai/dall-e-3",
            prompt="A red circle",
            extra_headers=extra_headers,
        )

        mock_openai_chat_completions.image_generation.assert_called_once()
        call_kwargs = mock_openai_chat_completions.image_generation.call_args
        optional_params = call_kwargs.kwargs.get(
            "optional_params", call_kwargs[1].get("optional_params", {})
        )

        assert "extra_headers" in optional_params
        assert optional_params["extra_headers"] == extra_headers

    @patch("litellm.images.main.openai_chat_completions")
    def test_no_extra_headers_when_not_provided(self, mock_openai_chat_completions):
        """
        When extra_headers is not passed, optional_params should not
        contain extra_headers.
        """
        mock_image_response = litellm.utils.ImageResponse(
            created=1234567890,
            data=[{"url": "https://example.com/image.png"}],
        )
        mock_openai_chat_completions.image_generation.return_value = mock_image_response

        image_generation(
            model="openai/dall-e-3",
            prompt="A red circle",
        )

        mock_openai_chat_completions.image_generation.assert_called_once()
        call_kwargs = mock_openai_chat_completions.image_generation.call_args
        optional_params = call_kwargs.kwargs.get(
            "optional_params", call_kwargs[1].get("optional_params", {})
        )

        assert "extra_headers" not in optional_params

    @patch("litellm.images.main.openai_chat_completions")
    def test_image_generation_forwards_custom_pricing_kwargs_to_logging(
        self, mock_openai_chat_completions
    ):
        """Pricing a deployment declares under litellm_params (e.g. input_cost_per_pixel)
        reaches self.litellm_params so use_custom_pricing_for_model fires on generations."""
        mock_openai_chat_completions.image_generation.return_value = litellm.utils.ImageResponse(
            created=1234567890,
            data=[{"url": "https://example.com/image.png"}],
        )
        captured_litellm_params = {}

        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {}

        original_update = mock_logging_obj.update_from_kwargs

        def capturing_update(**update_kwargs):
            captured_litellm_params.update(update_kwargs.get("litellm_params", {}))
            return original_update(**update_kwargs)

        mock_logging_obj.update_from_kwargs = capturing_update

        image_generation(
            model="openai/dall-e-3",
            prompt="A red circle",
            litellm_logging_obj=mock_logging_obj,
            input_cost_per_pixel=4.76837158203125e-08,
        )

        assert captured_litellm_params["input_cost_per_pixel"] == pytest.approx(4.76837158203125e-08)
        assert use_custom_pricing_for_model(captured_litellm_params) is True
