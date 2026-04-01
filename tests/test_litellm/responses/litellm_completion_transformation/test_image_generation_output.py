"""
Unit tests for Responses API image generation support

Tests the fix for Issue #16227:
https://github.com/BerriAI/litellm/issues/16227

Verifies that image generation outputs are correctly transformed
from /chat/completions format to /responses API format.
"""
import pytest
from unittest.mock import Mock
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.types.responses.main import OutputImageGenerationCall
from litellm.types.utils import ModelResponse, Choices, Message


class TestExtractBase64FromDataUrl:
    """Tests for _extract_base64_from_data_url helper function"""

    def test_extracts_base64_from_data_url(self):
        """Should extract pure base64 from data URL with prefix"""
        data_url = "data:image/png;base64,iVBORw0KGgoAAAANS"
        result = LiteLLMCompletionResponsesConfig._extract_base64_from_data_url(
            data_url
        )
        assert result == "iVBORw0KGgoAAAANS"

    def test_returns_base64_as_is_if_no_prefix(self):
        """Should return base64 as-is if no data: prefix"""
        pure_base64 = "iVBORw0KGgoAAAANS"
        result = LiteLLMCompletionResponsesConfig._extract_base64_from_data_url(
            pure_base64
        )
        assert result == pure_base64

    def test_handles_invalid_inputs(self):
        """Should return None for empty/None/malformed inputs"""
        assert LiteLLMCompletionResponsesConfig._extract_base64_from_data_url("") is None
        assert LiteLLMCompletionResponsesConfig._extract_base64_from_data_url(None) is None
        assert LiteLLMCompletionResponsesConfig._extract_base64_from_data_url("data:image/png;base64") is None


class TestExtractImageGenerationOutputItems:
    """Tests for _extract_image_generation_output_items function"""

    def test_extracts_images_correctly(self):
        """Should extract OutputImageGenerationCall objects from images"""
        mock_response = Mock(spec=ModelResponse)
        mock_response.id = "test_123"

        mock_message = Mock(spec=Message)
        mock_message.images = [
            {"image_url": {"url": "data:image/png;base64,IMG1"}, "type": "image_url", "index": 0},
            {"image_url": {"url": "data:image/jpeg;base64,IMG2"}, "type": "image_url", "index": 1},
        ]

        mock_choice = Mock(spec=Choices)
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        result = LiteLLMCompletionResponsesConfig._extract_image_generation_output_items(
            chat_completion_response=mock_response,
            choice=mock_choice,
        )

        assert len(result) == 2
        assert result[0].type == "image_generation_call"
        assert result[0].result == "IMG1"
        assert result[1].result == "IMG2"
        assert result[0].id == "test_123_img_0"
        assert result[1].id == "test_123_img_1"
        assert result[0].status == "completed"

    def test_returns_empty_for_no_images(self):
        """Should return empty list if no images"""
        mock_response = Mock(spec=ModelResponse)
        mock_message = Mock(spec=Message)
        mock_message.images = []

        mock_choice = Mock(spec=Choices)
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        result = LiteLLMCompletionResponsesConfig._extract_image_generation_output_items(
            chat_completion_response=mock_response,
            choice=mock_choice,
        )

        assert result == []

    def test_maps_finish_reason_to_status(self):
        """Should correctly map finish_reason to status"""
        mock_response = Mock(spec=ModelResponse)
        mock_response.id = "test_finish"

        mock_message = Mock(spec=Message)
        mock_message.images = [
            {"image_url": {"url": "data:image/png;base64,TEST"}, "type": "image_url", "index": 0}
        ]

        mock_choice = Mock(spec=Choices)
        mock_choice.message = mock_message
        mock_choice.finish_reason = "length"

        result = LiteLLMCompletionResponsesConfig._extract_image_generation_output_items(
            chat_completion_response=mock_response,
            choice=mock_choice,
        )

        assert result[0].status == "incomplete"


class TestExtractMessageOutputItemsIntegration:
    """Integration tests for _extract_message_output_items with images"""

    def test_detects_images_and_creates_image_generation_call(self):
        """Should detect images in message and create image_generation_call output"""
        mock_response = Mock(spec=ModelResponse)
        mock_response.id = "integration_test_123"

        mock_message = Mock(spec=Message)
        mock_message.images = [
            {
                "image_url": {"url": "data:image/png;base64,INTEGRATION_TEST"},
                "type": "image_url",
                "index": 0,
            }
        ]
        mock_message.role = "assistant"
        mock_message.content = "Here's your image!"

        mock_choice = Mock(spec=Choices)
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        result = LiteLLMCompletionResponsesConfig._extract_message_output_items(
            chat_completion_response=mock_response,
            choices=[mock_choice],
        )

        # Should return image_generation_call, NOT regular message
        assert len(result) == 1
        assert isinstance(result[0], OutputImageGenerationCall)
        assert result[0].type == "image_generation_call"
        assert result[0].result == "INTEGRATION_TEST"

    def test_creates_regular_message_when_no_images(self):
        """Should create regular GenericResponseOutputItem when no images"""
        from litellm.types.responses.main import GenericResponseOutputItem

        mock_response = Mock(spec=ModelResponse)
        mock_response.id = "no_images_123"

        mock_message = Mock(spec=Message)
        # No images attribute or empty
        mock_message.role = "assistant"
        mock_message.content = "Just text, no images"

        mock_choice = Mock(spec=Choices)
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        result = LiteLLMCompletionResponsesConfig._extract_message_output_items(
            chat_completion_response=mock_response,
            choices=[mock_choice],
        )

        assert len(result) == 1
        assert isinstance(result[0], GenericResponseOutputItem)
        assert result[0].type == "message"
