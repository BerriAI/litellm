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

    def test_handles_empty_string(self):
        """Should return None for empty string"""
        result = LiteLLMCompletionResponsesConfig._extract_base64_from_data_url("")
        assert result is None

    def test_handles_none(self):
        """Should return None for None input"""
        result = LiteLLMCompletionResponsesConfig._extract_base64_from_data_url(None)
        assert result is None

    def test_handles_data_url_without_comma(self):
        """Should return None if data URL has no comma separator"""
        invalid_url = "data:image/png;base64"
        result = LiteLLMCompletionResponsesConfig._extract_base64_from_data_url(
            invalid_url
        )
        assert result is None


class TestExtractImageGenerationOutputItems:
    """Tests for _extract_image_generation_output_items function"""

    def test_extracts_single_image(self):
        """Should extract one OutputImageGenerationCall for one image"""
        # Mock objects
        mock_response = Mock(spec=ModelResponse)
        mock_response.id = "test_response_123"

        mock_message = Mock(spec=Message)
        mock_message.images = [
            {
                "image_url": {"url": "data:image/png;base64,ABC123"},
                "type": "image_url",
                "index": 0,
            }
        ]

        mock_choice = Mock(spec=Choices)
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        # Execute
        result = LiteLLMCompletionResponsesConfig._extract_image_generation_output_items(
            chat_completion_response=mock_response,
            choice=mock_choice,
        )

        # Verify
        assert len(result) == 1
        assert isinstance(result[0], OutputImageGenerationCall)
        assert result[0].type == "image_generation_call"
        assert result[0].id == "test_response_123_img_0"
        assert result[0].status == "completed"
        assert result[0].result == "ABC123"

    def test_extracts_multiple_images(self):
        """Should extract multiple OutputImageGenerationCall objects"""
        mock_response = Mock(spec=ModelResponse)
        mock_response.id = "test_response_456"

        mock_message = Mock(spec=Message)
        mock_message.images = [
            {
                "image_url": {"url": "data:image/png;base64,IMG1"},
                "type": "image_url",
                "index": 0,
            },
            {
                "image_url": {"url": "data:image/jpeg;base64,IMG2"},
                "type": "image_url",
                "index": 1,
            },
        ]

        mock_choice = Mock(spec=Choices)
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        result = LiteLLMCompletionResponsesConfig._extract_image_generation_output_items(
            chat_completion_response=mock_response,
            choice=mock_choice,
        )

        assert len(result) == 2
        assert result[0].result == "IMG1"
        assert result[1].result == "IMG2"
        assert result[0].id == "test_response_456_img_0"
        assert result[1].id == "test_response_456_img_1"

    def test_returns_empty_list_if_no_images(self):
        """Should return empty list if message has no images"""
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

    def test_returns_empty_list_if_images_attribute_missing(self):
        """Should return empty list if message doesn't have images attribute"""
        mock_response = Mock(spec=ModelResponse)
        mock_message = Mock(spec=Message)
        # No images attribute

        mock_choice = Mock(spec=Choices)
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        result = LiteLLMCompletionResponsesConfig._extract_image_generation_output_items(
            chat_completion_response=mock_response,
            choice=mock_choice,
        )

        assert result == []

    def test_skips_images_with_invalid_url(self):
        """Should skip images that don't have valid base64 data"""
        mock_response = Mock(spec=ModelResponse)
        mock_response.id = "test_789"

        mock_message = Mock(spec=Message)
        mock_message.images = [
            {"image_url": {"url": ""}, "type": "image_url", "index": 0},  # Empty URL
            {
                "image_url": {"url": "data:image/png;base64,VALID"},
                "type": "image_url",
                "index": 1,
            },
        ]

        mock_choice = Mock(spec=Choices)
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        result = LiteLLMCompletionResponsesConfig._extract_image_generation_output_items(
            chat_completion_response=mock_response,
            choice=mock_choice,
        )

        # Should only extract the valid one
        assert len(result) == 1
        assert result[0].result == "VALID"

    def test_maps_finish_reason_to_status(self):
        """Should correctly map finish_reason to status"""
        mock_response = Mock(spec=ModelResponse)
        mock_response.id = "test_finish"

        mock_message = Mock(spec=Message)
        mock_message.images = [
            {
                "image_url": {"url": "data:image/png;base64,TEST"},
                "type": "image_url",
                "index": 0,
            }
        ]

        # Test with 'length' finish_reason (should map to 'incomplete')
        mock_choice = Mock(spec=Choices)
        mock_choice.message = mock_message
        mock_choice.finish_reason = "length"

        result = LiteLLMCompletionResponsesConfig._extract_image_generation_output_items(
            chat_completion_response=mock_response,
            choice=mock_choice,
        )

        assert result[0].status == "incomplete"


class TestOutputImageGenerationCallType:
    """Tests for OutputImageGenerationCall type definition"""

    def test_creates_valid_instance(self):
        """Should create valid OutputImageGenerationCall instance"""
        output = OutputImageGenerationCall(
            type="image_generation_call",
            id="img_123",
            status="completed",
            result="base64data",
        )

        assert output.type == "image_generation_call"
        assert output.id == "img_123"
        assert output.status == "completed"
        assert output.result == "base64data"

    def test_allows_none_result(self):
        """Should allow None as result value"""
        output = OutputImageGenerationCall(
            type="image_generation_call",
            id="img_456",
            status="failed",
            result=None,
        )

        assert output.result is None

    def test_type_is_literal(self):
        """Type field should only accept 'image_generation_call'"""
        # Valid
        output = OutputImageGenerationCall(
            type="image_generation_call",
            id="img_789",
            status="completed",
            result="data",
        )
        assert output.type == "image_generation_call"


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
