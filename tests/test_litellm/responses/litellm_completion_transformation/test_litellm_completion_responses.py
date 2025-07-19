import os
import sys

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.types.utils import ModelResponse, Choices, Message


class TestLiteLLMCompletionResponsesConfig:
    def test_transform_input_file_item_to_file_item_with_file_id(self):
        """Test transformation of input_file item with file_id to Chat Completion file format"""
        # Setup
        input_item = {"type": "input_file", "file_id": "file-abc123xyz"}

        # Execute
        result = (
            LiteLLMCompletionResponsesConfig._transform_input_file_item_to_file_item(
                input_item
            )
        )

        # Assert
        expected = {"type": "file", "file": {"file_id": "file-abc123xyz"}}
        assert result == expected
        assert result["type"] == "file"
        assert result["file"]["file_id"] == "file-abc123xyz"

    def test_transform_input_file_item_to_file_item_with_file_data(self):
        """Test transformation of input_file item with file_data to Chat Completion file format"""
        # Setup
        file_data = "base64encodeddata"
        input_item = {"type": "input_file", "file_data": file_data}

        # Execute
        result = (
            LiteLLMCompletionResponsesConfig._transform_input_file_item_to_file_item(
                input_item
            )
        )

        # Assert
        expected = {"type": "file", "file": {"file_data": file_data}}
        assert result == expected
        assert result["type"] == "file"
        assert result["file"]["file_data"] == file_data

    def test_transform_input_file_item_to_file_item_with_both_fields(self):
        """Test transformation of input_file item with both file_id and file_data"""
        # Setup
        input_item = {
            "type": "input_file",
            "file_id": "file-abc123xyz",
            "file_data": "base64encodeddata",
        }

        # Execute
        result = (
            LiteLLMCompletionResponsesConfig._transform_input_file_item_to_file_item(
                input_item
            )
        )

        # Assert
        expected = {
            "type": "file",
            "file": {"file_id": "file-abc123xyz", "file_data": "base64encodeddata"},
        }
        assert result == expected
        assert result["type"] == "file"
        assert result["file"]["file_id"] == "file-abc123xyz"
        assert result["file"]["file_data"] == "base64encodeddata"

    def test_transform_input_file_item_to_file_item_empty_file_fields(self):
        """Test transformation of input_file item with no file_id or file_data"""
        # Setup
        input_item = {"type": "input_file"}

        # Execute
        result = (
            LiteLLMCompletionResponsesConfig._transform_input_file_item_to_file_item(
                input_item
            )
        )

        # Assert
        expected = {"type": "file", "file": {}}
        assert result == expected
        assert result["type"] == "file"
        assert result["file"] == {}

    def test_transform_input_file_item_to_file_item_ignores_other_fields(self):
        """Test that transformation only includes file_id and file_data, ignoring other fields"""
        # Setup
        input_item = {
            "type": "input_file",
            "file_id": "file-abc123xyz",
            "extra_field": "should_be_ignored",
            "another_field": 123,
        }

        # Execute
        result = (
            LiteLLMCompletionResponsesConfig._transform_input_file_item_to_file_item(
                input_item
            )
        )

        # Assert
        expected = {"type": "file", "file": {"file_id": "file-abc123xyz"}}
        assert result == expected
        assert "extra_field" not in result["file"]
        assert "another_field" not in result["file"]

    def test_transform_input_file_item_to_file_item_ignores_other_fields(self):
        """Test that transformation only includes file_id and file_data, ignoring other fields"""
        # Setup
        input_item = {
            "type": "input_file",
            "file_id": "file-abc123xyz",
            "extra_field": "should_be_ignored",
            "another_field": 123,
        }

        # Execute
        result = (
            LiteLLMCompletionResponsesConfig._transform_input_file_item_to_file_item(
                input_item
            )
        )

        # Assert
        expected = {"type": "file", "file": {"file_id": "file-abc123xyz"}}
        assert result == expected
        assert "extra_field" not in result["file"]
        assert "another_field" not in result["file"]

    def test_transform_input_image_item_to_image_item_with_image_url(self):
        """Test transformation of input_image item with image_url to Chat Completion image format"""
        # Setup
        image_url = "https://example.com/image.png"
        input_item = {"type": "input_image", "image_url": image_url, "detail": "high"}

        # Execute
        result = (
            LiteLLMCompletionResponsesConfig._transform_input_image_item_to_image_item(
                input_item
            )
        )

        # Assert
        expected = {"type": "image", "image_url": {"url": image_url, "detail": "high"}}
        assert result == expected
        assert result["type"] == "image"
        assert result["image_url"]["url"] == image_url
        assert result["image_url"]["detail"] == "high"

    def test_transform_input_image_item_to_image_item_with_image_data(self):
        """Test transformation of input_image item with image_url to Chat Completion image format"""
        # Setup
        image_url = "data:image/png;base64,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"
        input_item = {"type": "input_image", "image_url": image_url, "detail": "high"}

        # Execute
        result = (
            LiteLLMCompletionResponsesConfig._transform_input_image_item_to_image_item(
                input_item
            )
        )

        # Assert
        expected = {"type": "image", "image_url": {"url": image_url, "detail": "high"}}
        assert result == expected
        assert result["type"] == "image"
        assert result["image_url"]["url"] == image_url
        assert result["image_url"]["detail"] == "high"

    def test_transform_input_image_item_to_image_item_without_detail(self):
        """Test transformation of input_image item with no detail"""
        # Setup
        image_url = "https://example.com/image.png"
        input_item = {"type": "input_image", "image_url": image_url}

        # Execute
        result = (
            LiteLLMCompletionResponsesConfig._transform_input_image_item_to_image_item(
                input_item
            )
        )

        # Assert
        expected = {"type": "image", "image_url": {"url": image_url, "detail": "auto"}}
        assert result == expected
        assert result["type"] == "image"
        assert result["image_url"]["url"] == image_url
        assert result["image_url"]["detail"] == "auto"

    def test_transform_input_image_item_to_image_item_empty_image_fields(self):
        """Test transformation of input_image item with no image_url or detail"""
        # Setup
        input_item = {"type": "input_image"}

        # Execute
        result = (
            LiteLLMCompletionResponsesConfig._transform_input_image_item_to_image_item(
                input_item
            )
        )

        # Assert
        expected = {"type": "image", "image_url": {"url": "", "detail": "auto"}}
        assert result == expected
        assert result["type"] == "image"
        assert result["image_url"]["url"] == ""
        assert result["image_url"]["detail"] == "auto"

    def test_transform_input_image_item_to_image_item_ignores_other_fields(self):
        """Test transformation of input_image item with other fields"""
        # Setup
        input_item = {
            "type": "input_image",
            "image_url": "https://example.com/image.png",
            "extra_field": "should_be_ignored",
            "another_field": 123,
        }

        # Execute
        result = (
            LiteLLMCompletionResponsesConfig._transform_input_image_item_to_image_item(
                input_item
            )
        )

        # Assert
        expected = {"type": "image", "image_url": {"url": "https://example.com/image.png", "detail": "auto"}}
        assert result == expected
        assert result["type"] == "image"
        assert result["image_url"]["url"] == "https://example.com/image.png"
        assert result["image_url"]["detail"] == "auto"
        assert "extra_field" not in result
        assert "another_field" not in result

    def test_transform_chat_completion_response_with_reasoning_content(self):
        """Test that reasoning content is preserved in the full transformation pipeline"""
        # Setup
        chat_completion_response = ModelResponse(
            id="test-response-id",
            created=1234567890,
            model="test-model",
            object="chat.completion",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        content="The answer is 42.",
                        role="assistant",
                        reasoning_content="Let me think about this step by step. The question asks for the meaning of life, and according to The Hitchhiker's Guide to the Galaxy, the answer is 42.",
                    ),
                )
            ],
        )

        # Execute
        responses_api_response = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
            request_input="What is the meaning of life?",
            responses_api_request={},
            chat_completion_response=chat_completion_response,
        )

        # Assert
        assert hasattr(responses_api_response, "output")
        assert (
            len(responses_api_response.output) >= 2
        )

        reasoning_items = [
            item for item in responses_api_response.output if item.type == "reasoning"
        ]
        assert len(reasoning_items) == 1, "Should have exactly one reasoning item"

        reasoning_item = reasoning_items[0]
        assert reasoning_item.id == "test-response-id_reasoning"
        assert reasoning_item.status == "stop"
        assert reasoning_item.role == "assistant"
        assert len(reasoning_item.content) == 1
        assert reasoning_item.content[0].type == "output_text"
        assert "step by step" in reasoning_item.content[0].text
        assert "42" in reasoning_item.content[0].text

        message_items = [
            item for item in responses_api_response.output if item.type == "message"
        ]
        assert len(message_items) == 1, "Should have exactly one message item"

        message_item = message_items[0]
        assert message_item.content[0].text == "The answer is 42."

    def test_transform_chat_completion_response_without_reasoning_content(self):
        """Test that transformation works normally when no reasoning content is present"""
        # Setup
        chat_completion_response = ModelResponse(
            id="test-response-id",
            created=1234567890,
            model="test-model",
            object="chat.completion",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        content="Just a regular answer.",
                        role="assistant",
                    ),
                )
            ],
        )

        # Execute
        responses_api_response = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
            request_input="A simple question?",
            responses_api_request={},
            chat_completion_response=chat_completion_response,
        )

        # Assert
        reasoning_items = [
            item for item in responses_api_response.output if item.type == "reasoning"
        ]
        assert len(reasoning_items) == 0, "Should have no reasoning items"

        message_items = [
            item for item in responses_api_response.output if item.type == "message"
        ]
        assert len(message_items) == 1, "Should have exactly one message item"
        assert message_items[0].content[0].text == "Just a regular answer."

    def test_transform_chat_completion_response_multiple_choices_with_reasoning(self):
        """Test that only reasoning from first choice is included when multiple choices exist"""
        # Setup
        chat_completion_response = ModelResponse(
            id="test-response-id",
            created=1234567890,
            model="test-model",
            object="chat.completion",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        content="First answer.",
                        role="assistant",
                        reasoning_content="First reasoning process.",
                    ),
                ),
                Choices(
                    finish_reason="stop",
                    index=1,
                    message=Message(
                        content="Second answer.",
                        role="assistant",
                        reasoning_content="Second reasoning process.",
                    ),
                ),
            ],
        )

        # Execute
        responses_api_response = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
            request_input="A question with multiple answers?",
            responses_api_request={},
            chat_completion_response=chat_completion_response,
        )

        # Assert
        reasoning_items = [
            item for item in responses_api_response.output if item.type == "reasoning"
        ]
        assert len(reasoning_items) == 1, "Should have exactly one reasoning item"
        assert reasoning_items[0].content[0].text == "First reasoning process."

        message_items = [
            item for item in responses_api_response.output if item.type == "message"
        ]
        assert len(message_items) == 2, "Should have two message items"
