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
