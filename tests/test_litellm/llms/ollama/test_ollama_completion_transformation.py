import json
import os
import sys
from litellm._uuid import uuid
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.ollama.completion.transformation import (
    OllamaConfig,
    OllamaTextCompletionResponseIterator,
)
from litellm.types.utils import Message, ModelResponse, ModelResponseStream


class TestOllamaConfig:
    def test_transform_response_standard(self):
        # Initialize config
        config = OllamaConfig()

        # Create mock response
        raw_response = MagicMock()
        raw_response.json.return_value = {
            "response": "Hello, I am an AI assistant",
            "prompt_eval_count": 10,
            "eval_count": 5,
        }

        # Create properly structured model response object
        model_response = ModelResponse(
            id="test_id",
            choices=[{"message": Message(content="")}],
        )

        # Create mock encoding
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]  # Return dummy token IDs

        # Transform response
        result = config.transform_response(
            model="llama2",
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=mock_encoding,
        )

        # Verify response
        assert result.choices[0]["message"].content == "Hello, I am an AI assistant"
        assert result.choices[0]["finish_reason"] == "stop"
        assert result.model == "ollama/llama2"
        assert result.created is not None
        # Access usage properly
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5
        assert result["usage"]["total_tokens"] == 15

    @patch("uuid.uuid4")
    def test_transform_response_json_function_call(self, mock_uuid4):
        # Setup mock UUID
        mock_uuid4.return_value = "test-uuid"

        # Initialize config
        config = OllamaConfig()

        # Create mock response with JSON function call format
        raw_response = MagicMock()
        raw_response.json.return_value = {
            "response": json.dumps(
                {"name": "get_weather", "arguments": {"location": "San Francisco"}}
            )
        }

        # Create properly structured model response object
        model_response = ModelResponse(
            id="test_id",
            choices=[{"message": Message(content="")}],
        )

        # Create mock encoding
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]  # Return dummy token IDs

        # Transform response
        result = config.transform_response(
            model="llama2",
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={"format": "json"},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=mock_encoding,
        )

        # Verify result has tool_calls
        assert result.choices[0]["message"].content is None
        assert result.choices[0]["finish_reason"] == "tool_calls"
        assert len(result.choices[0]["message"].tool_calls) == 1
        assert result.choices[0]["message"].tool_calls[0]["id"].startswith("call_")
        assert (
            result.choices[0]["message"].tool_calls[0]["function"]["name"]
            == "get_weather"
        )
        assert json.loads(
            result.choices[0]["message"].tool_calls[0]["function"]["arguments"]
        ) == {"location": "San Francisco"}
        # No usage assertions here as we don't need to test them in every case

    def test_transform_response_regular_json(self):
        # Initialize config
        config = OllamaConfig()

        # Create mock response with regular JSON (not function call)
        raw_response = MagicMock()
        raw_response.json.return_value = {
            "response": json.dumps(
                {"result": "success", "data": {"temperature": 72, "unit": "F"}}
            )
        }

        # Create properly structured model response object
        model_response = ModelResponse(
            id="test_id",
            choices=[{"message": Message(content="")}],
        )

        # Create mock encoding
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]  # Return dummy token IDs

        # Transform response
        result = config.transform_response(
            model="llama2",
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={"format": "json"},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=mock_encoding,
        )

        # Verify result has JSON content
        expected_content = json.dumps(
            {"result": "success", "data": {"temperature": 72, "unit": "F"}}
        )
        assert result.choices[0]["message"].content == expected_content
        assert result.choices[0]["finish_reason"] == "stop"
        # No usage assertions here as we don't need to test them in every case

    def test_transform_response_with_thinking_tags(self):
        """Test that responses with <think>...</think> tags parse reasoning content correctly."""
        # Initialize config
        config = OllamaConfig()

        # Create mock response with thinking tags
        raw_response = MagicMock()
        raw_response.json.return_value = {
            "response": "<think>I need to think about this problem step by step</think>Here is my answer",
            "prompt_eval_count": 15,
            "eval_count": 8,
        }

        # Create properly structured model response object
        model_response = ModelResponse(
            id="test_id",
            choices=[{"message": Message(content="")}],
        )

        # Create mock encoding
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]

        # Transform response
        result = config.transform_response(
            model="llama2",
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=mock_encoding,
        )

        # Verify reasoning content is extracted
        assert (
            result.choices[0]["message"].reasoning_content
            == "I need to think about this problem step by step"
        )
        assert result.choices[0]["message"].content == "Here is my answer"
        assert result.choices[0]["finish_reason"] == "stop"

    def test_transform_response_with_thinking_tags_alternative(self):
        """Test that responses with <thinking>...</thinking> tags parse reasoning content correctly."""
        # Initialize config
        config = OllamaConfig()

        # Create mock response with thinking tags (alternative format)
        raw_response = MagicMock()
        raw_response.json.return_value = {
            "response": "<thinking>Let me analyze this carefully</thinking>The solution is X",
        }

        # Create properly structured model response object
        model_response = ModelResponse(
            id="test_id",
            choices=[{"message": Message(content="")}],
        )

        # Create mock encoding
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]

        # Transform response
        result = config.transform_response(
            model="llama2",
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=mock_encoding,
        )

        # Verify reasoning content is extracted
        assert (
            result.choices[0]["message"].reasoning_content
            == "Let me analyze this carefully"
        )
        assert result.choices[0]["message"].content == "The solution is X"
        assert result.choices[0]["finish_reason"] == "stop"

    def test_transform_response_with_multiline_thinking_tags(self):
        """Test that responses with multiline thinking content work correctly."""
        # Initialize config
        config = OllamaConfig()

        # Create mock response with multiline thinking content
        raw_response = MagicMock()
        raw_response.json.return_value = {
            "response": "<think>\nThis is a complex problem.\nI need to break it down:\n1. First step\n2. Second step\n</think>Based on my analysis, the answer is Y",
        }

        # Create properly structured model response object
        model_response = ModelResponse(
            id="test_id",
            choices=[{"message": Message(content="")}],
        )

        # Create mock encoding
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]

        # Transform response
        result = config.transform_response(
            model="llama2",
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=mock_encoding,
        )

        # Verify multiline reasoning content is extracted
        expected_reasoning = "\nThis is a complex problem.\nI need to break it down:\n1. First step\n2. Second step\n"
        assert result.choices[0]["message"].reasoning_content == expected_reasoning
        assert (
            result.choices[0]["message"].content
            == "Based on my analysis, the answer is Y"
        )
        assert result.choices[0]["finish_reason"] == "stop"

    def test_transform_response_thinking_only(self):
        """Test response with only thinking content and no additional content."""
        # Initialize config
        config = OllamaConfig()

        # Create mock response with only thinking content
        raw_response = MagicMock()
        raw_response.json.return_value = {
            "response": "<think>Just internal thoughts, no response</think>",
        }

        # Create properly structured model response object
        model_response = ModelResponse(
            id="test_id",
            choices=[{"message": Message(content="")}],
        )

        # Create mock encoding
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]

        # Transform response
        result = config.transform_response(
            model="llama2",
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=mock_encoding,
        )

        # Verify reasoning content is extracted and content is empty
        assert (
            result.choices[0]["message"].reasoning_content
            == "Just internal thoughts, no response"
        )
        assert result.choices[0]["message"].content == ""
        assert result.choices[0]["finish_reason"] == "stop"

    def test_transform_response_json_mode_with_thinking_tags(self):
        """Test JSON mode with thinking tags - should handle as text when JSON parsing fails."""
        # Initialize config
        config = OllamaConfig()

        # Create mock response with thinking tags in JSON mode
        raw_response = MagicMock()
        raw_response.json.return_value = {
            "response": "<think>Planning my JSON response</think>This is not valid JSON",
        }

        # Create properly structured model response object
        model_response = ModelResponse(
            id="test_id",
            choices=[{"message": Message(content="")}],
        )

        # Create mock encoding
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]

        # Transform response
        result = config.transform_response(
            model="llama2",
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={"format": "json"},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=mock_encoding,
        )

        # Verify reasoning content is extracted even in JSON mode when JSON parsing fails
        assert (
            result.choices[0]["message"].reasoning_content
            == "Planning my JSON response"
        )
        assert result.choices[0]["message"].content == "This is not valid JSON"
        assert result.choices[0]["finish_reason"] == "stop"

    def test_transform_response_no_thinking_tags(self):
        """Test that responses without thinking tags work normally."""
        # Initialize config
        config = OllamaConfig()

        # Create mock response without thinking tags
        raw_response = MagicMock()
        raw_response.json.return_value = {
            "response": "Regular response without any thinking tags",
        }

        # Create properly structured model response object
        model_response = ModelResponse(
            id="test_id",
            choices=[{"message": Message(content="")}],
        )

        # Create mock encoding
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]

        # Transform response
        result = config.transform_response(
            model="llama2",
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=mock_encoding,
        )

        # Verify no reasoning content is extracted
        assert result.choices[0]["message"].reasoning_content is None
        assert (
            result.choices[0]["message"].content
            == "Regular response without any thinking tags"
        )
        assert result.choices[0]["finish_reason"] == "stop"


class TestOllamaTextCompletionResponseIterator:
    def test_chunk_parser_with_thinking_field(self):
        """Test that chunks with 'thinking' field and empty 'response' are handled correctly."""
        iterator = OllamaTextCompletionResponseIterator(
            streaming_response=iter([]), sync_stream=True, json_mode=False
        )

        # Test chunk with thinking field - this is the problematic case from the issue
        chunk_with_thinking = {
            "model": "gpt-oss:20b",
            "created_at": "2025-08-06T14:34:31.5276077Z",
            "response": "",
            "thinking": "User",
            "done": False,
        }

        result = iterator.chunk_parser(chunk_with_thinking)

        # Should return a ModelResponseStream with reasoning content
        assert isinstance(result, ModelResponseStream)
        assert result.choices and result.choices[0].delta is not None
        assert getattr(result.choices[0].delta, "reasoning_content") == "User"

    def test_chunk_parser_normal_response(self):
        """Test that normal response chunks still work."""
        iterator = OllamaTextCompletionResponseIterator(
            streaming_response=iter([]), sync_stream=True, json_mode=False
        )

        # Test normal chunk with response
        normal_chunk = {
            "model": "llama2",
            "created_at": "2025-08-06T14:34:31.5276077Z",
            "response": "Hello world",
            "done": False,
        }

        result = iterator.chunk_parser(normal_chunk)

        # Updated to handle ModelResponseStream return type
        assert isinstance(result, ModelResponseStream)
        assert result.choices and result.choices[0].delta is not None
        assert result.choices[0].delta.content == "Hello world"
        assert getattr(result.choices[0].delta, "reasoning_content", None) is None

    def test_chunk_parser_done_chunk(self):
        """Test that done chunks work correctly."""
        iterator = OllamaTextCompletionResponseIterator(
            streaming_response=iter([]), sync_stream=True, json_mode=False
        )

        # Test done chunk
        done_chunk = {
            "model": "llama2",
            "created_at": "2025-08-06T14:34:31.5276077Z",
            "response": "",
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 5,
        }

        result = iterator.chunk_parser(done_chunk)

        assert result["text"] == ""
        assert result["is_finished"] is True
        assert result["finish_reason"] == "stop"
        assert result["usage"] is not None
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5
        assert result["usage"]["total_tokens"] == 15
