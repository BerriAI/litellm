import json
import os
import sys
import uuid
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.ollama.completion.transformation import OllamaConfig
from litellm.types.utils import Message, ModelResponse


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
