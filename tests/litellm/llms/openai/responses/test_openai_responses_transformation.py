import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.types.llms.openai import (
    OutputTextDeltaEvent,
    ResponseCompletedEvent,
    ResponsesAPIRequestParams,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)


class TestOpenAIResponsesAPIConfig:
    def setup_method(self):
        self.config = OpenAIResponsesAPIConfig()
        self.model = "gpt-4o"
        self.logging_obj = MagicMock()

    def test_map_openai_params(self):
        """Test that parameters are correctly mapped"""
        test_params = {"input": "Hello world", "temperature": 0.7, "stream": True}

        result = self.config.map_openai_params(
            response_api_optional_params=test_params,
            model=self.model,
            drop_params=False,
        )

        # The function should return the params unchanged
        assert result == test_params

    def validate_responses_api_request_params(self, params, expected_fields):
        """
        Validate that the params dict has the expected structure of ResponsesAPIRequestParams

        Args:
            params: The dict to validate
            expected_fields: Dict of field names and their expected values
        """
        # Check that it's a dict
        assert isinstance(params, dict), "Result should be a dict"

        # Check expected fields have correct values
        for field, value in expected_fields.items():
            assert field in params, f"Missing expected field: {field}"
            assert (
                params[field] == value
            ), f"Field {field} has value {params[field]}, expected {value}"

    def test_transform_responses_api_request(self):
        """Test request transformation"""
        input_text = "What is the capital of France?"
        optional_params = {"temperature": 0.7, "stream": True}

        result = self.config.transform_responses_api_request(
            model=self.model,
            input=input_text,
            response_api_optional_request_params=optional_params,
            litellm_params={},
            headers={},
        )

        # Validate the result has the expected structure and values
        expected_fields = {
            "model": self.model,
            "input": input_text,
            "temperature": 0.7,
            "stream": True,
        }

        self.validate_responses_api_request_params(result, expected_fields)

    def test_transform_streaming_response(self):
        """Test streaming response transformation"""
        # Test with a text delta event
        chunk = {
            "type": "response.output_text.delta",
            "item_id": "item_123",
            "output_index": 0,
            "content_index": 0,
            "delta": "Hello",
        }

        result = self.config.transform_streaming_response(
            model=self.model, parsed_chunk=chunk, logging_obj=self.logging_obj
        )

        assert isinstance(result, OutputTextDeltaEvent)
        assert result.type == ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA
        assert result.delta == "Hello"
        assert result.item_id == "item_123"

        # Test with a completed event - providing all required fields
        completed_chunk = {
            "type": "response.completed",
            "response": {
                "id": "resp_123",
                "created_at": 1234567890,
                "model": "gpt-4o",
                "object": "response",
                "output": [],
                "parallel_tool_calls": False,
                "error": None,
                "incomplete_details": None,
                "instructions": None,
                "metadata": None,
                "temperature": 0.7,
                "tool_choice": "auto",
                "tools": [],
                "top_p": 1.0,
                "max_output_tokens": None,
                "previous_response_id": None,
                "reasoning": None,
                "status": "completed",
                "text": None,
                "truncation": "auto",
                "usage": None,
                "user": None,
            },
        }

        # Mock the get_event_model_class to avoid validation issues in tests
        with patch.object(
            OpenAIResponsesAPIConfig, "get_event_model_class"
        ) as mock_get_class:
            mock_get_class.return_value = ResponseCompletedEvent

            result = self.config.transform_streaming_response(
                model=self.model,
                parsed_chunk=completed_chunk,
                logging_obj=self.logging_obj,
            )

            assert result.type == ResponsesAPIStreamEvents.RESPONSE_COMPLETED
            assert result.response.id == "resp_123"

    def test_validate_environment(self):
        """Test that validate_environment correctly sets the Authorization header"""
        # Test with provided API key
        headers = {}
        api_key = "test_api_key"

        result = self.config.validate_environment(
            headers=headers, model=self.model, api_key=api_key
        )

        assert "Authorization" in result
        assert result["Authorization"] == f"Bearer {api_key}"

        # Test with empty headers
        headers = {}

        with patch("litellm.api_key", "litellm_api_key"):
            result = self.config.validate_environment(headers=headers, model=self.model)

            assert "Authorization" in result
            assert result["Authorization"] == "Bearer litellm_api_key"

        # Test with existing headers
        headers = {"Content-Type": "application/json"}

        with patch("litellm.openai_key", "openai_key"):
            with patch("litellm.api_key", None):
                result = self.config.validate_environment(
                    headers=headers, model=self.model
                )

                assert "Authorization" in result
                assert result["Authorization"] == "Bearer openai_key"
                assert "Content-Type" in result
                assert result["Content-Type"] == "application/json"

        # Test with environment variable
        headers = {}

        with patch("litellm.api_key", None):
            with patch("litellm.openai_key", None):
                with patch(
                    "litellm.llms.openai.responses.transformation.get_secret_str",
                    return_value="env_api_key",
                ):
                    result = self.config.validate_environment(
                        headers=headers, model=self.model
                    )

                    assert "Authorization" in result
                    assert result["Authorization"] == "Bearer env_api_key"

    def test_get_complete_url(self):
        """Test that get_complete_url returns the correct URL"""
        # Test with provided API base
        api_base = "https://custom-openai.example.com/v1"

        result = self.config.get_complete_url(
            api_base=api_base,
            litellm_params={},
        )

        assert result == "https://custom-openai.example.com/v1/responses"

        # Test with litellm.api_base
        with patch("litellm.api_base", "https://litellm-api-base.example.com/v1"):
            result = self.config.get_complete_url(
                api_base=None,
                litellm_params={},
            )

            assert result == "https://litellm-api-base.example.com/v1/responses"

        # Test with environment variable
        with patch("litellm.api_base", None):
            with patch(
                "litellm.llms.openai.responses.transformation.get_secret_str",
                return_value="https://env-api-base.example.com/v1",
            ):
                result = self.config.get_complete_url(
                    api_base=None,
                    litellm_params={},
                )

                assert result == "https://env-api-base.example.com/v1/responses"

        # Test with default API base
        with patch("litellm.api_base", None):
            with patch(
                "litellm.llms.openai.responses.transformation.get_secret_str",
                return_value=None,
            ):
                result = self.config.get_complete_url(
                    api_base=None,
                    litellm_params={},
                )

                assert result == "https://api.openai.com/v1/responses"

        # Test with trailing slash in API base
        api_base = "https://custom-openai.example.com/v1/"

        result = self.config.get_complete_url(
            api_base=api_base,
            litellm_params={},
        )

        assert result == "https://custom-openai.example.com/v1/responses"

    def test_get_event_model_class_generic_event(self):
        """Test that get_event_model_class returns the correct event model class"""
        from litellm.types.llms.openai import GenericEvent

        event_type = "test"
        result = self.config.get_event_model_class(event_type)
        assert result == GenericEvent

    def test_transform_streaming_response_generic_event(self):
        """Test that transform_streaming_response returns the correct event model class"""
        from litellm.types.llms.openai import GenericEvent

        chunk = {"type": "test", "test": "test"}
        result = self.config.transform_streaming_response(
            model=self.model, parsed_chunk=chunk, logging_obj=self.logging_obj
        )
        assert isinstance(result, GenericEvent)
        assert result.type == "test"
