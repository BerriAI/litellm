import json
import os
import sys

import pytest
from pydantic import BaseModel

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.types.llms.openai import (
    IncompleteDetails,
    ResponseAPIUsage,
    ResponsesAPIResponse,
)


class TestTextFormatConversion:
    """Test text_format to text parameter conversion for responses API"""

    def get_base_completion_call_args(self):
        """Get base arguments for completion call"""
        return {
            "model": "gpt-4o",
            "api_key": "test-key",
            "api_base": "https://api.openai.com/v1",
        }

    @pytest.mark.asyncio
    async def test_text_format_to_text_conversion(self):
        """
        Test that when text_format parameter is passed to litellm.aresponses,
        it gets converted to text parameter in the raw API call to OpenAI.
        """
        from unittest.mock import AsyncMock, MagicMock, patch

        class TestResponse(BaseModel):
            """Test Pydantic model for structured output"""

            answer: str
            confidence: float

        # Mock response from OpenAI
        mock_response_data = {
            "id": "resp_123",
            "object": "response",
            "created_at": 1741476542,
            "status": "completed",
            "model": "gpt-4o",
            "output": [
                {
                    "type": "message",
                    "id": "msg_123",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": '{"answer": "Paris", "confidence": 0.95}',
                            "annotations": [],
                        }
                    ],
                }
            ],
            "parallel_tool_calls": True,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
                "output_tokens_details": {"reasoning_tokens": 0},
            },
            "text": {"format": {"type": "json_object"}},
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "metadata": {},
            "temperature": 1.0,
            "tool_choice": "auto",
            "tools": [],
            "top_p": 1.0,
            "max_output_tokens": None,
            "previous_response_id": None,
            "reasoning": {"effort": None, "summary": None},
            "truncation": "disabled",
            "user": None,
        }

        base_completion_call_args = self.get_base_completion_call_args()

        # Mock the response_api_handler function to capture the request
        captured_request = {}

        def mock_handler(
            model,
            input,
            responses_api_provider_config,
            response_api_optional_request_params,
            custom_llm_provider,
            litellm_params,
            logging_obj,
            extra_headers=None,
            extra_body=None,
            timeout=None,
            client=None,
            fake_stream=False,
            litellm_metadata=None,
            shared_session=None,
            _is_async=False,
        ):
            # Capture the request parameters
            captured_request["model"] = model
            captured_request["input"] = input
            captured_request["params"] = response_api_optional_request_params
            
            # Return a mock ResponsesAPIResponse wrapped in a coroutine if async
            async def async_response():
                return ResponsesAPIResponse(
                    id="resp_123",
                    object="response",
                    created_at=1741476542,
                    status="completed",
                    model="gpt-4o",
                    output=mock_response_data["output"],
                    usage=ResponseAPIUsage(
                        input_tokens=10,
                        output_tokens=20,
                        total_tokens=30,
                    ),
                    text=mock_response_data.get("text"),
                    error=None,
                    incomplete_details=None,
                )
            
            if _is_async:
                return async_response()
            else:
                return ResponsesAPIResponse(
                    id="resp_123",
                    object="response",
                    created_at=1741476542,
                    status="completed",
                    model="gpt-4o",
                    output=mock_response_data["output"],
                    usage=ResponseAPIUsage(
                        input_tokens=10,
                        output_tokens=20,
                        total_tokens=30,
                    ),
                    text=mock_response_data.get("text"),
                    error=None,
                    incomplete_details=None,
                )

        with patch(
            "litellm.responses.main.base_llm_http_handler.response_api_handler",
            new=mock_handler,
        ):
            litellm._turn_on_debug()
            litellm.set_verbose = True

            # Call aresponses with text_format parameter
            response = await litellm.aresponses(
                input="What is the capital of France?",
                text_format=TestResponse,
                **base_completion_call_args,
            )

            # Verify the captured request
            print("Captured request:", json.dumps(captured_request, indent=4, default=str))

            # Validate that text_format was converted to text parameter
            assert (
                "text" in captured_request["params"]
            ), "text parameter should be present in request params"
            assert (
                "text_format" not in captured_request["params"]
            ), "text_format should not be in request params"

            # Validate the text parameter structure
            text_param = captured_request["params"]["text"]
            assert "format" in text_param, "text parameter should have format field"
            assert (
                text_param["format"]["type"] == "json_schema"
            ), "format type should be json_schema"
            assert "name" in text_param["format"], "format should have name field"
            assert (
                text_param["format"]["name"] == "TestResponse"
            ), "format name should match Pydantic model name"
            assert "schema" in text_param["format"], "format should have schema field"
            assert "strict" in text_param["format"], "format should have strict field"

            # Validate the schema structure
            schema = text_param["format"]["schema"]
            assert schema["type"] == "object", "schema type should be object"
            assert "properties" in schema, "schema should have properties"
            assert (
                "answer" in schema["properties"]
            ), "schema should have answer property"
            assert (
                "confidence" in schema["properties"]
            ), "schema should have confidence property"

            # Validate other request parameters
            assert captured_request["input"] == "What is the capital of France?"

            # Validate the response
            print("Response:", json.dumps(response, indent=4, default=str))
