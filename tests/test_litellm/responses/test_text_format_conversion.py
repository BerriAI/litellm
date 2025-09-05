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
        from unittest.mock import AsyncMock, patch

        class TestResponse(BaseModel):
            """Test Pydantic model for structured output"""

            answer: str
            confidence: float

        class MockResponse:
            """Mock response class for testing"""

            def __init__(self, json_data, status_code):
                self._json_data = json_data
                self.status_code = status_code
                self.text = json.dumps(json_data)

            def json(self):
                return self._json_data

        # Mock response from OpenAI
        mock_response = {
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

        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post:
            # Configure the mock to return our response
            mock_post.return_value = MockResponse(mock_response, 200)

            litellm._turn_on_debug()
            litellm.set_verbose = True

            # Call aresponses with text_format parameter
            response = await litellm.aresponses(
                input="What is the capital of France?",
                text_format=TestResponse,
                **base_completion_call_args,
            )

            # Verify the request was made correctly
            mock_post.assert_called_once()
            request_body = mock_post.call_args.kwargs["json"]
            print("Request body:", json.dumps(request_body, indent=4))

            # Validate that text_format was converted to text parameter
            assert (
                "text" in request_body
            ), "text parameter should be present in request body"
            assert (
                "text_format" not in request_body
            ), "text_format should not be in request body"

            # Validate the text parameter structure
            text_param = request_body["text"]
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
            assert request_body["input"] == "What is the capital of France?"

            # Validate the response
            print("Response:", json.dumps(response, indent=4, default=str))
