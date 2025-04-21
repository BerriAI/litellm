import os
import sys
import pytest
import asyncio
from typing import Optional
from unittest.mock import patch, AsyncMock

sys.path.insert(0, os.path.abspath("../.."))
import litellm
from litellm.integrations.custom_logger import CustomLogger
import json
from litellm.types.utils import StandardLoggingPayload
from litellm.types.llms.openai import (
    ResponseCompletedEvent,
    ResponsesAPIResponse,
    ResponseTextConfig,
    ResponseAPIUsage,
    IncompleteDetails,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from base_responses_api import BaseResponsesAPITest

class TestAzureResponsesAPITest(BaseResponsesAPITest):
    def get_base_completion_call_args(self):
        return {
            "model": "azure/computer-use-preview",
            "truncation": "auto",
            "api_base": os.getenv("AZURE_RESPONSES_OPENAI_ENDPOINT"),
            "api_key": os.getenv("AZURE_RESPONSES_OPENAI_API_KEY"),
            "api_version": os.getenv("AZURE_RESPONSES_OPENAI_API_VERSION"),
        }


@pytest.mark.asyncio
async def test_responses_api_routing_with_previous_response_id():
    """
    Test that when using a previous_response_id, the request is sent to the same model_id
    """
    # Create a mock response that simulates Azure responses API
    mock_response_id = "mock-resp-456"
    
    mock_response_data = {
        "id": mock_response_id,
        "object": "response",
        "created_at": 1741476542,
        "status": "completed",
        "model": "azure/computer-use-preview",
        "output": [
            {
                "type": "message",
                "id": "msg_123",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "I'm doing well, thank you for asking!", "annotations": []}
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
        "text": {"format": {"type": "text"}},
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
    
    class MockResponse:
        def __init__(self, json_data, status_code):
            self._json_data = json_data
            self.status_code = status_code
            self.text = json.dumps(json_data)

        def json(self):
            return self._json_data

    router = litellm.Router(
        model_list=[
            {
                "model_name": "azure-computer-use-preview",
                "litellm_params": {
                    "model": "azure/computer-use-preview",
                    "api_key": "mock-api-key",
                    "api_version": "mock-api-version",
                    "api_base": "https://mock-endpoint.openai.azure.com",
                },
            },
            {
                "model_name": "azure-computer-use-preview",
                "litellm_params": {
                    "model": "azure/computer-use-preview-2",
                    "api_key": "mock-api-key-2",
                    "api_version": "mock-api-version-2",
                    "api_base": "https://mock-endpoint-2.openai.azure.com",
                },
            },
        ],
    )
    MODEL = "azure-computer-use-preview"

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        # Configure the mock to return our response
        mock_post.return_value = MockResponse(mock_response_data, 200)
        
        # Make the initial request
        litellm._turn_on_debug()
        response = await router.aresponses(
            model=MODEL,
            input="Hello, how are you?",
            truncation="auto",
        )
        
        # Store the model_id from the response
        expected_model_id = response._hidden_params["model_id"]
        response_id = response.id

        print("Response ID=", response_id, "came from model_id=", expected_model_id)


        # Make 10 other requests with previous_response_id, assert that they are sent to the same model_id
        for i in range(10):
            # Reset the mock for the next call
            mock_post.reset_mock()
            
            # Set up the mock to return our response again
            mock_post.return_value = MockResponse(mock_response_data, 200)
            
            response = await router.aresponses(
                model=MODEL,
                input=f"Follow-up question {i+1}",
                truncation="auto",
                previous_response_id=response_id,
            )
            
            # Assert the model_id is preserved
            assert response._hidden_params["model_id"] == expected_model_id

  

