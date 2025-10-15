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
async def test_azure_responses_api_preview_api_version():
    """
    Ensure new azure preview api version is working
    """
    litellm._turn_on_debug()
    response = await litellm.aresponses(
        model="azure/computer-use-preview",
        truncation="auto",
        api_version="preview",
        api_base=os.getenv("AZURE_RESPONSES_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_RESPONSES_OPENAI_API_KEY"),
        input="Hello, can you tell me a short joke?",
    )


@pytest.mark.asyncio
async def test_azure_responses_api_status_error():
    """
    Test that 'status' field is not sent in the final request body to Azure API.
    The status field should be filtered out from input messages before making the API call.
    """
    from unittest.mock import AsyncMock, MagicMock
    import json

    request_data = {
        "model": "computer-use-preview",
        "input": [
            {"content": "tell me an interesting fact", "role": "user"},
            {
                "id": "rs_0ab687487834d9df0068e462a1b2d88197aabbc832c9ba5316",
                "summary": [],
                "type": "reasoning",
                "content": None,
                "encrypted_content": None,
                "status": "completed",
            },
            {
                "id": "msg_0ab687487834d9df0068e462a1df188197b74b1eef05102c18",
                "content": [
                    {
                        "annotations": [],
                        "text": "very good morning",
                        "type": "output_text",
                        "logprobs": [],
                    }
                ],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            },
            {"role": "user", "content": "tell me another"},
        ],
        "include": [],
        "instructions": "You are a helpful assistant.",
        "reasoning": {"effort": "minimal"},
        "stream": False,
        "tools": [],
    }

    # Mock response
    mock_response_data = {
        "id": "resp_123",
        "object": "response",
        "created_at": 1234567890,
        "model": "computer-use-preview",
        "status": "completed",
        "output": [
            {
                "id": "msg_123",
                "role": "assistant",
                "type": "message",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Here's an interesting fact."}],
            }
        ],
    }

    captured_request_body = {}

    async def mock_post(*args, **kwargs):
        # Capture the request body
        nonlocal captured_request_body
        if "json" in kwargs:
            captured_request_body = kwargs["json"]
        elif "data" in kwargs:
            captured_request_body = json.loads(kwargs["data"])

        import httpx
        
        # Create a proper httpx Response object
        response_content = json.dumps(mock_response_data).encode("utf-8")
        response = httpx.Response(
            status_code=200,
            headers={"content-type": "application/json"},
            content=response_content,
            request=httpx.Request(method="POST", url="https://test.openai.azure.com"),
        )
        return response

    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
    from unittest.mock import patch

    with patch.object(AsyncHTTPHandler, "post", new=mock_post):
        response = await litellm.aresponses(
            model="azure/computer-use-preview",
            truncation="auto",
            api_version="preview",
            api_base="https://test.openai.azure.com",
            api_key="test-key",
            input=request_data["input"],
        )

    # Verify that 'status' field is not present in any of the input messages
    print("Final request body:", json.dumps(captured_request_body, indent=4, default=str))
    assert "input" in captured_request_body, "Request body should contain 'input' field"
    
    expected_input = [
        {
            "content": "tell me an interesting fact",
            "role": "user"
        },
        {
            "id": "rs_0ab687487834d9df0068e462a1b2d88197aabbc832c9ba5316",
            "summary": [],
            "type": "reasoning"
        },
        {
            "id": "msg_0ab687487834d9df0068e462a1df188197b74b1eef05102c18",
            "content": [
                {
                    "annotations": [],
                    "text": "very good morning",
                    "type": "output_text",
                    "logprobs": []
                }
            ],
            "role": "assistant",
            "type": "message"
        },
        {
            "role": "user",
            "content": "tell me another"
        }
    ]
    
    assert captured_request_body["input"] == expected_input, (
        f"Request body input should match expected format without 'status' field.\n"
        f"Expected: {json.dumps(expected_input, indent=2)}\n"
        f"Got: {json.dumps(captured_request_body['input'], indent=2)}"
    )
