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
            "model": "azure/gpt-4.1-mini",
            "truncation": "auto",
            "api_base": os.getenv("AZURE_API_BASE"),
            "api_key": os.getenv("AZURE_API_KEY"),
            "api_version": "2025-03-01-preview",
        }


@pytest.mark.asyncio
async def test_azure_responses_api_preview_api_version():
    """
    Ensure new azure preview api version is working
    """
    litellm._turn_on_debug()
    response = await litellm.aresponses(
        model="azure/gpt-5-mini",
        truncation="auto",
        api_version="preview",
        api_base=os.getenv("AZURE_API_BASE"),
        api_key=os.getenv("AZURE_API_KEY"),
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


@pytest.mark.asyncio
async def test_azure_responses_api_headers_with_llm_provider_prefix():
    """
    Test that Azure-specific headers like 'x-request-id' and 'apim-request-id'
    are properly forwarded with 'llm_provider-' prefix in response._hidden_params["headers"].
    
    Issue: https://github.com/BerriAI/litellm/issues/16538
    
    The fix ensures that processed headers (with llm_provider- prefix) are stored
    in response._hidden_params["headers"] instead of additional_headers, making them
    accessible via completion.headers in the same way as the completion API.
    """
    import json
    import httpx

    mock_response_data = {
        "id": "resp_123",
        "object": "response",
        "created_at": 1234567890,
        "model": "gpt-5-codex",
        "status": "completed",
        "output": [
            {
                "id": "msg_123",
                "role": "assistant",
                "type": "message",
                "content": [{"type": "output_text", "text": "Hello!"}],
            }
        ],
    }

    # Mock headers that Azure returns - exactly like in the issue
    mock_headers = {
        "date": "Wed, 12 Nov 2025 15:31:28 GMT",
        "server": "uvicorn",
        "content-type": "application/json",
        "x-ratelimit-remaining-tokens": "5010000",
        "x-ratelimit-limit-tokens": "5010000",
        # These are the Azure-specific headers that should be forwarded with llm_provider- prefix
        "x-request-id": "12086715-aca3-4006-a29f-2f1e1d552043",
        "apim-request-id": "25664b0d-cf4b-4e10-8d27-c7272e7efd49",
        "x-ms-region": "Sweden Central",
    }

    async def mock_post(*args, **kwargs):
        response_content = json.dumps(mock_response_data).encode("utf-8")
        response = httpx.Response(
            status_code=200,
            headers=mock_headers,
            content=response_content,
            request=httpx.Request(method="POST", url="https://test.openai.azure.com"),
        )
        return response

    with patch.object(AsyncHTTPHandler, "post", new=mock_post):
        response = await litellm.aresponses(
            model="azure/gpt-5-codex",
            api_version="2025-03-01-preview",
            api_base="https://test.openai.azure.com",
            api_key="test-key",
            input="Hello, can you tell me a short joke?",
        )

    # Check that the response has the expected headers structure
    assert hasattr(response, "_hidden_params"), "Response should have _hidden_params"
    assert "additional_headers" in response._hidden_params, (
        "Response _hidden_params should contain 'additional_headers' with the LLM provider headers"
    )

    headers = response._hidden_params["additional_headers"]
    
    # Verify that Azure-specific headers are present with llm_provider- prefix
    assert "llm_provider-x-request-id" in headers, (
        f"Response should contain 'llm_provider-x-request-id' header. "
        f"Headers: {list(headers.keys())}"
    )
    assert "llm_provider-apim-request-id" in headers, (
        f"Response should contain 'llm_provider-apim-request-id' header. "
        f"Headers: {list(headers.keys())}"
    )
    
    # Verify the header values match
    assert headers["llm_provider-x-request-id"] == "12086715-aca3-4006-a29f-2f1e1d552043"
    assert headers["llm_provider-apim-request-id"] == "25664b0d-cf4b-4e10-8d27-c7272e7efd49"
    assert headers["llm_provider-x-ms-region"] == "Sweden Central"
    
    # Also verify openai-compatible headers are included
    assert "x-ratelimit-limit-tokens" in headers
    assert "x-ratelimit-remaining-tokens" in headers
