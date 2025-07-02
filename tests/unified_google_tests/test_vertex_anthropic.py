import asyncio
import json
import sys
import os
from typing import Any, AsyncIterator, Dict, List, Optional, Union
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.google_genai import (
    agenerate_content,
    agenerate_content_stream
)
from google.genai.types import ContentDict, PartDict, GenerateContentResponse
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import StandardLoggingPayload


def vertex_anthropic_mock_response(*args, **kwargs):
    """Mock response for vertex AI anthropic call"""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "application/json"}
    mock_response.json.return_value = {
        "id": "msg_vrtx_013Wki5RFQXAspL7rmxRFjZg",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4",
        "content": [
            {
                "type": "text",
                "text": "Why don't scientists trust atoms? Because they make up everything!"
            }
        ],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 15, "output_tokens": 20},
    }
    return mock_response


@pytest.mark.asyncio
async def test_vertex_anthropic_mocked():
    """Test agenerate_content with mocked HTTP calls to validate URL and request body"""
    from litellm.llms.custom_httpx.llm_http_handler import AsyncHTTPHandler
    
    # Set up test data
    contents = ContentDict(
        parts=[
            PartDict(
                text="Hello, can you tell me a short joke?"
            )
        ],
        role="user",
    )
    
    # Create HTTP client and mock response
    client = AsyncHTTPHandler()
    httpx_response = AsyncMock()
    httpx_response.side_effect = vertex_anthropic_mock_response
    
    # Expected values for validation
    expected_url = "https://us-east5-aiplatform.googleapis.com/v1/projects/internal-litellm-local-dev/locations/us-east5/publishers/anthropic/models/claude-sonnet-4:rawPredict"
    expected_body_keys = {"messages", "anthropic_version", "max_tokens"}
    expected_message_content = "Hello, can you tell me a short joke?"
    
    # Patch the HTTP client and make the call
    with patch.object(client, "post", new=httpx_response) as mock_call:
        response = await agenerate_content(
            contents=contents,
            model="vertex_ai/claude-sonnet-4",
            vertex_location="us-east5",
            vertex_project="internal-litellm-local-dev",
            custom_llm_provider="vertex_ai",
            client=client,
        )
        
        # Verify the call was made
        assert mock_call.call_count == 1
        
        # Get the call arguments
        call_args = mock_call.call_args
        call_kwargs = call_args.kwargs if call_args else {}
        
        # Extract URL (could be in args[0] or kwargs['url'])
        if call_args and len(call_args[0]) > 0:
            actual_url = call_args[0][0]
        else:
            actual_url = call_kwargs.get("url", "")
        
        # Validate URL
        print(f"Expected URL: {expected_url}")
        print(f"Actual URL: {actual_url}")
        assert actual_url == expected_url, f"Expected URL {expected_url}, but got {actual_url}"
        
        # Validate headers
        actual_headers = call_kwargs.get("headers", {})
        print(f"Actual headers: {actual_headers}")
        

        # Validate Authorization header exists
        auth_header_found = any(k.lower() == "authorization" for k in actual_headers.keys())
        assert auth_header_found, f"Authorization header should be present. Found headers: {list(actual_headers.keys())}"
        
        # Validate request body
        request_body = None
        if "data" in call_kwargs:
            request_body = json.loads(call_kwargs["data"]) if isinstance(call_kwargs["data"], str) else call_kwargs["data"]
        elif "json" in call_kwargs:
            request_body = call_kwargs["json"]
        
        print(f"Request body: {json.dumps(request_body, indent=2)}")
        assert request_body is not None, "Request body should not be None"
        
        # Validate required keys in request body
        actual_body_keys = set(request_body.keys())
        assert expected_body_keys.issubset(actual_body_keys), f"Expected keys {expected_body_keys} not found in {actual_body_keys}"
        
        # Validate message content
        messages = request_body.get("messages", [])
        assert len(messages) > 0, "Messages should not be empty"
        assert messages[0]["role"] == "user", f"Expected first message role to be 'user', got {messages[0]['role']}"
        
        # Check message content structure
        content = messages[0]["content"]
        if isinstance(content, list):
            text_content = next((item["text"] for item in content if item.get("type") == "text"), None)
        else:
            text_content = content
        
        assert text_content == expected_message_content, f"Expected message content '{expected_message_content}', got '{text_content}'"
        
        # Validate anthropic_version
        assert request_body["anthropic_version"] == "vertex-2023-10-16", f"Expected anthropic_version 'vertex-2023-10-16', got {request_body['anthropic_version']}"
        
        # Validate max_tokens
        assert "max_tokens" in request_body, "max_tokens should be present in request body"
        assert isinstance(request_body["max_tokens"], int), f"max_tokens should be integer, got {type(request_body['max_tokens'])}"
        
        print("✅ All validations passed!")
        print(f"Response: {response}")