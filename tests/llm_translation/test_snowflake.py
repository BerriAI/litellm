import os
import sys
import json
import httpx
from typing import Any, Dict, List
from unittest.mock import Mock, MagicMock, patch
from dotenv import load_dotenv

load_dotenv()
import pytest

from litellm import completion, acompletion, responses
from litellm.exceptions import APIConnectionError
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler


def mock_snowflake_chat_response() -> Dict[str, Any]:
    """
    Mock response for Snowflake chat completion.
    """
    return {
        "id": "chatcmpl-snowflake-123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "mistral-7b",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The sky above is painted blue,\nWith clouds of white and morning dew.\nA canvas vast, serene and bright,\nThat fills my heart with pure delight.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 30,
            "total_tokens": 40,
        },
    }


def mock_snowflake_streaming_response_chunks() -> List[str]:
    """
    Mock streaming response chunks for Snowflake.
    """
    return [
        json.dumps({
            "id": "chatcmpl-snowflake-stream-123",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "mistral-7b",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "The"},
                    "finish_reason": None,
                }
            ],
        }),
        json.dumps({
            "id": "chatcmpl-snowflake-stream-123",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "mistral-7b",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": " sky"},
                    "finish_reason": None,
                }
            ],
        }),
        json.dumps({
            "id": "chatcmpl-snowflake-stream-123",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "mistral-7b",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": " is blue"},
                    "finish_reason": "stop",
                }
            ],
        }),
    ]


@pytest.mark.parametrize("sync_mode", [True, False])
def test_chat_completion_snowflake(sync_mode):
    """
    Test Snowflake chat completion with mocked HTTP responses.
    """
    messages = [
        {
            "role": "user",
            "content": "Write me a poem about the blue sky",
        },
    ]

    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = mock_snowflake_chat_response()

    if sync_mode:
        sync_handler = HTTPHandler()
        with patch.object(HTTPHandler, "post", return_value=mock_response):
            response = completion(
                model="snowflake/mistral-7b",
                messages=messages,
                api_base="https://exampleopenaiendpoint-production.up.railway.app/v1/chat/completions",
                client=sync_handler,
            )
            assert response is not None
            assert response.choices[0].message.content is not None
            assert "sky" in response.choices[0].message.content.lower()
    else:
        async_handler = AsyncHTTPHandler()
        with patch.object(AsyncHTTPHandler, "post", return_value=mock_response):
            import asyncio
            response = asyncio.run(
                acompletion(
                    model="snowflake/mistral-7b",
                    messages=messages,
                    api_base="https://exampleopenaiendpoint-production.up.railway.app/v1/chat/completions",
                    client=async_handler,
                )
            )
            assert response is not None
            assert response.choices[0].message.content is not None
            assert "sky" in response.choices[0].message.content.lower()


@pytest.mark.parametrize("sync_mode", [True, False])
def test_chat_completion_snowflake_stream(sync_mode):
    """
    Test Snowflake streaming chat completion with mocked HTTP responses.
    """
    messages = [
        {
            "role": "user",
            "content": "Write me a poem about the blue sky",
        },
    ]

    if sync_mode:
        sync_handler = HTTPHandler()
        mock_chunks = mock_snowflake_streaming_response_chunks()
        
        def mock_iter_lines():
            for chunk in mock_chunks:
                for line in [f"data: {chunk}", "data: [DONE]"]:
                    yield line
        
        mock_response = MagicMock()
        mock_response.iter_lines.side_effect = mock_iter_lines
        mock_response.status_code = 200
        
        with patch.object(HTTPHandler, "post", return_value=mock_response):
            response = completion(
                model="snowflake/mistral-7b",
                messages=messages,
                max_tokens=100,
                stream=True,
                api_base="https://exampleopenaiendpoint-production.up.railway.app/v1/chat/completions",
                client=sync_handler,
            )
            
            chunks_received = []
            for chunk in response:
                chunks_received.append(chunk)
            
            assert len(chunks_received) > 0
    else:
        async_handler = AsyncHTTPHandler()
        mock_chunks = mock_snowflake_streaming_response_chunks()
        
        async def mock_iter_lines():
            for chunk in mock_chunks:
                for line in [f"data: {chunk}", "data: [DONE]"]:
                    yield line
        
        mock_response = MagicMock()
        mock_response.iter_lines.side_effect = mock_iter_lines
        mock_response.status_code = 200
        
        with patch.object(AsyncHTTPHandler, "post", return_value=mock_response):
            import asyncio
            
            async def test_async_stream():
                response = await acompletion(
                    model="snowflake/mistral-7b",
                    messages=messages,
                    max_tokens=100,
                    stream=True,
                    api_base="https://exampleopenaiendpoint-production.up.railway.app/v1/chat/completions",
                    client=async_handler,
                )
                
                chunks_received = []
                async for chunk in response:
                    chunks_received.append(chunk)
                
                assert len(chunks_received) > 0
            
            asyncio.run(test_async_stream())


@pytest.mark.skip(reason="Requires Snowflake credentials - run manually when needed")
def test_snowflake_tool_calling_responses_api():
    """
    Test Snowflake tool calling with Responses API.
    Requires SNOWFLAKE_JWT and SNOWFLAKE_ACCOUNT_ID environment variables.
    """
    import litellm

    # Skip if credentials not available
    if not os.getenv("SNOWFLAKE_JWT") or not os.getenv("SNOWFLAKE_ACCOUNT_ID"):
        pytest.skip("Snowflake credentials not available")

    litellm.drop_params = False  # We now support tools!

    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"],
            },
        }
    ]

    try:
        # Test with tool_choice to force tool use
        response = responses(
            model="snowflake/claude-3-5-sonnet",
            input="What's the weather in Paris?",
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
            max_output_tokens=200,
        )

        assert response is not None
        assert hasattr(response, "output")
        assert len(response.output) > 0

        # Verify tool call was made
        tool_call_found = False
        for item in response.output:
            if hasattr(item, "type") and item.type == "function_call":
                tool_call_found = True
                assert item.name == "get_weather"
                assert hasattr(item, "arguments")
                print(f"✅ Tool call detected: {item.name}({item.arguments})")
                break

        assert tool_call_found, "Expected tool call but none was found"

    except APIConnectionError as e:
        if "JWT token is invalid" in str(e):
            pytest.skip("Invalid Snowflake JWT token")
        elif "Application failed to respond" in str(e) or "502" in str(e):
            pytest.skip(f"Snowflake API unavailable: {e}")
        else:
            raise
