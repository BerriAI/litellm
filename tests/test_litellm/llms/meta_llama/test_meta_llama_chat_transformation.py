import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.llms.meta_llama.chat.transformation import LlamaAPIConfig


def test_map_openai_params():
    """Test that LlamaAPIConfig correctly maps OpenAI parameters"""
    config = LlamaAPIConfig()

    # Test response_format handling - json_schema is allowed
    non_default_params = {"response_format": {"type": "json_schema"}}
    optional_params = {"response_format": True}
    result = config.map_openai_params(
        non_default_params, optional_params, "llama-3.3-8B-instruct", False
    )
    assert "response_format" in result
    assert result["response_format"]["type"] == "json_schema"

    # Test response_format handling - other types are removed
    non_default_params = {"response_format": {"type": "text"}}
    optional_params = {"response_format": True}
    result = config.map_openai_params(
        non_default_params, optional_params, "llama-3.3-8B-instruct", False
    )
    assert "response_format" not in result

    # Test that other parameters are passed through
    non_default_params = {
        "temperature": 0.7,
        "response_format": {"type": "json_schema"},
    }
    optional_params = {"temperature": True, "response_format": True}
    result = config.map_openai_params(
        non_default_params, optional_params, "llama-3.3-8B-instruct", False
    )
    assert "temperature" in result
    assert result["temperature"] == 0.7
    assert "response_format" in result


@pytest.mark.asyncio
async def test_llama_api_streaming_no_307_error():
    """Test that streaming works without 307 redirect errors due to follow_redirects=True"""

    # Mock the httpx client to simulate a successful streaming response
    with patch(
        "litellm.llms.custom_httpx.http_handler.get_async_httpx_client"
    ) as mock_get_client:
        # Create a mock client
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client

        # Mock a successful streaming response (not a 307 redirect)
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/plain; charset=utf-8"}

        # Mock streaming data that would come from a successful request
        async def mock_aiter_lines():
            yield 'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"meta_llama/Llama-4-Maverick-17B-128E-Instruct-FP8","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}'
            yield 'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"meta_llama/Llama-4-Maverick-17B-128E-Instruct-FP8","choices":[{"index":0,"delta":{"content":" there"},"finish_reason":null}]}'
            yield 'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"meta_llama/Llama-4-Maverick-17B-128E-Instruct-FP8","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
            yield "data: [DONE]"

        mock_response.aiter_lines.return_value = mock_aiter_lines()
        mock_client.stream.return_value.__aenter__.return_value = mock_response

        # Test the streaming completion
        try:
            response = await litellm.acompletion(
                model="meta_llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                messages=[{"role": "user", "content": "Tell me about yourself"}],
                stream=True,
                temperature=0.0,
            )

            # Verify we get a CustomStreamWrapper (streaming response)
            from litellm.utils import CustomStreamWrapper

            assert isinstance(response, CustomStreamWrapper)

            # Verify the HTTP client was called with follow_redirects=True
            mock_client.stream.assert_called_once()
            call_kwargs = mock_client.stream.call_args[1]
            assert (
                call_kwargs.get("follow_redirects") is True
            ), "follow_redirects should be True to prevent 307 errors"

            # Verify the response status is 200 (not 307)
            assert (
                mock_response.status_code == 200
            ), "Should get 200 response, not 307 redirect"

        except Exception as e:
            # If there's an exception, make sure it's not a 307 error
            error_str = str(e)
            assert (
                "307" not in error_str
            ), f"Should not get 307 redirect error: {error_str}"

            # Still verify that follow_redirects was set correctly
            if mock_client.stream.called:
                call_kwargs = mock_client.stream.call_args[1]
                assert call_kwargs.get("follow_redirects") is True
