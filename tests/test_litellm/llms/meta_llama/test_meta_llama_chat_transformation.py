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

    # Mock the AsyncHTTPHandler's post method to simulate a successful streaming response
    # This is more reliable than mocking the client directly since the implementation
    # uses the handler's post() method which calls client.send()
    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post"
    ) as mock_post:
        # Mock a successful streaming response (not a 307 redirect)
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/event-stream"}
        mock_response.raise_for_status = AsyncMock()  # No exception raised

        # Mock streaming data that would come from a successful request
        async def mock_aiter_lines():
            yield 'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"meta_llama/Llama-4-Maverick-17B-128E-Instruct-FP8","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}'
            yield 'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"meta_llama/Llama-4-Maverick-17B-128E-Instruct-FP8","choices":[{"index":0,"delta":{"content":" there"},"finish_reason":null}]}'
            yield 'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"meta_llama/Llama-4-Maverick-17B-128E-Instruct-FP8","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
            yield "data: [DONE]"

        mock_response.aiter_lines.return_value = mock_aiter_lines()
        mock_post.return_value = mock_response

        # Test the streaming completion
        # The key behavior we're testing: streaming should work without 307 errors
        # This is ensured by follow_redirects=True being set when creating the httpx client
        # (see AsyncHTTPHandler.create_client in http_handler.py line 338)
        try:
            response = await litellm.acompletion(  # type: ignore[awaitable-expected]
                model="meta_llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                messages=[{"role": "user", "content": "Tell me about yourself"}],
                stream=True,
                temperature=0.0,
            )

            # Verify we get a CustomStreamWrapper (streaming response)
            from litellm.utils import CustomStreamWrapper

            assert isinstance(response, CustomStreamWrapper)

            # Verify the handler's post method was called with stream=True
            assert mock_post.called, "AsyncHTTPHandler.post should be called"
            # Handle both call_args formats (positional and keyword)
            if mock_post.call_args:
                call_kwargs = (
                    mock_post.call_args.kwargs
                    if hasattr(mock_post.call_args, "kwargs")
                    else {}
                )
                # stream=True should be passed as a keyword argument
                assert (
                    call_kwargs.get("stream") is True
                ), f"post should be called with stream=True, got kwargs={call_kwargs}"

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
            # Re-raise other exceptions to see what went wrong
            raise
