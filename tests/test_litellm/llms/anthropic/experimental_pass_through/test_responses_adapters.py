import pytest
import json
import httpx
from unittest.mock import patch
from litellm.llms.anthropic.experimental_pass_through.responses_adapters.handler import (
    LiteLLMMessagesToResponsesAPIHandler,
)



@pytest.fixture
def mock_httpx_post():
    with patch("litellm.llms.custom_httpx.llm_http_handler.AsyncHTTPHandler.post") as mock_post:
        yield mock_post


@pytest.mark.asyncio
async def test_responses_api_format_valid(mock_httpx_post):
    """
    Test that a valid Responses API format (with "object": "response" and "output")
    is successfully parsed into an Anthropic Messages response.
    """
    mock_response_data = {
        "object": "response",
        "model": "deepseek-v4-flash-free",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Hello."}],
            }
        ],
        "stop_reason": "stop",
    }

    mock_response = httpx.Response(
        status_code=200,
        content=json.dumps(mock_response_data).encode("utf-8"),
        request=httpx.Request("POST", "https://opencode.ai/zen/v1/responses"),
    )
    mock_httpx_post.return_value = mock_response

    # Call the handler directly
    response = await LiteLLMMessagesToResponsesAPIHandler.async_anthropic_messages_handler(
        max_tokens=100,
        messages=[{"role": "user", "content": "Hi"}],
        model="openai/deepseek-v4-flash-free",
        api_base="https://opencode.ai/zen/v1",
        api_key="test-key",
    )

    assert response["type"] == "message"
    assert response["role"] == "assistant"
    assert len(response["content"]) == 1
    assert response["content"][0]["text"] == "Hello."
    assert response["stop_reason"] == "end_turn"


@pytest.mark.asyncio
async def test_responses_api_empty_output_max_tokens(mock_httpx_post):
    """
    Test that an empty output[] with stop_reason: "max_output_tokens"
    returns an Anthropic response with stop_reason: "max_tokens"
    """
    mock_response_data = {
        "object": "response",
        "model": "deepseek-v4-flash-free",
        "output": [],
        "stop_reason": "max_output_tokens",
    }

    mock_response = httpx.Response(
        status_code=200,
        content=json.dumps(mock_response_data).encode("utf-8"),
        request=httpx.Request("POST", "https://opencode.ai/zen/v1/responses"),
    )
    mock_httpx_post.return_value = mock_response

    response = await LiteLLMMessagesToResponsesAPIHandler.async_anthropic_messages_handler(
        max_tokens=100,
        messages=[{"role": "user", "content": "Hi"}],
        model="openai/deepseek-v4-flash-free",
        api_base="https://opencode.ai/zen/v1",
        api_key="test-key",
    )

    assert response["type"] == "message"
    assert response["role"] == "assistant"
    assert len(response["content"]) == 0
    assert response["stop_reason"] == "max_tokens"
