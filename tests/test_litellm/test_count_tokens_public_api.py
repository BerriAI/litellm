"""
Tests for litellm.acount_tokens() public API.
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.types.utils import TokenCountResponse


def test_acount_tokens_routes_to_openai():
    """Test that acount_tokens routes to OpenAI token counter for openai/ models."""
    with patch(
        "litellm.llms.openai.responses.count_tokens.token_counter.openai_count_tokens_handler.handle_count_tokens_request",
        new_callable=AsyncMock,
        return_value={"input_tokens": 15},
    ):
        result = asyncio.run(
            litellm.acount_tokens(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                api_key="sk-test-key",
            )
        )

        assert result.total_tokens == 15
        assert result.tokenizer_type == "openai_api"
        assert result.request_model == "openai/gpt-4o"


def test_acount_tokens_routes_to_anthropic():
    """Test that acount_tokens routes to Anthropic token counter for anthropic/ models."""
    with patch(
        "litellm.llms.anthropic.count_tokens.token_counter.anthropic_count_tokens_handler.handle_count_tokens_request",
        new_callable=AsyncMock,
        return_value={"input_tokens": 20},
    ):
        result = asyncio.run(
            litellm.acount_tokens(
                model="anthropic/claude-3-5-sonnet-20241022",
                messages=[{"role": "user", "content": "Hello Claude!"}],
                api_key="sk-ant-test-key",
            )
        )

        assert result.total_tokens == 20
        assert result.tokenizer_type == "anthropic_api"
        assert result.request_model == "anthropic/claude-3-5-sonnet-20241022"


def test_acount_tokens_fallback_to_local():
    """Test that unsupported providers fall back to local tiktoken counting."""
    result = asyncio.run(
        litellm.acount_tokens(
            model="together_ai/meta-llama/Llama-3-8b-chat-hf",
            messages=[{"role": "user", "content": "Hello"}],
        )
    )

    assert result.total_tokens > 0
    assert result.tokenizer_type == "local_tokenizer"


def test_acount_tokens_with_tools():
    """Test that tools are passed through to the token counter."""
    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather info",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
    ]

    with patch(
        "litellm.llms.openai.responses.count_tokens.token_counter.openai_count_tokens_handler.handle_count_tokens_request",
        new_callable=AsyncMock,
        return_value={"input_tokens": 30},
    ) as mock_handler:
        result = asyncio.run(
            litellm.acount_tokens(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "What's the weather?"}],
                tools=tools,
                api_key="sk-test-key",
            )
        )

        assert result.total_tokens == 30
        mock_handler.assert_called_once()
        call_kwargs = mock_handler.call_args
        assert call_kwargs.kwargs.get("tools") == tools


def test_acount_tokens_with_system():
    """Test that system messages are passed through."""
    with patch(
        "litellm.llms.openai.responses.count_tokens.token_counter.openai_count_tokens_handler.handle_count_tokens_request",
        new_callable=AsyncMock,
        return_value={"input_tokens": 25},
    ):
        result = asyncio.run(
            litellm.acount_tokens(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                system="You are a helpful assistant.",
                api_key="sk-test-key",
            )
        )

        assert result.total_tokens == 25


def test_acount_tokens_api_error_falls_back():
    """Test that API errors in token counting return error response."""
    from litellm.llms.openai.common_utils import OpenAIError

    with patch(
        "litellm.llms.openai.responses.count_tokens.token_counter.openai_count_tokens_handler.handle_count_tokens_request",
        new_callable=AsyncMock,
        side_effect=OpenAIError(status_code=401, message="Invalid API key"),
    ):
        result = asyncio.run(
            litellm.acount_tokens(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                api_key="sk-bad-key",
            )
        )

        # Should fall back to local tokenizer when provider API errors
        assert result.error is False
        assert result.tokenizer_type == "local_tokenizer"
        assert result.total_tokens > 0


def test_acount_tokens_no_api_key_falls_back():
    """Test that missing API key falls back to local counting."""
    env_backup = os.environ.pop("OPENAI_API_KEY", None)
    try:
        result = asyncio.run(
            litellm.acount_tokens(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
            )
        )

        # Should fall back to local tokenizer since no API key
        assert result.total_tokens > 0
        assert result.tokenizer_type == "local_tokenizer"
    finally:
        if env_backup:
            os.environ["OPENAI_API_KEY"] = env_backup
