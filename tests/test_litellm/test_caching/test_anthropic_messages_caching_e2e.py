"""
End-to-end integration tests for anthropic_messages caching flow.

Verifies the full cache lifecycle:
1. Cache miss → provider call → store in cache
2. Cache hit → return from cache (no provider call)
3. Cache bypass via cache={"no-cache": True}
4. Different parameters produce separate cache entries (no false hits)
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

import litellm
from litellm.caching.caching import Cache


def _mock_anthropic_response(text: str = "Hello!", msg_id: str = "msg_test_123"):
    """Return a minimal valid AnthropicMessagesResponse dict."""
    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-20250514",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


@pytest.fixture(autouse=True)
def setup_and_teardown_cache():
    """Set up a local in-memory cache before each test and clean up after."""
    original_cache = litellm.cache
    litellm.cache = Cache(type="local")
    yield
    litellm.cache = original_cache


class TestAnthropicMessagesCachingE2E:
    """End-to-end tests for anthropic_messages caching."""

    @pytest.mark.asyncio
    async def test_non_streaming_cache_miss_then_hit(self):
        """
        First call should hit the provider (cache miss).
        Second identical call should return from cache (no provider call).
        """
        mock_response = _mock_anthropic_response()

        with patch(
            "litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.async_anthropic_messages_handler",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_handler:
            # First call - cache miss, should call provider
            result1 = await litellm.anthropic_messages(
                model="anthropic/claude-sonnet-4-20250514",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=100,
                api_key="fake-key",
            )

            assert mock_handler.call_count == 1
            assert result1["id"] == "msg_test_123"
            assert result1["content"][0]["text"] == "Hello!"

            # Allow the async cache write task to complete
            await asyncio.sleep(0.1)

            # Second call - identical params, should hit cache
            result2 = await litellm.anthropic_messages(
                model="anthropic/claude-sonnet-4-20250514",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=100,
                api_key="fake-key",
            )

            # Provider should NOT have been called again
            assert mock_handler.call_count == 1
            # Result should match the cached response
            assert result2["id"] == "msg_test_123"
            assert result2["content"][0]["text"] == "Hello!"

    @pytest.mark.asyncio
    async def test_cache_bypass_with_no_cache(self):
        """
        After a cached response exists, passing cache={"no-cache": True}
        should force a fresh provider call.
        """
        mock_response = _mock_anthropic_response(
            text="First response", msg_id="msg_first"
        )
        mock_response_fresh = _mock_anthropic_response(
            text="Fresh response", msg_id="msg_fresh"
        )

        with patch(
            "litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.async_anthropic_messages_handler",
            new_callable=AsyncMock,
        ) as mock_handler:
            mock_handler.return_value = mock_response

            # First call - populates cache
            result1 = await litellm.anthropic_messages(
                model="anthropic/claude-sonnet-4-20250514",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=100,
                api_key="fake-key",
            )
            assert mock_handler.call_count == 1
            assert result1["content"][0]["text"] == "First response"

            # Second call with no-cache - should bypass cache
            mock_handler.return_value = mock_response_fresh
            result2 = await litellm.anthropic_messages(
                model="anthropic/claude-sonnet-4-20250514",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=100,
                api_key="fake-key",
                cache={"no-cache": True},
            )

            # Provider should have been called again
            assert mock_handler.call_count == 2
            assert result2["content"][0]["text"] == "Fresh response"

    @pytest.mark.asyncio
    async def test_non_streaming_cache_hit_returns_dict(self):
        """
        Verify that a cache hit for non-streaming returns the dict directly
        through _convert_cached_result_to_model_response.
        """
        mock_response = _mock_anthropic_response(
            text="Cached dict response", msg_id="msg_dict_cache"
        )

        with patch(
            "litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.async_anthropic_messages_handler",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_handler:
            # First call - cache miss
            await litellm.anthropic_messages(
                model="anthropic/claude-sonnet-4-20250514",
                messages=[{"role": "user", "content": "Cache dict test"}],
                max_tokens=50,
                api_key="fake-key",
            )
            assert mock_handler.call_count == 1

            # Allow cache write
            await asyncio.sleep(0.1)

            # Second call - cache hit, exercises _convert_cached_result_to_model_response
            result = await litellm.anthropic_messages(
                model="anthropic/claude-sonnet-4-20250514",
                messages=[{"role": "user", "content": "Cache dict test"}],
                max_tokens=50,
                api_key="fake-key",
            )

            # Should NOT call provider again
            assert mock_handler.call_count == 1
            # Should return the cached dict
            assert result["id"] == "msg_dict_cache"
            assert result["content"][0]["text"] == "Cached dict response"

    @pytest.mark.asyncio
    async def test_different_system_prompts_no_false_hit(self):
        """
        Two calls with different system prompts should both call the provider.
        The cache should not return a false hit for different system prompts.
        """
        mock_response_assistant = _mock_anthropic_response(
            text="I am an assistant", msg_id="msg_assistant"
        )
        mock_response_pirate = _mock_anthropic_response(
            text="Arrr matey!", msg_id="msg_pirate"
        )

        with patch(
            "litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.async_anthropic_messages_handler",
            new_callable=AsyncMock,
        ) as mock_handler:
            mock_handler.return_value = mock_response_assistant

            # First call with system="You are a helpful assistant."
            result1 = await litellm.anthropic_messages(
                model="anthropic/claude-sonnet-4-20250514",
                messages=[{"role": "user", "content": "Who are you?"}],
                max_tokens=100,
                system="You are a helpful assistant.",
                api_key="fake-key",
            )
            assert mock_handler.call_count == 1
            assert result1["content"][0]["text"] == "I am an assistant"

            # Second call with different system prompt
            mock_handler.return_value = mock_response_pirate
            result2 = await litellm.anthropic_messages(
                model="anthropic/claude-sonnet-4-20250514",
                messages=[{"role": "user", "content": "Who are you?"}],
                max_tokens=100,
                system="You are a pirate.",
                api_key="fake-key",
            )

            # Provider should have been called again (different cache key)
            assert mock_handler.call_count == 2
            assert result2["content"][0]["text"] == "Arrr matey!"
