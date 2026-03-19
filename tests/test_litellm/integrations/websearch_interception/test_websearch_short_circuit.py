"""
Unit tests for WebSearch Short-Circuit

Tests the short-circuit path that detects web-search-only /v1/messages requests
and executes the search directly without routing through the backend LLM.
"""

from unittest.mock import AsyncMock, patch

import pytest

from litellm.integrations.websearch_interception.handler import (
    WebSearchInterceptionLogger,
)

# ---------------------------------------------------------------------------
# Detection tests
# ---------------------------------------------------------------------------


class TestTryShortCircuitSearch:
    """Tests for WebSearchInterceptionLogger.try_short_circuit_search"""

    @pytest.mark.asyncio
    async def test_short_circuits_single_web_search_tool(self):
        """Single web_search_20250305 tool → short-circuit fires"""
        logger = WebSearchInterceptionLogger(enabled_providers=["github_copilot"])

        with patch.object(
            logger, "_execute_search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = (
                "Title: Result\nURL: https://example.com\nSnippet: test"
            )

            result = await logger.try_short_circuit_search(
                model="github_copilot/claude-sonnet-4",
                messages=[
                    {"role": "user", "content": "Search for Claude Code releases"}
                ],
                tools=[
                    {"type": "web_search_20250305", "name": "web_search", "max_uses": 8}
                ],
                custom_llm_provider="github_copilot",
            )

        assert result is not None
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["stop_reason"] == "end_turn"
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"
        assert "Result" in result["content"][0]["text"]
        mock_search.assert_called_once_with("Search for Claude Code releases")

    @pytest.mark.asyncio
    async def test_does_not_short_circuit_mixed_tools(self):
        """Mix of web_search and other tools → NOT short-circuited"""
        logger = WebSearchInterceptionLogger(enabled_providers=["github_copilot"])

        result = await logger.try_short_circuit_search(
            model="github_copilot/claude-sonnet-4",
            messages=[{"role": "user", "content": "Do something"}],
            tools=[
                {"type": "web_search_20250305", "name": "web_search", "max_uses": 8},
                {"name": "Read", "description": "Read a file", "input_schema": {}},
            ],
            custom_llm_provider="github_copilot",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_does_not_short_circuit_no_tools(self):
        """No tools → NOT short-circuited"""
        logger = WebSearchInterceptionLogger(enabled_providers=["github_copilot"])

        result = await logger.try_short_circuit_search(
            model="github_copilot/claude-sonnet-4",
            messages=[{"role": "user", "content": "Hello"}],
            tools=None,
            custom_llm_provider="github_copilot",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_does_not_short_circuit_empty_tools(self):
        """Empty tools list → NOT short-circuited"""
        logger = WebSearchInterceptionLogger(enabled_providers=["github_copilot"])

        result = await logger.try_short_circuit_search(
            model="github_copilot/claude-sonnet-4",
            messages=[{"role": "user", "content": "Hello"}],
            tools=[],
            custom_llm_provider="github_copilot",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_does_not_short_circuit_wrong_provider(self):
        """Provider not in enabled_providers → NOT short-circuited"""
        logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])

        result = await logger.try_short_circuit_search(
            model="github_copilot/claude-sonnet-4",
            messages=[{"role": "user", "content": "Search for something"}],
            tools=[
                {"type": "web_search_20250305", "name": "web_search", "max_uses": 8}
            ],
            custom_llm_provider="github_copilot",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_does_not_short_circuit_no_messages(self):
        """Empty messages → NOT short-circuited"""
        logger = WebSearchInterceptionLogger(enabled_providers=["github_copilot"])

        result = await logger.try_short_circuit_search(
            model="github_copilot/claude-sonnet-4",
            messages=[],
            tools=[
                {"type": "web_search_20250305", "name": "web_search", "max_uses": 8}
            ],
            custom_llm_provider="github_copilot",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_search_failure_returns_error_text(self):
        """Search failure → response with error message, not exception"""
        logger = WebSearchInterceptionLogger(enabled_providers=["github_copilot"])

        with patch.object(
            logger, "_execute_search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.side_effect = RuntimeError("Tavily API error")

            result = await logger.try_short_circuit_search(
                model="github_copilot/claude-sonnet-4",
                messages=[{"role": "user", "content": "Search for something"}],
                tools=[
                    {"type": "web_search_20250305", "name": "web_search", "max_uses": 8}
                ],
                custom_llm_provider="github_copilot",
            )

        assert result is not None
        assert "Search failed" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_response_has_valid_structure(self):
        """Synthetic response has all required AnthropicMessagesResponse fields"""
        logger = WebSearchInterceptionLogger(enabled_providers=["github_copilot"])

        with patch.object(
            logger, "_execute_search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = "search results here"

            result = await logger.try_short_circuit_search(
                model="github_copilot/claude-sonnet-4",
                messages=[{"role": "user", "content": "Search query"}],
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                custom_llm_provider="github_copilot",
            )

        assert result is not None
        # Required fields
        assert "id" in result
        assert result["id"].startswith("msg_")
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["model"] == "github_copilot/claude-sonnet-4"
        assert result["stop_reason"] == "end_turn"
        assert result["stop_sequence"] is None
        assert "usage" in result
        assert "content" in result


# ---------------------------------------------------------------------------
# Query extraction tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Integration with entry point
# ---------------------------------------------------------------------------


class TestShortCircuitEntryPoint:
    """Tests for _try_websearch_short_circuit in the /v1/messages handler"""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_callbacks(self):
        """No callbacks configured → returns None"""
        from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
            _try_websearch_short_circuit,
        )

        with patch("litellm.callbacks", []):
            result = await _try_websearch_short_circuit(
                model="test",
                messages=[],
                tools=[],
                custom_llm_provider="github_copilot",
                stream=False,
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_dict_when_not_streaming(self):
        """Non-streaming short-circuit → returns dict"""
        from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
            _try_websearch_short_circuit,
        )

        logger = WebSearchInterceptionLogger(enabled_providers=["github_copilot"])
        with patch.object(
            logger, "_execute_search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = "results"
            with patch("litellm.callbacks", [logger]):
                result = await _try_websearch_short_circuit(
                    model="github_copilot/claude-sonnet-4",
                    messages=[{"role": "user", "content": "search query"}],
                    tools=[{"type": "web_search_20250305", "name": "web_search"}],
                    custom_llm_provider="github_copilot",
                    stream=False,
                )

        assert isinstance(result, dict)
        assert result["content"][0]["text"] == "results"

    @pytest.mark.asyncio
    async def test_returns_stream_iterator_when_streaming(self):
        """Streaming short-circuit → returns FakeAnthropicMessagesStreamIterator"""
        from litellm.llms.anthropic.experimental_pass_through.messages.fake_stream_iterator import (
            FakeAnthropicMessagesStreamIterator,
        )
        from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
            _try_websearch_short_circuit,
        )

        logger = WebSearchInterceptionLogger(enabled_providers=["github_copilot"])
        with patch.object(
            logger, "_execute_search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = "streaming results"
            with patch("litellm.callbacks", [logger]):
                result = await _try_websearch_short_circuit(
                    model="github_copilot/claude-sonnet-4",
                    messages=[{"role": "user", "content": "search query"}],
                    tools=[{"type": "web_search_20250305", "name": "web_search"}],
                    custom_llm_provider="github_copilot",
                    stream=True,
                )

        assert isinstance(result, FakeAnthropicMessagesStreamIterator)

        # Verify stream produces valid SSE events
        chunks = []
        async for chunk in result:
            chunks.append(chunk)

        assert len(chunks) > 0
        # First chunk should be message_start
        assert b"event: message_start" in chunks[0]
        # Last chunk should be message_stop
        assert b"event: message_stop" in chunks[-1]
        # Should contain the search results text
        all_data = b"".join(chunks)
        assert b"streaming results" in all_data

    @pytest.mark.asyncio
    async def test_skips_non_websearch_callbacks(self):
        """Non-WebSearchInterceptionLogger callbacks are ignored"""
        from unittest.mock import MagicMock

        from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
            _try_websearch_short_circuit,
        )

        other_callback = MagicMock()
        with patch("litellm.callbacks", [other_callback]):
            result = await _try_websearch_short_circuit(
                model="test",
                messages=[{"role": "user", "content": "search"}],
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                custom_llm_provider="github_copilot",
                stream=False,
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_uses_original_stream_not_hook_converted(self):
        """Verify that the entry point passes original_stream to the short-circuit.

        The pre-request hook converts stream=True → stream=False for the agentic
        loop. The short-circuit must use the ORIGINAL stream value so streaming
        callers get SSE events instead of a plain dict.
        """
        from litellm.llms.anthropic.experimental_pass_through.messages.fake_stream_iterator import (
            FakeAnthropicMessagesStreamIterator,
        )
        from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
            _try_websearch_short_circuit,
        )

        logger = WebSearchInterceptionLogger(enabled_providers=["github_copilot"])
        with patch.object(
            logger, "_execute_search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = "streaming results"
            with patch("litellm.callbacks", [logger]):
                # Simulate what anthropic_messages() does: original_stream=True
                # is passed to the short-circuit, even though the hook would have
                # already converted stream to False in request_kwargs.
                result = await _try_websearch_short_circuit(
                    model="github_copilot/claude-sonnet-4",
                    messages=[{"role": "user", "content": "search query"}],
                    tools=[{"type": "web_search_20250305", "name": "web_search"}],
                    custom_llm_provider="github_copilot",
                    stream=True,  # original_stream, NOT the hook-converted value
                )

        # Must return a stream iterator, not a plain dict
        assert isinstance(result, FakeAnthropicMessagesStreamIterator)

    @pytest.mark.asyncio
    async def test_short_circuits_with_provider_from_model_string(self):
        """Provider embedded in model string (custom_llm_provider=None) should
        still fire the short-circuit when the caller propagates the derived
        provider.
        """
        from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
            _try_websearch_short_circuit,
        )

        logger = WebSearchInterceptionLogger(enabled_providers=["github_copilot"])
        with patch.object(
            logger, "_execute_search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = "results"
            with patch("litellm.callbacks", [logger]):
                # Simulate the caller having derived custom_llm_provider from
                # the model string before calling _try_websearch_short_circuit
                result = await _try_websearch_short_circuit(
                    model="github_copilot/claude-sonnet-4",
                    messages=[{"role": "user", "content": "search query"}],
                    tools=[{"type": "web_search_20250305", "name": "web_search"}],
                    custom_llm_provider="github_copilot",
                    stream=False,
                )

        assert result is not None
        assert result["content"][0]["text"] == "results"
