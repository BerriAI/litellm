"""
Unit tests for WebSearch Short-Circuit

Tests the short-circuit path that detects web-search-only /v1/messages requests
and executes the search directly without routing through the backend LLM.

The response uses native Anthropic format (server_tool_use + web_search_tool_result)
so Claude Code's WebSearchTool parser works correctly.
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
        # Native format: server_tool_use + web_search_tool_result + text
        assert result["content"][0]["type"] == "server_tool_use"
        assert result["content"][0]["name"] == "web_search"
        assert result["content"][1]["type"] == "web_search_tool_result"
        assert result["content"][2]["type"] == "text"
        assert "Result" in result["content"][2]["text"]
        mock_search.assert_called_once_with("Search for Claude Code releases")

    @pytest.mark.asyncio
    async def test_native_format_search_hits(self):
        """Search results are structured as web_search_result hits"""
        logger = WebSearchInterceptionLogger(enabled_providers=["github_copilot"])

        with patch.object(
            logger, "_execute_search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = (
                "Title: First Result\nURL: https://example.com/1\nSnippet: first\n\n"
                "Title: Second Result\nURL: https://example.com/2\nSnippet: second"
            )

            result = await logger.try_short_circuit_search(
                model="github_copilot/claude-sonnet-4",
                messages=[{"role": "user", "content": "Search query"}],
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                custom_llm_provider="github_copilot",
            )

        assert result is not None
        hits = result["content"][1]["content"]
        assert len(hits) == 2
        assert hits[0]["type"] == "web_search_result"
        assert hits[0]["url"] == "https://example.com/1"
        assert hits[0]["title"] == "First Result"
        assert hits[1]["url"] == "https://example.com/2"

    @pytest.mark.asyncio
    async def test_server_tool_use_has_query(self):
        """server_tool_use block contains the original search query"""
        logger = WebSearchInterceptionLogger(enabled_providers=["github_copilot"])

        with patch.object(
            logger, "_execute_search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = "Title: R\nURL: https://x.com\nSnippet: s"

            result = await logger.try_short_circuit_search(
                model="github_copilot/claude-sonnet-4",
                messages=[{"role": "user", "content": "trending AI topics"}],
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                custom_llm_provider="github_copilot",
            )

        stu = result["content"][0]
        assert stu["type"] == "server_tool_use"
        assert stu["name"] == "web_search"
        assert stu["input"]["query"] == "trending AI topics"
        assert stu["id"].startswith("srvtoolu_")

    @pytest.mark.asyncio
    async def test_tool_use_id_links_blocks(self):
        """server_tool_use.id matches web_search_tool_result.tool_use_id"""
        logger = WebSearchInterceptionLogger(enabled_providers=["github_copilot"])

        with patch.object(
            logger, "_execute_search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = "Title: R\nURL: https://x.com\nSnippet: s"

            result = await logger.try_short_circuit_search(
                model="github_copilot/claude-sonnet-4",
                messages=[{"role": "user", "content": "query"}],
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                custom_llm_provider="github_copilot",
            )

        assert result["content"][0]["id"] == result["content"][1]["tool_use_id"]

    @pytest.mark.asyncio
    async def test_usage_includes_web_search_requests(self):
        """Usage includes server_tool_use.web_search_requests count"""
        logger = WebSearchInterceptionLogger(enabled_providers=["github_copilot"])

        with patch.object(
            logger, "_execute_search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = "Title: R\nURL: https://x.com\nSnippet: s"

            result = await logger.try_short_circuit_search(
                model="github_copilot/claude-sonnet-4",
                messages=[{"role": "user", "content": "query"}],
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                custom_llm_provider="github_copilot",
            )

        assert result["usage"]["server_tool_use"]["web_search_requests"] == 1

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
    async def test_does_not_short_circuit_native_providers(self):
        """Providers with native Anthropic Messages support (anthropic, bedrock,
        vertex_ai) are skipped — their API handles web search natively."""
        for provider in ["anthropic", "bedrock"]:
            logger = WebSearchInterceptionLogger(
                enabled_providers=[provider, "github_copilot"]
            )

            result = await logger.try_short_circuit_search(
                model=f"{provider}/claude-sonnet-4",
                messages=[{"role": "user", "content": "search query"}],
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                custom_llm_provider=provider,
            )

            assert result is None, f"Short-circuit should NOT fire for native provider {provider}"

    @pytest.mark.asyncio
    async def test_short_circuits_non_native_providers(self):
        """Non-native providers (github_copilot, etc.) get short-circuited."""
        logger = WebSearchInterceptionLogger(enabled_providers=["github_copilot"])

        with patch.object(
            logger, "_execute_search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = "Title: R\nURL: https://x.com\nSnippet: s"

            result = await logger.try_short_circuit_search(
                model="github_copilot/claude-sonnet-4",
                messages=[{"role": "user", "content": "search query"}],
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                custom_llm_provider="github_copilot",
            )

        assert result is not None
        assert result["content"][0]["type"] == "server_tool_use"

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
        # Error text is in the last content block (text)
        text_block = result["content"][-1]
        assert text_block["type"] == "text"
        assert "Search failed" in text_block["text"]

    @pytest.mark.asyncio
    async def test_response_has_valid_structure(self):
        """Synthetic response has all required AnthropicMessagesResponse fields"""
        logger = WebSearchInterceptionLogger(enabled_providers=["github_copilot"])

        with patch.object(
            logger, "_execute_search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = "Title: R\nURL: https://x.com\nSnippet: test"

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
        # Content has 3 blocks: server_tool_use, web_search_tool_result, text
        assert len(result["content"]) == 3


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
        """Non-streaming short-circuit → returns dict with native format"""
        from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
            _try_websearch_short_circuit,
        )

        logger = WebSearchInterceptionLogger(enabled_providers=["github_copilot"])
        with patch.object(
            logger, "_execute_search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = "Title: R\nURL: https://x.com\nSnippet: results"
            with patch("litellm.callbacks", [logger]):
                result = await _try_websearch_short_circuit(
                    model="github_copilot/claude-sonnet-4",
                    messages=[{"role": "user", "content": "search query"}],
                    tools=[{"type": "web_search_20250305", "name": "web_search"}],
                    custom_llm_provider="github_copilot",
                    stream=False,
                )

        assert isinstance(result, dict)
        assert result["content"][0]["type"] == "server_tool_use"
        assert result["content"][1]["type"] == "web_search_tool_result"

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
            mock_search.return_value = (
                "Title: Result\nURL: https://example.com\nSnippet: streaming results"
            )
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
        assert b"event: message_start" in chunks[0]
        assert b"event: message_stop" in chunks[-1]
        # Should contain server_tool_use and web_search_tool_result blocks
        all_data = b"".join(chunks)
        assert b"server_tool_use" in all_data
        assert b"web_search_tool_result" in all_data
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
        """Verify that the entry point passes original_stream to the short-circuit."""
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
            mock_search.return_value = "Title: R\nURL: https://x.com\nSnippet: s"
            with patch("litellm.callbacks", [logger]):
                result = await _try_websearch_short_circuit(
                    model="github_copilot/claude-sonnet-4",
                    messages=[{"role": "user", "content": "search query"}],
                    tools=[{"type": "web_search_20250305", "name": "web_search"}],
                    custom_llm_provider="github_copilot",
                    stream=True,
                )

        assert isinstance(result, FakeAnthropicMessagesStreamIterator)
