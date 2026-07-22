"""
Unit tests for WebSearch Short-Circuit

Tests the short-circuit path that detects web-search-only /v1/messages requests
and executes the search directly without routing through the backend LLM.
"""

import json
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
                "Title: Result\nURL: https://example.com\nSnippet: test",
                None,
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
        # Native web_search_20250305 client → short-circuit emits native
        # blocks (server_tool_use + web_search_tool_result) plus the legacy
        # text block so Cowork / Claude Desktop citations panels populate.
        block_types = [b["type"] for b in result["content"]]
        assert "server_tool_use" in block_types
        assert "web_search_tool_result" in block_types
        assert "text" in block_types
        text_block = next(b for b in result["content"] if b["type"] == "text")
        assert "Result" in text_block["text"]
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
    async def test_does_not_short_circuit_bedrock(self):
        """Bedrock has native agentic loop support → NOT short-circuited.

        Providers with a BaseAnthropicMessagesConfig (bedrock, vertex_ai, etc.)
        use the agentic loop which includes a follow-up LLM synthesis step.
        The short-circuit must not fire for them.
        """
        logger = WebSearchInterceptionLogger(
            enabled_providers=["bedrock", "github_copilot"]
        )

        result = await logger.try_short_circuit_search(
            model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            messages=[{"role": "user", "content": "Search for something"}],
            tools=[
                {"type": "web_search_20250305", "name": "web_search", "max_uses": 8}
            ],
            custom_llm_provider="bedrock",
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
        text_block = next(b for b in result["content"] if b["type"] == "text")
        assert "Search failed" in text_block["text"]

    @pytest.mark.asyncio
    async def test_response_has_valid_structure(self):
        """Synthetic response has all required AnthropicMessagesResponse fields"""
        logger = WebSearchInterceptionLogger(enabled_providers=["github_copilot"])

        with patch.object(
            logger, "_execute_search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = ("search results here", None)

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

    @pytest.mark.asyncio
    async def test_emits_native_blocks_for_post_conversion_tool(self):
        """Regression: native blocks must be emitted even when the tool has
        already been renamed to ``litellm_web_search`` by the pre-request hook.

        In production, ``async_pre_request_hook`` converts the client's native
        ``web_search_20250305`` tool to the LiteLLM standard ``litellm_web_search``
        *before* ``try_short_circuit_search`` runs. The other tests in this class
        pass the pre-conversion Anthropic shape, which no longer reaches this
        function on the real request path — so they did not catch that the
        short-circuit emitted a text-only response for every real request,
        causing Claude Code to report "Did 0 searches" and discard the results.
        """
        from litellm.integrations.websearch_interception.tools import (
            get_litellm_web_search_tool,
        )

        logger = WebSearchInterceptionLogger(enabled_providers=["github_copilot"])

        with patch.object(
            logger, "_execute_search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = ("search results text", None)

            result = await logger.try_short_circuit_search(
                model="github_copilot/claude-sonnet-4",
                messages=[{"role": "user", "content": "Search for X"}],
                # The post-conversion shape the real pipeline delivers here.
                tools=[get_litellm_web_search_tool()],
                custom_llm_provider="github_copilot",
            )

        assert result is not None
        block_types = [b["type"] for b in result["content"]]
        assert "server_tool_use" in block_types
        assert "web_search_tool_result" in block_types
        assert "text" in block_types
        # server_tool_use carries the executed query so the client pairs it.
        stu = next(b for b in result["content"] if b["type"] == "server_tool_use")
        assert stu["input"]["query"] == "Search for X"
        # usage advertises the search so clients that read counts see 1.
        assert (
            result["usage"]["server_tool_use"]["web_search_requests"] == 1
        )
        mock_search.assert_called_once_with("Search for X")

    @pytest.mark.asyncio
    async def test_native_result_block_carries_sources(self):
        """The web_search_tool_result block must carry the structured sources
        (url/title) returned by the search provider, not just text."""
        from litellm.llms.base_llm.search.transformation import (
            SearchResponse,
            SearchResult,
        )

        logger = WebSearchInterceptionLogger(enabled_providers=["github_copilot"])

        structured = SearchResponse(
            results=[
                SearchResult(
                    title="Kubernetes Releases",
                    url="https://kubernetes.io/releases",
                    snippet="Latest stable release information.",
                )
            ]
        )

        with patch.object(
            logger, "_execute_search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = ("formatted text", structured)

            result = await logger.try_short_circuit_search(
                model="github_copilot/claude-sonnet-4",
                messages=[{"role": "user", "content": "k8s releases"}],
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                custom_llm_provider="github_copilot",
            )

        wstr = next(
            b for b in result["content"] if b["type"] == "web_search_tool_result"
        )
        assert len(wstr["content"]) == 1
        assert wstr["content"][0]["url"] == "https://kubernetes.io/releases"
        assert wstr["content"][0]["title"] == "Kubernetes Releases"
        # tool_use_id must pair with the server_tool_use block.
        stu = next(b for b in result["content"] if b["type"] == "server_tool_use")
        assert wstr["tool_use_id"] == stu["id"]


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
            mock_search.return_value = ("results", None)
            with patch("litellm.callbacks", [logger]):
                result = await _try_websearch_short_circuit(
                    model="github_copilot/claude-sonnet-4",
                    messages=[{"role": "user", "content": "search query"}],
                    tools=[{"type": "web_search_20250305", "name": "web_search"}],
                    custom_llm_provider="github_copilot",
                    stream=False,
                )

        assert isinstance(result, dict)
        text_block = next(b for b in result["content"] if b["type"] == "text")
        assert text_block["text"] == "results"

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
            mock_search.return_value = ("streaming results", None)
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
            mock_search.return_value = ("streaming results", None)
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
            mock_search.return_value = ("results", None)
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
        text_block = next(b for b in result["content"] if b["type"] == "text")
        assert text_block["text"] == "results"


# ---------------------------------------------------------------------------
# Streaming serialization of native blocks
# ---------------------------------------------------------------------------


class TestFakeStreamIteratorNativeBlocks:
    """The FakeAnthropicMessagesStreamIterator must serialize the native
    server_tool_use and web_search_tool_result blocks the short-circuit emits.

    Claude Code issues its WebSearch sub-request with stream=True, so the
    synthetic dict is rebuilt into SSE by this iterator. Before this fix the
    iterator only knew text/thinking/tool_use block types and dropped the two
    web-search block types (emitting a bare content_block_stop with no
    matching content_block_start), so the client never saw the search.
    """

    def _collect_sse(self, response: dict):
        from litellm.llms.anthropic.experimental_pass_through.messages.fake_stream_iterator import (
            FakeAnthropicMessagesStreamIterator,
        )

        it = FakeAnthropicMessagesStreamIterator(response)
        events = []
        for raw in it.chunks:
            text = raw.decode()
            for line in text.splitlines():
                if line.startswith("data:"):
                    events.append(json.loads(line[len("data:"):].strip()))
        return events

    def test_serializes_server_tool_use_and_result_blocks(self):
        response = {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "github_copilot/claude-sonnet-4",
            "content": [
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_abc",
                    "name": "web_search",
                    "input": {"query": "kubernetes latest"},
                },
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srvtoolu_abc",
                    "content": [
                        {
                            "type": "web_search_result",
                            "url": "https://kubernetes.io/releases",
                            "title": "Releases",
                            "page_age": None,
                            "encrypted_content": "",
                        }
                    ],
                },
                {"type": "text", "text": "Kubernetes 1.34 is latest."},
            ],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }

        events = self._collect_sse(response)
        starts = [e for e in events if e["type"] == "content_block_start"]
        stops = [e for e in events if e["type"] == "content_block_stop"]

        # Every emitted block gets a well-formed start/stop pair.
        assert len(starts) == 3
        assert len(stops) == 3
        started_types = [e["content_block"]["type"] for e in starts]
        assert started_types == [
            "server_tool_use",
            "web_search_tool_result",
            "text",
        ]

        # server_tool_use carries id/name and the query arrives via input_json_delta.
        stu_start = starts[0]["content_block"]
        assert stu_start["id"] == "srvtoolu_abc"
        assert stu_start["name"] == "web_search"
        deltas = [e for e in events if e["type"] == "content_block_delta"]
        json_deltas = [
            json.loads(e["delta"]["partial_json"])
            for e in deltas
            if e["delta"]["type"] == "input_json_delta"
        ]
        assert {"query": "kubernetes latest"} in json_deltas

        # web_search_tool_result carries the tool_use_id and sources.
        wstr_start = starts[1]["content_block"]
        assert wstr_start["tool_use_id"] == "srvtoolu_abc"
        assert wstr_start["content"][0]["url"] == "https://kubernetes.io/releases"
