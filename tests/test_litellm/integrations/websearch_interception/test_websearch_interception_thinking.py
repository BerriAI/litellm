"""
Unit tests for WebSearch Interception with Extended Thinking

Tests that the websearch interception correctly handles thinking/redacted_thinking
blocks, both at the transformation layer and the async_execute_tool_calls layer.
"""

import asyncio
from unittest.mock import Mock, AsyncMock, patch

import pytest

from litellm.integrations.websearch_interception.handler import (
    WebSearchInterceptionLogger,
)
from litellm.integrations.websearch_interception.transformation import (
    WebSearchTransformation,
)


class TestTransformResponseWithThinking:
    """Tests for _transform_response_anthropic with thinking blocks."""

    def test_thinking_blocks_prepended_to_assistant_message(self):
        """Test that thinking blocks are prepended before tool_use blocks."""
        tool_calls = [
            {
                "id": "toolu_01",
                "type": "tool_use",
                "name": "litellm_web_search",
                "input": {"query": "latest news"},
            }
        ]
        search_results = [
            "Title: News\nURL: https://example.com\nSnippet: Latest news"
        ]
        thinking_blocks = [
            {
                "type": "thinking",
                "thinking": "Let me search for that.",
                "signature": "sig123",
            },
            {"type": "redacted_thinking", "data": "abc123"},
        ]

        assistant_msg, user_msg = (
            WebSearchTransformation._transform_response_anthropic(
                tool_calls=tool_calls,
                search_results=search_results,
                thinking_blocks=thinking_blocks,
            )
        )

        # Verify thinking blocks come first
        content = assistant_msg["content"]
        assert len(content) == 3  # 2 thinking + 1 tool_use
        assert content[0]["type"] == "thinking"
        assert content[0]["thinking"] == "Let me search for that."
        assert content[1]["type"] == "redacted_thinking"
        assert content[1]["data"] == "abc123"
        assert content[2]["type"] == "tool_use"
        assert content[2]["id"] == "toolu_01"

    def test_no_thinking_blocks_backward_compat(self):
        """Test that transform works without thinking blocks (backward compat)."""
        tool_calls = [
            {
                "id": "toolu_01",
                "type": "tool_use",
                "name": "litellm_web_search",
                "input": {"query": "test"},
            }
        ]
        search_results = ["Search result text"]

        assistant_msg, _ = (
            WebSearchTransformation._transform_response_anthropic(
                tool_calls=tool_calls,
                search_results=search_results,
                thinking_blocks=[],
            )
        )

        content = assistant_msg["content"]
        assert len(content) == 1
        assert content[0]["type"] == "tool_use"

    def test_empty_thinking_blocks_list(self):
        """Test that an empty thinking_blocks list behaves like None."""
        tool_calls = [
            {
                "id": "toolu_01",
                "type": "tool_use",
                "name": "litellm_web_search",
                "input": {"query": "test"},
            }
        ]
        search_results = ["Search result text"]

        assistant_msg, _ = (
            WebSearchTransformation._transform_response_anthropic(
                tool_calls=tool_calls,
                search_results=search_results,
                thinking_blocks=[],
            )
        )

        content = assistant_msg["content"]
        assert len(content) == 1
        assert content[0]["type"] == "tool_use"

    def test_transform_response_passes_thinking_to_anthropic(self):
        """Test that transform_response routes thinking_blocks correctly."""
        tool_calls = [
            {
                "id": "toolu_01",
                "type": "tool_use",
                "name": "litellm_web_search",
                "input": {"query": "test"},
            }
        ]
        search_results = ["Search result"]
        thinking_blocks = [
            {
                "type": "thinking",
                "thinking": "Reasoning here.",
                "signature": "sig",
            },
        ]

        assistant_msg, _ = WebSearchTransformation.transform_response(
            tool_calls=tool_calls,
            search_results=search_results,
            response_format="anthropic",
            thinking_blocks=thinking_blocks,
        )

        content = assistant_msg["content"]
        assert content[0]["type"] == "thinking"
        assert content[1]["type"] == "tool_use"

    def test_transform_response_openai_ignores_thinking(self):
        """Test that OpenAI format is unaffected by thinking_blocks param."""
        tool_calls = [
            {
                "id": "call_01",
                "type": "function",
                "name": "litellm_web_search",
                "function": {
                    "name": "litellm_web_search",
                    "arguments": {"query": "test"},
                },
                "input": {"query": "test"},
            }
        ]
        search_results = ["Search result"]
        thinking_blocks = [
            {
                "type": "thinking",
                "thinking": "Should not appear.",
                "signature": "sig",
            },
        ]

        assistant_msg, _ = WebSearchTransformation.transform_response(
            tool_calls=tool_calls,
            search_results=search_results,
            response_format="openai",
            thinking_blocks=thinking_blocks,
        )

        # OpenAI format uses tool_calls key, not content — thinking is irrelevant
        assert "tool_calls" in assistant_msg
        assert "content" not in assistant_msg


class TestAsyncExecuteToolCallsWithThinking:
    """Tests for async_execute_tool_calls with thinking blocks in response."""

    @pytest.mark.asyncio
    async def test_executes_tool_calls_with_thinking_in_response(self):
        """Test that async_execute_tool_calls works when response has thinking + tool_use blocks."""
        logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])

        response = {
            "content": [
                {
                    "type": "thinking",
                    "thinking": "Let me think...",
                    "signature": "sig1",
                },
                {"type": "redacted_thinking", "data": "redacted_data"},
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "litellm_web_search",
                    "input": {"query": "latest news"},
                },
            ]
        }

        with patch.object(logger, "_execute_search", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = "Title: News\nURL: https://example.com\nSnippet: Latest"

            results = await logger.async_execute_tool_calls(
                response=response,
                kwargs={"custom_llm_provider": "bedrock"},
            )

        assert len(results) == 1
        assert results[0].tool_call_id == "toolu_01"
        assert results[0].is_error is False
        assert "News" in results[0].content

    @pytest.mark.asyncio
    async def test_no_results_when_no_tool_calls(self):
        """Test that thinking-only responses don't trigger tool execution."""
        logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])

        response = {
            "content": [
                {
                    "type": "thinking",
                    "thinking": "Just thinking...",
                    "signature": "sig",
                },
                {"type": "text", "text": "Here is my response."},
            ]
        }

        results = await logger.async_execute_tool_calls(
            response=response,
            kwargs={"custom_llm_provider": "bedrock"},
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_no_results_when_thinking_disabled(self):
        """Test that tool_use without thinking blocks still works."""
        logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])

        response = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "litellm_web_search",
                    "input": {"query": "test"},
                },
            ]
        }

        with patch.object(logger, "_execute_search", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = "Search result text"

            results = await logger.async_execute_tool_calls(
                response=response,
                kwargs={"custom_llm_provider": "bedrock"},
            )

        assert len(results) == 1
        assert results[0].tool_call_id == "toolu_01"
        assert results[0].is_error is False

    @pytest.mark.asyncio
    async def test_skips_wrong_provider(self):
        """Test that async_execute_tool_calls returns empty for wrong provider."""
        logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])

        response = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "litellm_web_search",
                    "input": {"query": "test"},
                },
            ]
        }

        results = await logger.async_execute_tool_calls(
            response=response,
            kwargs={"custom_llm_provider": "openai"},
        )

        assert results == []
