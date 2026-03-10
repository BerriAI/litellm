"""
Unit tests for WebSearch Interception with Extended Thinking

Tests that the websearch interception agentic loop correctly handles
thinking/redacted_thinking blocks when extended thinking is enabled.
"""

from unittest.mock import Mock

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

        # No thinking_blocks param (default None)
        assistant_msg, _ = (
            WebSearchTransformation._transform_response_anthropic(
                tool_calls=tool_calls,
                search_results=search_results,
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


class TestAgenticLoopThinkingExtraction:
    """Tests for thinking block extraction in async_should_run_agentic_loop."""

    @pytest.mark.asyncio
    async def test_extracts_thinking_blocks_from_dict_response(self):
        """Test extraction of thinking blocks from dict-style response."""
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

        should_run, tools_dict = await logger.async_should_run_agentic_loop(
            response=response,
            model="bedrock/claude",
            messages=[],
            tools=[{"name": "WebSearch"}],
            stream=False,
            custom_llm_provider="bedrock",
            kwargs={},
        )

        assert should_run is True
        assert len(tools_dict["tool_calls"]) == 1
        assert len(tools_dict["thinking_blocks"]) == 2
        assert tools_dict["thinking_blocks"][0]["type"] == "thinking"
        assert tools_dict["thinking_blocks"][0]["thinking"] == "Let me think..."
        assert tools_dict["thinking_blocks"][1]["type"] == "redacted_thinking"
        assert tools_dict["thinking_blocks"][1]["data"] == "redacted_data"

    @pytest.mark.asyncio
    async def test_extracts_thinking_blocks_from_object_response(self):
        """Test extraction of thinking blocks from non-dict response objects.

        In practice, the Anthropic pass-through always returns plain dicts
        (TypedDict(**raw_json) produces a dict). This test covers the safety
        branch for non-dict response objects.
        """
        logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])

        # Simulate object-style response blocks
        thinking_block = Mock()
        thinking_block.type = "thinking"
        thinking_block.thinking = "Reasoning..."
        thinking_block.signature = "sig"

        redacted_block = Mock()
        redacted_block.type = "redacted_thinking"
        redacted_block.data = "abc"

        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "litellm_web_search"
        tool_block.id = "toolu_01"
        tool_block.input = {"query": "test"}

        response = Mock()
        response.content = [thinking_block, redacted_block, tool_block]

        should_run, tools_dict = await logger.async_should_run_agentic_loop(
            response=response,
            model="bedrock/claude",
            messages=[],
            tools=[{"name": "WebSearch"}],
            stream=False,
            custom_llm_provider="bedrock",
            kwargs={},
        )

        assert should_run is True
        assert len(tools_dict["thinking_blocks"]) == 2
        # Verify getattr-based conversion produced correct dicts
        assert tools_dict["thinking_blocks"][0] == {
            "type": "thinking",
            "thinking": "Reasoning...",
            "signature": "sig",
        }
        assert tools_dict["thinking_blocks"][1] == {
            "type": "redacted_thinking",
            "data": "abc",
        }

    @pytest.mark.asyncio
    async def test_no_thinking_blocks_when_thinking_disabled(self):
        """Test that thinking_blocks is empty when response has no thinking."""
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

        should_run, tools_dict = await logger.async_should_run_agentic_loop(
            response=response,
            model="bedrock/claude",
            messages=[],
            tools=[{"name": "WebSearch"}],
            stream=False,
            custom_llm_provider="bedrock",
            kwargs={},
        )

        assert should_run is True
        assert tools_dict["thinking_blocks"] == []

    @pytest.mark.asyncio
    async def test_thinking_blocks_not_extracted_when_no_tool_calls(self):
        """Test that no extraction happens when no websearch tool calls found."""
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

        should_run, tools_dict = await logger.async_should_run_agentic_loop(
            response=response,
            model="bedrock/claude",
            messages=[],
            tools=[{"name": "WebSearch"}],
            stream=False,
            custom_llm_provider="bedrock",
            kwargs={},
        )

        assert should_run is False
        assert tools_dict == {}
