"""
Base test class for Anthropic Messages API tool search E2E tests.

Tests that tool search works correctly via litellm.anthropic.messages interface
by making actual API calls and validating that tool search discovers deferred tools.

Reference: https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List


import pytest
import litellm


# Sample tools for tool search testing
def get_deferred_tools() -> List[Dict[str, Any]]:
    """
    Returns a list of tools with defer_loading: true.
    These tools should only be discovered via tool search.
    """
    return [
        {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"],
            },
            "defer_loading": True,
        },
        {
            "name": "get_stock_price",
            "description": "Get the current stock price for a ticker symbol",
            "input_schema": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol, e.g. AAPL",
                    }
                },
                "required": ["ticker"],
            },
            "defer_loading": True,
        },
        {
            "name": "search_web",
            "description": "Search the web for information",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"],
            },
            "defer_loading": True,
        },
    ]


def get_tool_search_tool_regex() -> Dict[str, Any]:
    """Returns the tool search tool using regex variant."""
    return {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}


def get_tool_search_tool_bm25() -> Dict[str, Any]:
    """Returns the tool search tool using BM25 variant."""
    return {"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"}


class BaseAnthropicMessagesToolSearchTest(ABC):
    """
    Base test class for tool search E2E tests across different providers.

    Subclasses must implement:
    - get_model(): Returns the model string to use for tests

    Tests pass the anthropic-beta header via extra_headers to validate
    that the header is correctly forwarded to downstream providers.
    """

    @abstractmethod
    def get_model(self) -> str:
        """
        Returns the model string to use for tests.

        Examples:
        - "anthropic/claude-sonnet-4-5-20250929"
        - "vertex_ai/claude-sonnet-4-5@20250929"
        - "bedrock/invoke/anthropic.claude-sonnet-4-5-20250929-v1:0"
        """
        pass

    def get_extra_headers(self) -> Dict[str, str]:
        """
        Returns extra headers to pass with the request.
        Includes the anthropic-beta header for tool search.

        This is what claude code forwards, simulate the same behavior here.
        """
        return {"anthropic-beta": "advanced-tool-use-2025-11-20"}

    def get_tools_with_tool_search(self) -> List[Dict[str, Any]]:
        """
        Returns tools list with tool search tool and deferred tools.
        """
        return [get_tool_search_tool_regex()] + get_deferred_tools()

    @pytest.mark.asyncio
    async def test_tool_search_basic_request(self):
        """
        E2E test: Basic tool search request should succeed.

        This validates that the tool search beta header is being passed via
        extra_headers and forwarded correctly to the downstream provider.
        """
        litellm.turn_on_debug()

        tools = self.get_tools_with_tool_search()
        messages = [{"role": "user", "content": "What's the weather in San Francisco?"}]

        response = await litellm.anthropic.messages.acreate(
            model=self.get_model(),
            messages=messages,
            tools=tools,
            max_tokens=1024,
            extra_headers=self.get_extra_headers(),
        )

        print(f"Response: {json.dumps(response, indent=2, default=str)}")

        # Validate response structure
        assert "content" in response, "Response should contain content"
        assert "usage" in response, "Response should contain usage"

        # The model should either respond with text or use a tool
        content = response.get("content", [])
        assert len(content) > 0, "Response should have content"

    @pytest.mark.asyncio
    @pytest.mark.flaky(retries=3, delay=5)
    async def test_tool_search_streaming(self):
        """
        E2E test: Tool search should work with streaming responses.
        """
        litellm.turn_on_debug()

        tools = self.get_tools_with_tool_search()
        messages = [{"role": "user", "content": "What's the weather like in Tokyo?"}]

        response = await litellm.anthropic.messages.acreate(
            model=self.get_model(),
            messages=messages,
            tools=tools,
            max_tokens=1024,
            stream=True,
            extra_headers=self.get_extra_headers(),
        )

        # Collect all chunks
        chunks = []
        async for chunk in response:
            if isinstance(chunk, bytes):
                chunk_str = chunk.decode("utf-8")
                for line in chunk_str.split("\n"):
                    if line.startswith("data: "):
                        try:
                            json_data = json.loads(line[6:])
                            chunks.append(json_data)
                            print(
                                f"Chunk: {json.dumps(json_data, indent=2, default=str)}"
                            )
                        except json.JSONDecodeError:
                            pass
            elif isinstance(chunk, dict):
                chunks.append(chunk)
                print(f"Chunk: {json.dumps(chunk, indent=2, default=str)}")

        # Should have received chunks
        assert len(chunks) > 0, "Expected to receive streaming chunks"

        # Should have message_start
        message_starts = [c for c in chunks if c.get("type") == "message_start"]
        assert len(message_starts) > 0, "Expected message_start in streaming response"
