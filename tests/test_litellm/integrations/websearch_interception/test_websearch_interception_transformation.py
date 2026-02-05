"""
Unit tests for WebSearch Interception Transformation

Tests the WebSearchTransformation class methods:
- transform_request: Extract WebSearch tool calls and thinking blocks from responses
- transform_response: Build follow-up messages with tool_use and tool_result blocks
- format_search_response: Format search results for tool_result content
"""

from unittest.mock import Mock


from litellm.constants import LITELLM_WEB_SEARCH_TOOL_NAME
from litellm.integrations.websearch_interception.transformation import (
    WebSearchTransformation,
)


class TestTransformRequest:
    """Tests for WebSearchTransformation.transform_request()"""

    def test_streaming_response_returns_empty(self):
        """Streaming responses should return empty result (we handle by converting to non-streaming)"""
        response = {"content": [{"type": "tool_use", "name": "WebSearch"}]}

        result = WebSearchTransformation.transform_request(
            response=response,
            stream=True,
        )

        assert result.has_websearch is False
        assert result.tool_calls == []
        assert result.thinking_blocks == []

    def test_dict_response_with_websearch_tool(self):
        """Dict response with WebSearch tool_use should be detected"""
        response = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_123",
                    "name": "WebSearch",
                    "input": {"query": "weather in SF"},
                }
            ]
        }

        result = WebSearchTransformation.transform_request(
            response=response,
            stream=False,
        )

        assert result.has_websearch is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["id"] == "tool_123"
        assert result.tool_calls[0]["name"] == "WebSearch"
        assert result.tool_calls[0]["input"]["query"] == "weather in SF"

    def test_object_response_with_websearch_tool(self):
        """Object response (with attributes) with WebSearch tool_use should be detected"""
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.id = "tool_456"
        tool_block.name = "WebSearch"
        tool_block.input = {"query": "latest news"}

        response = Mock()
        response.content = [tool_block]

        result = WebSearchTransformation.transform_request(
            response=response,
            stream=False,
        )

        assert result.has_websearch is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["id"] == "tool_456"
        assert result.tool_calls[0]["input"]["query"] == "latest news"

    def test_detects_litellm_web_search_tool_name(self):
        """Should detect the LiteLLM standard web search tool name"""
        response = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_789",
                    "name": LITELLM_WEB_SEARCH_TOOL_NAME,
                    "input": {"query": "test query"},
                }
            ]
        }

        result = WebSearchTransformation.transform_request(
            response=response,
            stream=False,
        )

        assert result.has_websearch is True
        assert result.tool_calls[0]["name"] == LITELLM_WEB_SEARCH_TOOL_NAME

    def test_detects_web_search_lowercase(self):
        """Should detect 'web_search' tool name (lowercase variant)"""
        response = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_abc",
                    "name": "web_search",
                    "input": {"query": "another query"},
                }
            ]
        }

        result = WebSearchTransformation.transform_request(
            response=response,
            stream=False,
        )

        assert result.has_websearch is True
        assert result.tool_calls[0]["name"] == "web_search"

    def test_captures_thinking_blocks(self):
        """Should capture thinking blocks from response"""
        response = {
            "content": [
                {
                    "type": "thinking",
                    "thinking": "Let me search for this information...",
                },
                {
                    "type": "tool_use",
                    "id": "tool_def",
                    "name": "WebSearch",
                    "input": {"query": "AI news"},
                },
            ]
        }

        result = WebSearchTransformation.transform_request(
            response=response,
            stream=False,
        )

        assert result.has_websearch is True
        assert len(result.thinking_blocks) == 1
        assert result.thinking_blocks[0]["type"] == "thinking"

    def test_captures_redacted_thinking_blocks(self):
        """Should capture redacted_thinking blocks from response"""
        response = {
            "content": [
                {
                    "type": "redacted_thinking",
                    "data": "base64redacteddata",
                },
                {
                    "type": "tool_use",
                    "id": "tool_ghi",
                    "name": "WebSearch",
                    "input": {"query": "sensitive query"},
                },
            ]
        }

        result = WebSearchTransformation.transform_request(
            response=response,
            stream=False,
        )

        assert result.has_websearch is True
        assert len(result.thinking_blocks) == 1
        assert result.thinking_blocks[0]["type"] == "redacted_thinking"

    def test_thinking_blocks_normalized_to_dict_from_sdk_objects(self):
        """SDK object thinking blocks should be normalized to dicts for JSON serialization"""
        # Create mock SDK objects (not dicts)
        thinking_block = Mock()
        thinking_block.type = "thinking"
        thinking_block.thinking = "Let me search for this..."
        thinking_block.signature = "sig123"

        redacted_block = Mock()
        redacted_block.type = "redacted_thinking"
        redacted_block.data = "base64redacteddata"

        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.id = "tool_xyz"
        tool_block.name = "WebSearch"
        tool_block.input = {"query": "test"}

        response = Mock()
        response.content = [thinking_block, redacted_block, tool_block]

        result = WebSearchTransformation.transform_request(
            response=response,
            stream=False,
        )

        assert result.has_websearch is True
        assert len(result.thinking_blocks) == 2

        # Verify blocks were normalized to dicts (not Mock objects)
        assert isinstance(result.thinking_blocks[0], dict)
        assert isinstance(result.thinking_blocks[1], dict)

        # Verify thinking block content including signature
        assert result.thinking_blocks[0]["type"] == "thinking"
        assert result.thinking_blocks[0]["thinking"] == "Let me search for this..."
        assert result.thinking_blocks[0]["signature"] == "sig123"

        # Verify redacted_thinking block content
        assert result.thinking_blocks[1]["type"] == "redacted_thinking"
        assert result.thinking_blocks[1]["data"] == "base64redacteddata"

    def test_multiple_tool_calls(self):
        """Should handle multiple WebSearch tool_use blocks"""
        response = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_1",
                    "name": "WebSearch",
                    "input": {"query": "query 1"},
                },
                {
                    "type": "tool_use",
                    "id": "tool_2",
                    "name": "WebSearch",
                    "input": {"query": "query 2"},
                },
            ]
        }

        result = WebSearchTransformation.transform_request(
            response=response,
            stream=False,
        )

        assert result.has_websearch is True
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["input"]["query"] == "query 1"
        assert result.tool_calls[1]["input"]["query"] == "query 2"

    def test_no_websearch_in_response(self):
        """Response without WebSearch tool should return has_websearch=False"""
        response = {
            "content": [
                {"type": "text", "text": "Here is a response"},
                {
                    "type": "tool_use",
                    "id": "tool_other",
                    "name": "calculator",
                    "input": {"expression": "2+2"},
                },
            ]
        }

        result = WebSearchTransformation.transform_request(
            response=response,
            stream=False,
        )

        assert result.has_websearch is False
        assert result.tool_calls == []

    def test_empty_content_returns_empty_result(self):
        """Empty content should return empty result"""
        response = {"content": []}

        result = WebSearchTransformation.transform_request(
            response=response,
            stream=False,
        )

        assert result.has_websearch is False
        assert result.tool_calls == []
        assert result.thinking_blocks == []

    def test_response_without_content_attribute(self):
        """Response object without content attribute should return empty result"""
        response = Mock(spec=[])  # Mock with no attributes

        result = WebSearchTransformation.transform_request(
            response=response,
            stream=False,
        )

        assert result.has_websearch is False


class TestTransformResponse:
    """Tests for WebSearchTransformation.transform_response()"""

    def test_builds_messages_without_thinking_blocks(self):
        """Should build correct messages without thinking blocks"""
        tool_calls = [
            {
                "id": "tool_1",
                "name": "WebSearch",
                "input": {"query": "test query"},
            }
        ]
        search_results = ["Title: Result\nURL: http://example.com\nSnippet: Test snippet"]
        thinking_blocks = []

        assistant_msg, user_msg = WebSearchTransformation.transform_response(
            tool_calls=tool_calls,
            search_results=search_results,
            thinking_blocks=thinking_blocks,
        )

        # Check assistant message
        assert assistant_msg["role"] == "assistant"
        assert len(assistant_msg["content"]) == 1
        assert assistant_msg["content"][0]["type"] == "tool_use"
        assert assistant_msg["content"][0]["id"] == "tool_1"
        assert assistant_msg["content"][0]["name"] == "WebSearch"

        # Check user message
        assert user_msg["role"] == "user"
        assert len(user_msg["content"]) == 1
        assert user_msg["content"][0]["type"] == "tool_result"
        assert user_msg["content"][0]["tool_use_id"] == "tool_1"
        assert "Test snippet" in user_msg["content"][0]["content"]

    def test_builds_messages_with_thinking_blocks(self):
        """Should include thinking blocks at start of assistant message"""
        tool_calls = [
            {
                "id": "tool_2",
                "name": "WebSearch",
                "input": {"query": "another query"},
            }
        ]
        search_results = ["Search result text"]
        thinking_blocks = [{"type": "thinking", "thinking": "I need to search for this..."}]

        assistant_msg, user_msg = WebSearchTransformation.transform_response(
            tool_calls=tool_calls,
            search_results=search_results,
            thinking_blocks=thinking_blocks,
        )

        # Check assistant message has thinking block first, then tool_use
        assert len(assistant_msg["content"]) == 2
        assert assistant_msg["content"][0]["type"] == "thinking"
        assert assistant_msg["content"][1]["type"] == "tool_use"

    def test_multiple_tool_calls_and_results(self):
        """Should handle multiple tool calls and their results"""
        tool_calls = [
            {"id": "tool_a", "name": "WebSearch", "input": {"query": "q1"}},
            {"id": "tool_b", "name": "WebSearch", "input": {"query": "q2"}},
        ]
        search_results = ["Result A", "Result B"]
        thinking_blocks = []

        assistant_msg, user_msg = WebSearchTransformation.transform_response(
            tool_calls=tool_calls,
            search_results=search_results,
            thinking_blocks=thinking_blocks,
        )

        # Check tool_use blocks in assistant message
        assert len(assistant_msg["content"]) == 2
        assert assistant_msg["content"][0]["id"] == "tool_a"
        assert assistant_msg["content"][1]["id"] == "tool_b"

        # Check tool_result blocks in user message
        assert len(user_msg["content"]) == 2
        assert user_msg["content"][0]["tool_use_id"] == "tool_a"
        assert user_msg["content"][0]["content"] == "Result A"
        assert user_msg["content"][1]["tool_use_id"] == "tool_b"
        assert user_msg["content"][1]["content"] == "Result B"


class TestFormatSearchResponse:
    """Tests for WebSearchTransformation.format_search_response()"""

    def test_formats_search_response_with_results(self):
        """Should format SearchResponse with results into readable text"""
        # Create mock SearchResponse
        result1 = Mock()
        result1.title = "First Result"
        result1.url = "https://example.com/1"
        result1.snippet = "This is the first snippet."

        result2 = Mock()
        result2.title = "Second Result"
        result2.url = "https://example.com/2"
        result2.snippet = "This is the second snippet."

        search_response = Mock()
        search_response.results = [result1, result2]

        formatted = WebSearchTransformation.format_search_response(search_response)

        assert "Title: First Result" in formatted
        assert "URL: https://example.com/1" in formatted
        assert "Snippet: This is the first snippet." in formatted
        assert "Title: Second Result" in formatted

    def test_formats_empty_results(self):
        """Should handle SearchResponse with no results"""
        search_response = Mock()
        search_response.results = []

        formatted = WebSearchTransformation.format_search_response(search_response)

        # Should fallback to str(result)
        assert formatted  # Not empty

    def test_formats_response_without_results_attribute(self):
        """Should fallback to str() for responses without results attribute"""

        # Create a simple class without 'results' attribute that converts to string
        class SimpleResponse:
            def __str__(self):
                return "Fallback string representation"

        search_response = SimpleResponse()

        formatted = WebSearchTransformation.format_search_response(search_response)

        # Should use str() fallback since no results attribute
        assert formatted == "Fallback string representation"
