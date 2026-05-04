"""Extended tests for transformation.py to cover all uncovered methods.

Cover transform_response, _build_anthropic_messages, _build_openai_messages,
convert_native_to_litellm, and format_fetch_response.
"""

import json
import pytest
from unittest.mock import MagicMock

from litellm.integrations.webfetch_interception.transformation import (
    WebFetchTransformation,
)
from litellm.llms.base_llm.fetch.transformation import WebFetchResponse


class TestTransformRequest:
    """Test transform_request."""

    def test_streaming_return_false(self):
        """Streaming responses return no tool calls."""
        result = WebFetchTransformation.transform_request({}, stream=True)
        assert result == (False, [])

    def test_openai_format(self):
        """OpenAI format detection."""
        response = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call-1",
                        "function": {
                            "name": "litellm-web-fetch",
                            "arguments": json.dumps({"url": "https://example.com"}),
                        }
                    }]
                }
            }]
        }
        result = WebFetchTransformation.transform_request(
            response, stream=False, response_format="openai"
        )
        assert result[0] is True
        assert len(result[1]) == 1

    def test_anthropic_format(self):
        """Anthropic format detection."""
        response = {
            "content": [{
                "type": "tool_use",
                "name": "litellm-web-fetch",
                "id": "call-1",
                "input": {"url": "https://example.com"},
            }]
        }
        result = WebFetchTransformation.transform_request(
            response, stream=False, response_format="anthropic"
        )
        assert result[0] is True
        assert len(result[1]) == 1

    def test_no_content(self):
        """Empty content returns no tool calls."""
        result = WebFetchTransformation.transform_request(
            {"content": []}, stream=False
        )
        assert result == (False, [])

    def test_object_response(self):
        """Response as object with .content attribute."""
        mock_response = MagicMock()
        mock_response.content = [{
            "type": "tool_use",
            "name": "litellm-web-fetch",
            "id": "call-1",
            "input": {"url": "https://example.com"},
        }]
        result = WebFetchTransformation.transform_request(
            mock_response, stream=False
        )
        assert result[0] is True

    def test_no_content_attribute(self):
        """Response without content attribute returns no tool calls."""
        mock_response = MagicMock()
        result = WebFetchTransformation.transform_request(
            mock_response, stream=False
        )
        assert result == (False, [])


class TestDetectFromOpenAIResponse:
    """Test _detect_from_openai_response."""

    def test_no_choices(self):
        """No choices returns no tool calls."""
        result = WebFetchTransformation._detect_from_openai_response({})
        assert result == (False, [])

    def test_no_tool_calls(self):
        """No tool_calls returns no tool calls."""
        response = {"choices": [{"message": {}}]}
        result = WebFetchTransformation._detect_from_openai_response(response)
        assert result == (False, [])

    def test_model_response_object(self):
        """ModelResponse object with choices."""
        mock_choice = MagicMock()
        mock_choice.message = MagicMock()
        mock_choice.message.tool_calls = None

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        result = WebFetchTransformation._detect_from_openai_response(mock_response)
        assert result == (False, [])

    def test_json_string_arguments(self):
        """JSON string arguments are parsed."""
        response = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call-1",
                        "function": {
                            "name": "litellm-web-fetch",
                            "arguments": '{"url": "https://example.com"}',
                        }
                    }]
                }
            }]
        }
        result = WebFetchTransformation._detect_from_openai_response(response)
        assert result[0] is True
        assert isinstance(result[1][0]["function"]["arguments"], dict)

    def test_invalid_json_arguments(self):
        """Invalid JSON string defaults to empty dict."""
        response = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call-1",
                        "function": {
                            "name": "litellm-web-fetch",
                            "arguments": "not json",
                        }
                    }]
                }
            }]
        }
        result = WebFetchTransformation._detect_from_openai_response(response)
        assert result[0] is True
        assert result[1][0]["function"]["arguments"] == {}


class TestTransformResponse:
    """Test transform_response."""

    def test_anthropic_format(self):
        """Anthropic format transformation."""
        tool_calls = [{"id": "call-1", "name": "litellm-web-fetch", "input": {"url": "https://example.com"}}]
        fetch_results = ["# Hello\nWorld"]

        result = WebFetchTransformation.transform_response(
            tool_calls,
            fetch_results,
            response_format="anthropic",
        )

        assert "assistant_message" in result
        assert "tool_result_message" in result
        assert result["assistant_message"]["role"] == "assistant"

    def test_openai_format(self):
        """OpenAI format transformation."""
        tool_calls = [{"id": "call-1", "function": {"name": "litellm-web-fetch", "arguments": {"url": "https://example.com"}}}]
        fetch_results = ["Test content"]

        result = WebFetchTransformation.transform_response(
            tool_calls,
            fetch_results,
            response_format="openai",
        )

        assert "assistant_message" in result
        assert "tool_result_message" in result
        assert result["assistant_message"]["role"] == "assistant"

    def test_with_thinking_blocks(self):
        """Preserve thinking blocks in response."""
        tool_calls = [{"id": "call-1", "name": "litellm-web-fetch", "input": {"url": "https://example.com"}}]
        fetch_results = ["Content"]
        thinking_blocks = [{"type": "thinking", "thinking": "reasoning"}]

        result = WebFetchTransformation.transform_response(
            tool_calls,
            fetch_results,
            response_format="anthropic",
            thinking_blocks=thinking_blocks,
        )

        assert "content" in result["assistant_message"]

    def test_empty_tool_calls(self):
        """Empty tool calls handled gracefully."""
        result = WebFetchTransformation.transform_response(
            [],
            [],
            response_format="anthropic",
        )
        assert result[0] == {}


class TestBuildAnthropicMessages:
    """Test _build_anthropic_messages."""

    def test_basic(self):
        """Build messages from tool calls and results."""
        tool_calls = [{"id": "call-1", "name": "litellm-web-fetch", "input": {"url": "https://example.com"}}]
        fetch_results = ["# Title\nContent"]

        result = WebFetchTransformation._build_anthropic_messages(
            tool_calls, fetch_results
        )

        assert result[0]["role"] == "assistant"
        assert result[1]["role"] == "user"
        assert "# Title" in result[1]["content"][0]["text"]


class TestBuildOpenAIMessages:
    """Test _build_openai_messages."""

    def test_basic(self):
        """Build OpenAI messages."""
        tool_calls = [{"id": "call-1", "function": {"name": "litellm-web-fetch", "arguments": {"url": "https://example.com"}}}]
        fetch_results = ["OpenAI result"]

        result = WebFetchTransformation._build_openai_messages(
            tool_calls, fetch_results
        )

        assert result[0]["role"] == "assistant"
        assert result[1]["role"] == "tool"
        assert "OpenAI result" in result[1]["content"]


class TestFormatFetchResponse:
    """Test format_fetch_response."""

    def test_with_all_fields(self):
        """Format with title, URL, and content."""
        response = WebFetchResponse(
            url="https://example.com",
            title="Test Page",
            content="Hello World",
        )
        result = WebFetchTransformation.format_fetch_response(response)
        assert "Test Page" in result
        assert "https://example.com" in result
        assert "Hello World" in result

    def test_no_title(self):
        """Format without title."""
        response = WebFetchResponse(
            url="https://example.com",
            content="Hello",
        )
        result = WebFetchTransformation.format_fetch_response(response)
        assert "https://example.com" in result
        assert "Hello" in result

    def test_empty_content(self):
        """Format with empty content."""
        response = WebFetchResponse(
            url="https://example.com",
            title="Empty",
            content="",
        )
        result = WebFetchTransformation.format_fetch_response(response)
        assert "Empty" in result


class TestConvertNativeToLitellm:
    """Test convert_native_to_litellm."""

    def test_anthropic_native(self):
        """Convert Anthropic native tool to LiteLLM format."""
        native_tool = {
            "type": "web_fetch_20250305",
            "name": "web_fetch",
            "input_schema": {
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        }
        result = WebFetchTransformation.convert_native_to_litellm(native_tool)
        assert result["name"] == "litellm-web-fetch"
        assert "url" in result["input_schema"]["properties"]

    def test_missing_input_schema(self):
        """Convert with missing input_schema."""
        native_tool = {"type": "web_fetch_20250305"}
        result = WebFetchTransformation.convert_native_to_litellm(native_tool)
        assert result["name"] == "litellm-web-fetch"
        assert result["input_schema"]["properties"]["url"]["type"] == "string"
