"""Tests for remaining uncovered methods in tools.py.

These tests cover functions not already covered by test_webfetch_interception.py.
"""

import pytest

from litellm.integrations.webfetch_interception.tools import (
    get_litellm_web_fetch_tool_openai,
    is_web_fetch_tool_chat_completion,
    is_native_fetch_tool,
    convert_native_fetch_to_litellm,
    LITELLM_NATIVE_FETCH_TOOLS,
    LITELLM_WEB_FETCH_TOOL_NAME,
)


class TestGetLitellmWebFetchToolOpenAI:
    """Test OpenAI format tool definition."""

    def test_returns_correct_name(self):
        """Tool name matches constant."""
        tool = get_litellm_web_fetch_tool_openai()
        assert tool["type"] == "function"
        assert tool["function"]["name"] == LITELLM_WEB_FETCH_TOOL_NAME

    def test_has_url_property(self):
        """Schema requires url parameter."""
        tool = get_litellm_web_fetch_tool_openai()
        assert "url" in tool["function"]["parameters"]["properties"]
        assert "required" in tool["function"]["parameters"]


class TestIsNativeFetchTool:
    """Test native fetch tool detection."""

    def test_web_fetch_20250305(self):
        """Anthropic format is native."""
        assert is_native_fetch_tool("web_fetch_20250305") is True

    def test_web_fetch_name(self):
        """web_fetch name is native."""
        assert is_native_fetch_tool("web_fetch") is True

    def test_WebFetch(self):
        """Legacy WebFetch is native."""
        assert is_native_fetch_tool("WebFetch") is True

    def test_non_native(self):
        """Other tools are not native."""
        assert is_native_fetch_tool("calculator") is False
        assert is_native_fetch_tool("") is False

    def test_type_matching(self):
        """Tool type matching works."""
        assert is_native_fetch_tool("anything", "web_fetch_20250305") is True
        assert is_native_fetch_tool("anything", "other_type") is False


class TestIsWebFetchToolChatCompletion:
    """Test chat completion tool detection."""

    def test_litellm_name(self):
        """LiteLLM tool name is detected."""
        tool = {"function": {"name": LITELLM_WEB_FETCH_TOOL_NAME}}
        assert is_web_fetch_tool_chat_completion(tool) is True

    def test_anthropic_type(self):
        """Anthropic type tool."""
        tool = {"type": "web_fetch_20250305"}
        assert is_web_fetch_tool_chat_completion(tool) is True

    def test_non_fetch(self):
        """Non-fetch tool not detected."""
        assert is_web_fetch_tool_chat_completion({"function": {"name": "other"}}) is False


class TestConvertNativeFetchToLitellm:
    """Test native to LiteLLM format conversion."""

    def test_basic_conversion(self):
        """Basic tool conversion."""
        native = {
            "type": "web_fetch_20250305",
            "name": "web_fetch",
            "input_schema": {
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        }

        result = convert_native_fetch_to_litellm(native)

        assert result["name"] == LITELLM_WEB_FETCH_TOOL_NAME
        assert "url" in result["input_schema"]["properties"]

    def test_preserves_description(self):
        """Description is preserved if present."""
        native = {
            "description": "Custom fetch",
            "input_schema": {},
        }

        result = convert_native_fetch_to_litellm(native)
        assert result.get("description") == "Custom fetch"

    def test_preserves_required(self):
        """Required fields preserved."""
        native = {
            "input_schema": {
                "required": ["url", "extra"],
            },
        }

        result = convert_native_fetch_to_litellm(native)
        assert "extra" in result["input_schema"]["required"]

    def test_empty_input_schema(self):
        """Empty input schema handled gracefully."""
        native = {
            "type": "web_fetch_20250305",
        }

        result = convert_native_fetch_to_litellm(native)
        assert result["name"] == LITELLM_WEB_FETCH_TOOL_NAME
        assert result["input_schema"]["properties"]["url"]["type"] == "string"
