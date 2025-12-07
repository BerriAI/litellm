"""
Unit tests for Kimi-K2-Thinking and MiniMax M2 Bedrock support.

Tests the tool call parsing and model identification functions.
"""

import pytest

from litellm.llms.bedrock.tool_parsers import (
    is_kimi_model,
    is_minimax_m2_model,
    needs_kimi_parsing,
    parse_kimi_tool_calls,
)


class TestKimiModelIdentification:
    """Tests for model identification functions."""

    def test_is_kimi_model_positive(self):
        """Should correctly identify Kimi models."""
        assert is_kimi_model("moonshot.kimi-k2-thinking") is True
        assert is_kimi_model("bedrock/converse/moonshot.kimi-k2-thinking") is True
        assert is_kimi_model("MOONSHOT.KIMI-K2-THINKING") is True  # case insensitive

    def test_is_kimi_model_negative(self):
        """Should not match non-Kimi models."""
        assert is_kimi_model("anthropic.claude-3-sonnet") is False
        assert is_kimi_model("minimax.minimax-m2") is False
        assert is_kimi_model("moonshot/kimi-k2-thinking") is False  # different format

    def test_is_minimax_m2_model_positive(self):
        """Should correctly identify MiniMax M2 models."""
        assert is_minimax_m2_model("minimax.minimax-m2") is True
        assert is_minimax_m2_model("bedrock/converse/minimax.minimax-m2") is True
        assert is_minimax_m2_model("MINIMAX.MINIMAX-M2") is True  # case insensitive

    def test_is_minimax_m2_model_negative(self):
        """Should not match non-MiniMax models."""
        assert is_minimax_m2_model("anthropic.claude-3-sonnet") is False
        assert is_minimax_m2_model("moonshot.kimi-k2-thinking") is False
        assert is_minimax_m2_model("openrouter/minimax/minimax-m2") is False


class TestKimiToolCallParsing:
    """Tests for Kimi tool call parsing functions."""

    def test_needs_kimi_parsing_with_tool_calls_section(self):
        """Should detect tool calls section markers."""
        text = "<|tool_calls_section_begin|>some content<|tool_calls_section_end|>"
        assert needs_kimi_parsing(text) is True

    def test_needs_kimi_parsing_with_tool_call_begin(self):
        """Should detect individual tool call markers."""
        text = "text before <|tool_call_begin|>function_name"
        assert needs_kimi_parsing(text) is True

    def test_needs_kimi_parsing_no_markers(self):
        """Should return False when no markers present."""
        assert needs_kimi_parsing("regular text content") is False
        assert needs_kimi_parsing("") is False

    def test_parse_kimi_tool_calls_single_call(self):
        """Should parse a single tool call correctly."""
        text = '<|tool_call_begin|>get_weather<|tool_call_argument_begin|>{"location": "NYC"}<|tool_call_end|>'
        tool_calls = parse_kimi_tool_calls(text)

        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "get_weather"
        assert tool_calls[0]["function"]["arguments"] == '{"location": "NYC"}'
        assert tool_calls[0]["type"] == "function"
        assert tool_calls[0]["index"] == 0

    def test_parse_kimi_tool_calls_multiple_calls(self):
        """Should parse multiple tool calls correctly."""
        text = (
            '<|tool_call_begin|>get_weather<|tool_call_argument_begin|>{"city": "NYC"}<|tool_call_end|>'
            '<|tool_call_begin|>get_time<|tool_call_argument_begin|>{"timezone": "EST"}<|tool_call_end|>'
        )
        tool_calls = parse_kimi_tool_calls(text)

        assert len(tool_calls) == 2
        assert tool_calls[0]["function"]["name"] == "get_weather"
        assert tool_calls[1]["function"]["name"] == "get_time"
        assert tool_calls[0]["index"] == 0
        assert tool_calls[1]["index"] == 1

    def test_parse_kimi_tool_calls_no_markers(self):
        """Should return empty list when no tool calls present."""
        text = "Just regular text without any tool calls"
        tool_calls = parse_kimi_tool_calls(text)

        assert len(tool_calls) == 0

    def test_parse_kimi_tool_calls_with_whitespace(self):
        """Should handle whitespace in tool call markers."""
        text = '<|tool_call_begin|> get_weather <|tool_call_argument_begin|> {"city": "LA"} <|tool_call_end|>'
        tool_calls = parse_kimi_tool_calls(text)

        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "get_weather"

    def test_parse_kimi_tool_calls_generates_unique_ids(self):
        """Should generate unique IDs for each tool call."""
        text = (
            '<|tool_call_begin|>func1<|tool_call_argument_begin|>{"a": 1}<|tool_call_end|>'
            '<|tool_call_begin|>func2<|tool_call_argument_begin|>{"b": 2}<|tool_call_end|>'
        )
        tool_calls = parse_kimi_tool_calls(text)

        assert tool_calls[0]["id"] != tool_calls[1]["id"]
        assert tool_calls[0]["id"].startswith("call_kimi_")
        assert tool_calls[1]["id"].startswith("call_kimi_")
