"""
Unit tests for Bedrock tool call token cleaning.

Tests the generic token cleaning functions for models
with proprietary tool call formats (like Kimi-K2-Thinking).
"""

import pytest

from litellm.llms.bedrock.tool_parsers import (
    clean_tool_tokens_from_text,
    detect_pattern_in_text,
)


class TestTokenCleaning:
    """Tests for token cleaning functions."""

    def test_detect_pattern_kimi(self):
        text = "Some text <|tool_call_begin|>func<|tool_call_end|>"
        assert detect_pattern_in_text(text) == "pipe_markers"

    def test_detect_pattern_none(self):
        text = "Just regular text"
        assert detect_pattern_in_text(text) is None

    def test_clean_kimi_tokens_simple(self):
        text = 'Hello <|tool_call_begin|>func<|tool_call_argument_begin|>{"x":1}<|tool_call_end|> world'
        cleaned = clean_tool_tokens_from_text(text)
        assert cleaned == "Hello  world"

    def test_clean_kimi_tokens_with_section(self):
        text = (
            "Start "
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>f1<|tool_call_end|>"
            "<|tool_calls_section_end|>"
            " End"
        )
        cleaned = clean_tool_tokens_from_text(text)
        assert cleaned == "Start  End"

    def test_clean_no_tokens(self):
        text = "Clean text"
        cleaned = clean_tool_tokens_from_text(text)
        assert cleaned == "Clean text"

    def test_clean_whitespace_handling(self):
        text = 'Line 1\n<|tool_call_begin|>func<|tool_call_argument_begin|>{"a":1}<|tool_call_end|>\nLine 2'
        cleaned = clean_tool_tokens_from_text(text)
        assert cleaned == "Line 1\n\nLine 2"
