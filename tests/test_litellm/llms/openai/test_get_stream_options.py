"""
Tests for OpenAIChatCompletion.get_stream_options

Ensures token usage is always requested from the API for streaming responses,
regardless of the api_base hostname.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.llms.openai.openai import OpenAIChatCompletion


class TestGetStreamOptions:
    """Tests for OpenAIChatCompletion.get_stream_options"""

    def setup_method(self):
        self.openai_chat = OpenAIChatCompletion()

    def test_explicit_stream_options_takes_precedence(self):
        """When stream_options is explicitly provided, it should be returned as-is"""
        result = self.openai_chat.get_stream_options(
            stream_options={"include_usage": True, "continuous_usage": True},
            api_base="https://my-proxy.example.com/v1",
        )
        assert result == {
            "stream_options": {"include_usage": True, "continuous_usage": True}
        }

    def test_default_includes_usage_for_unknown_api_base(self):
        """
        Non-OpenAI api_base endpoints should still receive
        stream_options: {include_usage: True} by default.
        This is the core fix for the issue.
        """
        result = self.openai_chat.get_stream_options(
            stream_options=None,
            api_base="https://my-proxy.example.com/v1",
        )
        assert result == {"stream_options": {"include_usage": True}}

    def test_default_includes_usage_for_openai_api_base(self):
        """Existing behavior: api.openai.com still gets include_usage"""
        result = self.openai_chat.get_stream_options(
            stream_options=None,
            api_base="https://api.openai.com/v1",
        )
        assert result == {"stream_options": {"include_usage": True}}

    def test_default_includes_usage_when_api_base_is_none(self):
        """When api_base is None, should still default to include_usage"""
        result = self.openai_chat.get_stream_options(
            stream_options=None,
            api_base=None,
        )
        assert result == {"stream_options": {"include_usage": True}}

    def test_default_includes_usage_for_localhost(self):
        """Local development endpoints should also get include_usage"""
        result = self.openai_chat.get_stream_options(
            stream_options=None,
            api_base="http://localhost:8000/v1",
        )
        assert result == {"stream_options": {"include_usage": True}}

    def test_explicit_false_stream_options_is_honored(self):
        """
        If a caller explicitly passes stream_options without include_usage,
        it should be honored (not overridden).
        """
        result = self.openai_chat.get_stream_options(
            stream_options={"include_usage": False},
            api_base="https://my-proxy.example.com/v1",
        )
        assert result == {"stream_options": {"include_usage": False}}