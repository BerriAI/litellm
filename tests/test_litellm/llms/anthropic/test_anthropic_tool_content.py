"""
Tests for Anthropic-format tool_use handling in sanitization and user-message validation.

Covers:
- _is_orphaned_tool_result matching prior assistant tool_use content blocks
- validate_chat_completion_user_messages accepting tool_result content blocks in user messages
"""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)

from litellm.litellm_core_utils.prompt_templates.factory import _is_orphaned_tool_result
from litellm.types.llms.openai import ValidUserMessageContentTypes
from litellm.utils import validate_chat_completion_user_messages


class TestIsOrphanedToolResultAnthropicFormat:
    """_is_orphaned_tool_result with Anthropic-format tool_use content blocks."""

    def test_not_orphaned_when_anthropic_tool_use_present(self):
        """Tool message is not orphaned when the latest assistant has a matching tool_use block."""
        sanitized_messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check that."},
                    {
                        "type": "tool_use",
                        "id": "call_123",
                        "name": "get_weather",
                        "input": {"city": "SF"},
                    },
                ],
            }
        ]
        current_message = {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": "72F sunny",
        }
        assert _is_orphaned_tool_result(current_message, sanitized_messages) is False

    def test_orphaned_when_no_tool_use_in_assistant(self):
        """Tool message is orphaned when the latest assistant has no tool_calls or tool_use."""
        sanitized_messages = [
            {
                "role": "assistant",
                "content": "Just a plain text response",
            }
        ]
        current_message = {
            "role": "tool",
            "tool_call_id": "call_456",
            "content": "result data",
        }
        assert _is_orphaned_tool_result(current_message, sanitized_messages) is True

    def test_not_orphaned_with_openai_format_tool_calls(self):
        """Tool message is not orphaned when the assistant uses OpenAI-format tool_calls."""
        sanitized_messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_789",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    }
                ],
            }
        ]
        current_message = {
            "role": "tool",
            "tool_call_id": "call_789",
            "content": "search results",
        }
        assert _is_orphaned_tool_result(current_message, sanitized_messages) is False


class TestToolResultUserMessageValidation:
    """validate_chat_completion_user_messages with Anthropic-style tool_result blocks."""

    def test_tool_result_in_valid_types(self):
        """tool_result is an allowed user content type for mixed Anthropic-style payloads."""
        assert "tool_result" in ValidUserMessageContentTypes

    def test_validate_accepts_tool_result_in_user_message(self):
        """User messages may include type=tool_result content blocks (e.g. Cursor / Anthropic)."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here is the tool output."},
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_abc",
                        "content": "ok",
                    },
                ],
            }
        ]
        out = validate_chat_completion_user_messages(messages)
        assert out == messages
