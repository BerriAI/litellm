"""
Test message sanitization for Anthropic API when modify_params=True

Tests three cases:
A. Missing tool_result for tool_use (orphaned tool calls)
B. Orphaned tool_result without matching tool_use
C. Empty text content
"""

import pytest
import sys
import os

# Add the parent directory to the path so we can import litellm
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)

import litellm
from litellm.litellm_core_utils.prompt_templates.factory import (
    sanitize_messages_for_tool_calling,
    anthropic_messages_pt,
)


class TestMessageSanitization:
    """Test message sanitization for tool calling scenarios"""

    def setup_method(self):
        """Setup for each test"""
        # Save original modify_params value
        self.original_modify_params = litellm.modify_params
        litellm.modify_params = True

    def teardown_method(self):
        """Cleanup after each test"""
        # Restore original modify_params value
        litellm.modify_params = self.original_modify_params

    def test_case_a_orphaned_tool_call_single(self):
        """
        Test Case A: Assistant message with tool_calls but no tool result
        Should add a dummy tool result message
        """
        messages = [
            {"role": "user", "content": "What is the weather in Nashik?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "toolu_01Kus2cC3ydjBW7UK4GJqBP4",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Nashik, India"}',
                        },
                    }
                ],
            },
        ]

        sanitized = sanitize_messages_for_tool_calling(messages)

        # Should have 3 messages: user, assistant, and dummy tool result
        assert len(sanitized) == 3
        assert sanitized[0]["role"] == "user"
        assert sanitized[1]["role"] == "assistant"
        assert sanitized[2]["role"] == "tool"
        assert sanitized[2]["tool_call_id"] == "toolu_01Kus2cC3ydjBW7UK4GJqBP4"
        assert (
            "skipped" in sanitized[2]["content"].lower()
            or "interrupted" in sanitized[2]["content"].lower()
        )
        assert "get_weather" in sanitized[2]["content"]

    def test_case_a_orphaned_tool_call_multiple(self):
        """
        Test Case A: Assistant message with multiple tool_calls, some missing results
        """
        messages = [
            {"role": "user", "content": "Get weather for Nashik and Mumbai"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Nashik"}',
                        },
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Mumbai"}',
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "Weather in Nashik: 25°C",
            },
        ]

        sanitized = sanitize_messages_for_tool_calling(messages)

        # Should have 4 messages: user, assistant, tool result for call_1, dummy for call_2
        assert len(sanitized) == 4
        assert sanitized[0]["role"] == "user"
        assert sanitized[1]["role"] == "assistant"
        assert (
            sanitized[2]["tool_call_id"] == "call_1"
        )  # Original tool result (first in tool_calls)
        assert (
            sanitized[3]["tool_call_id"] == "call_2"
        )  # Dummy added for missing call_2

    def test_case_b_orphaned_tool_result(self):
        """
        Test Case B: Tool result without matching tool_call in previous assistant message
        Should remove the orphaned tool result
        """
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {
                "role": "tool",
                "tool_call_id": "nonexistent_id",
                "content": "Some result",
            },
        ]

        sanitized = sanitize_messages_for_tool_calling(messages)

        # Should have only 2 messages, orphaned tool result removed
        assert len(sanitized) == 2
        assert sanitized[0]["role"] == "user"
        assert sanitized[1]["role"] == "assistant"

    def test_case_b_valid_tool_result_preserved(self):
        """
        Test Case B: Valid tool result with matching tool_call should be preserved
        """
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Boston"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_123", "content": "Weather: 20°C"},
        ]

        sanitized = sanitize_messages_for_tool_calling(messages)

        # All messages should be preserved
        assert len(sanitized) == 3
        assert sanitized[2]["role"] == "tool"
        assert sanitized[2]["tool_call_id"] == "call_123"

    def test_case_c_empty_text_content_user(self):
        """
        Test Case C: Empty text content in user message
        Should replace with placeholder
        """
        messages = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "Hello!"},
        ]

        sanitized = sanitize_messages_for_tool_calling(messages)

        assert len(sanitized) == 2
        assert sanitized[0]["role"] == "user"
        assert (
            sanitized[0]["content"]
            == "[System: Empty message content sanitised to satisfy protocol]"
        )

    def test_case_c_whitespace_only_content(self):
        """
        Test Case C: Whitespace-only content
        Should replace with placeholder
        """
        messages = [
            {"role": "user", "content": "   \n  \t  "},
            {"role": "assistant", "content": "  "},
        ]

        sanitized = sanitize_messages_for_tool_calling(messages)

        assert len(sanitized) == 2
        assert (
            sanitized[0]["content"]
            == "[System: Empty message content sanitised to satisfy protocol]"
        )
        assert (
            sanitized[1]["content"]
            == "[System: Empty message content sanitised to satisfy protocol]"
        )

    def test_case_c_valid_content_preserved(self):
        """
        Test Case C: Valid non-empty content should be preserved
        """
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        sanitized = sanitize_messages_for_tool_calling(messages)

        assert len(sanitized) == 2
        assert sanitized[0]["content"] == "Hello"
        assert sanitized[1]["content"] == "Hi there!"

    def test_combined_cases(self):
        """
        Test combination of multiple cases
        """
        messages = [
            {"role": "user", "content": "Get weather"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "NYC"}',
                        },
                    }
                ],
            },
            # Missing tool result for call_1
            {"role": "user", "content": ""},  # Empty content
            {"role": "assistant", "content": "Response"},
            {
                "role": "tool",
                "tool_call_id": "orphaned_id",  # Orphaned tool result
                "content": "Some data",
            },
        ]

        sanitized = sanitize_messages_for_tool_calling(messages)

        # Should have: user, assistant, dummy tool result, user (sanitized), assistant
        # Orphaned tool result should be removed
        assert len(sanitized) == 5
        assert sanitized[0]["role"] == "user"
        assert sanitized[1]["role"] == "assistant"
        assert sanitized[2]["role"] == "tool"
        assert sanitized[2]["tool_call_id"] == "call_1"  # Dummy added
        assert sanitized[3]["role"] == "user"
        assert (
            sanitized[3]["content"]
            == "[System: Empty message content sanitised to satisfy protocol]"
        )
        assert sanitized[4]["role"] == "assistant"

    def test_modify_params_false_no_sanitization(self):
        """
        Test that sanitization is skipped when modify_params=False
        """
        litellm.modify_params = False

        messages = [
            {"role": "user", "content": ""},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
        ]

        sanitized = sanitize_messages_for_tool_calling(messages)

        # Messages should be unchanged
        assert len(sanitized) == 2
        assert sanitized[0]["content"] == ""
        assert len(sanitized[1].get("tool_calls", [])) == 1

    def test_anthropic_messages_pt_integration(self):
        """
        Test that sanitization is integrated into anthropic_messages_pt
        """
        litellm.modify_params = True

        messages = [
            {"role": "user", "content": "What is the weather in Nashik?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "toolu_01Kus2cC3ydjBW7UK4GJqBP4",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Nashik, India"}',
                        },
                    }
                ],
            },
        ]

        # This should not raise an error and should add dummy tool result
        result = anthropic_messages_pt(
            messages=messages, model="claude-sonnet-4-5", llm_provider="anthropic"
        )

        # Should have at least 2 messages (user and assistant)
        # The tool result will be merged into user content
        assert len(result) >= 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    def test_empty_string_content_sanitized_without_modify_params(self):
        """
        Regression: An empty user message ({"role": "user", "content": ""}) must
        be rewritten to a non-empty placeholder *before* it reaches Anthropic,
        even when litellm.modify_params is False. Otherwise Anthropic returns:
            "messages: text content blocks must be non-empty"
        Reproduces a real failure from the pr-review agent (pydantic-ai).
        """
        litellm.modify_params = False

        messages = [
            {"role": "user", "content": "First message"},
            {"role": "user", "content": "please review this"},
            {"role": "user", "content": ""},
        ]

        result = anthropic_messages_pt(
            messages=messages, model="claude-sonnet-4-5", llm_provider="anthropic"
        )

        # All three user messages get merged into one user turn for Anthropic.
        assert len(result) == 1
        assert result[0]["role"] == "user"
        text_blocks = [
            b for b in result[0]["content"] if isinstance(b, dict) and b.get("type") == "text"
        ]
        assert len(text_blocks) == 3
        # No text block may be empty — that's the contract Anthropic enforces.
        for block in text_blocks:
            assert block["text"].strip() != ""
        assert text_blocks[2]["text"] == (
            "[System: Empty message content sanitised to satisfy protocol]"
        )

    def test_empty_text_block_in_list_content_sanitized(self):
        """
        Same regression for the list-of-blocks form:
            {"role": "user", "content": [{"type": "text", "text": ""}]}
        Empty text *blocks* must be rewritten too, regardless of modify_params.
        """
        litellm.modify_params = False

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "real content"},
                    {"type": "text", "text": ""},
                    {"type": "text", "text": "  \n  "},
                ],
            },
        ]

        result = anthropic_messages_pt(
            messages=messages, model="claude-sonnet-4-5", llm_provider="anthropic"
        )

        assert len(result) == 1
        text_blocks = [
            b for b in result[0]["content"] if isinstance(b, dict) and b.get("type") == "text"
        ]
        assert len(text_blocks) == 3
        assert text_blocks[0]["text"] == "real content"
        for block in text_blocks[1:]:
            assert block["text"].strip() != ""

    def test_non_empty_content_unchanged_without_modify_params(self):
        """
        Sanity check: when nothing is empty, the messages flow through unchanged
        even with modify_params disabled.
        """
        litellm.modify_params = False

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ]

        result = anthropic_messages_pt(
            messages=messages, model="claude-sonnet-4-5", llm_provider="anthropic"
        )

        # Two user turns + one assistant turn (alternation preserved).
        assert len(result) == 3
        assert result[0]["content"][0]["text"] == "Hello"
        assert result[1]["content"][0]["text"] == "Hi there"
        assert result[2]["content"][0]["text"] == "How are you?"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
