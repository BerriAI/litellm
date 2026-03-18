"""
Tests for AnthropicMessagesConfig._sanitize_empty_text_content_blocks()

Covers the fix for https://github.com/BerriAI/litellm/issues/22930:
Claude returns assistant messages with empty text blocks alongside tool_use,
but rejects them when sent back (400: text content blocks must be non-empty).
"""

import copy

from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)

config = AnthropicMessagesConfig()


class TestSanitizeEmptyTextBlocks:
    """Unit tests for _sanitize_empty_text_content_blocks"""

    def test_empty_text_block_stripped_alongside_tool_use(self):
        """Main bug scenario: empty text block next to tool_use should be removed."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": ""},
                    {
                        "type": "tool_use",
                        "id": "toolu_123",
                        "name": "get_weather",
                        "input": {"location": "NYC"},
                    },
                ],
            }
        ]
        result = config._sanitize_empty_text_content_blocks(messages)
        assert len(result) == 1
        assert len(result[0]["content"]) == 1
        assert result[0]["content"][0]["type"] == "tool_use"

    def test_non_empty_text_blocks_preserved(self):
        """Non-empty text blocks should be kept as-is."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Here is the result:"},
                    {
                        "type": "tool_use",
                        "id": "toolu_456",
                        "name": "search",
                        "input": {"q": "test"},
                    },
                ],
            }
        ]
        result = config._sanitize_empty_text_content_blocks(messages)
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][0]["text"] == "Here is the result:"

    def test_whitespace_only_text_blocks_stripped(self):
        """Whitespace-only text blocks should be treated as empty."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "   \n\t  "},
                    {
                        "type": "tool_use",
                        "id": "toolu_789",
                        "name": "calc",
                        "input": {},
                    },
                ],
            }
        ]
        result = config._sanitize_empty_text_content_blocks(messages)
        assert len(result[0]["content"]) == 1
        assert result[0]["content"][0]["type"] == "tool_use"

    def test_all_empty_content_replaced_with_placeholder(self):
        """If all blocks are empty text, replace with a placeholder."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": ""},
                    {"type": "text", "text": "  "},
                ],
            }
        ]
        result = config._sanitize_empty_text_content_blocks(messages)
        assert len(result[0]["content"]) == 1
        assert result[0]["content"][0] == {"type": "text", "text": "..."}

    def test_string_content_untouched(self):
        """Messages with string content should pass through unchanged."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = config._sanitize_empty_text_content_blocks(messages)
        assert result == messages

    def test_no_content_key_untouched(self):
        """Messages without a content key should pass through."""
        messages = [{"role": "assistant", "tool_calls": []}]
        result = config._sanitize_empty_text_content_blocks(messages)
        assert result == messages

    def test_user_message_content_also_sanitized(self):
        """User messages with list content should also be sanitized."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ""},
                    {"type": "text", "text": "actual question"},
                ],
            }
        ]
        result = config._sanitize_empty_text_content_blocks(messages)
        assert len(result[0]["content"]) == 1
        assert result[0]["content"][0]["text"] == "actual question"

    def test_tool_result_blocks_untouched(self):
        """tool_result blocks should never be stripped (they don't have type=text)."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_123",
                        "content": "result data",
                    },
                    {"type": "text", "text": ""},
                ],
            }
        ]
        result = config._sanitize_empty_text_content_blocks(messages)
        assert len(result[0]["content"]) == 1
        assert result[0]["content"][0]["type"] == "tool_result"

    def test_input_immutability(self):
        """Original messages must not be mutated."""
        original_messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": ""},
                    {
                        "type": "tool_use",
                        "id": "toolu_imm",
                        "name": "fn",
                        "input": {},
                    },
                ],
            }
        ]
        snapshot = copy.deepcopy(original_messages)
        config._sanitize_empty_text_content_blocks(original_messages)
        assert original_messages == snapshot

    def test_none_text_value_treated_as_empty(self):
        """A text block with text=None should be treated as empty."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": None},
                    {
                        "type": "tool_use",
                        "id": "toolu_none",
                        "name": "fn",
                        "input": {},
                    },
                ],
            }
        ]
        result = config._sanitize_empty_text_content_blocks(messages)
        assert len(result[0]["content"]) == 1
        assert result[0]["content"][0]["type"] == "tool_use"

    def test_multi_message_end_to_end(self):
        """End-to-end scenario with multiple conversation turns."""
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": ""},
                    {
                        "type": "tool_use",
                        "id": "toolu_w1",
                        "name": "get_weather",
                        "input": {"location": "NYC"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_w1",
                        "content": "72F and sunny",
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "The weather in NYC is 72F and sunny."},
                ],
            },
        ]
        result = config._sanitize_empty_text_content_blocks(messages)
        # First user message: string content, unchanged
        assert result[0]["content"] == "What's the weather?"
        # Assistant with tool_use: empty text removed
        assert len(result[1]["content"]) == 1
        assert result[1]["content"][0]["type"] == "tool_use"
        # Tool result: unchanged
        assert len(result[2]["content"]) == 1
        assert result[2]["content"][0]["type"] == "tool_result"
        # Final assistant: non-empty text preserved
        assert len(result[3]["content"]) == 1
        assert "72F" in result[3]["content"][0]["text"]

    def test_empty_messages_list(self):
        """Empty messages list should return empty list."""
        assert config._sanitize_empty_text_content_blocks([]) == []
