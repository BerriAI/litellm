"""
Test empty text content block sanitization for the /v1/messages native path.

The Anthropic API returns assistant messages with empty text blocks
({"type": "text", "text": ""}) alongside tool_use blocks, but rejects
them when sent back. The /v1/messages endpoint must strip these before
forwarding to providers.

Ref: https://github.com/BerriAI/litellm/issues/22930
"""

import pytest

from litellm.llms.custom_httpx.llm_http_handler import (
    _sanitize_anthropic_messages_empty_text_blocks,
)


class TestSanitizeAnthropicMessagesEmptyTextBlocks:
    """Unit tests for _sanitize_anthropic_messages_empty_text_blocks."""

    def test_strips_empty_text_alongside_tool_use(self):
        """
        The most common case from the bug report: an assistant message
        containing an empty text block next to a tool_use block.
        """
        messages = [
            {"role": "user", "content": "Run the command."},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": ""},
                    {
                        "type": "tool_use",
                        "id": "toolu_xxx",
                        "name": "Bash",
                        "input": {"command": "ls"},
                    },
                ],
            },
        ]

        result = _sanitize_anthropic_messages_empty_text_blocks(messages)

        assert len(result) == 2
        assert result[0] == messages[0]  # user message unchanged
        # assistant content should only have the tool_use block
        assert len(result[1]["content"]) == 1
        assert result[1]["content"][0]["type"] == "tool_use"

    def test_preserves_nonempty_text_blocks(self):
        """Non-empty text blocks must not be removed."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check that."},
                    {
                        "type": "tool_use",
                        "id": "toolu_yyy",
                        "name": "Bash",
                        "input": {"command": "pwd"},
                    },
                ],
            },
        ]

        result = _sanitize_anthropic_messages_empty_text_blocks(messages)

        assert len(result[0]["content"]) == 2
        assert result[0]["content"][0] == {"type": "text", "text": "Let me check that."}

    def test_whitespace_only_text_block_stripped(self):
        """Whitespace-only text blocks should also be stripped."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "   \n\t  "},
                    {
                        "type": "tool_use",
                        "id": "toolu_zzz",
                        "name": "Bash",
                        "input": {},
                    },
                ],
            },
        ]

        result = _sanitize_anthropic_messages_empty_text_blocks(messages)

        assert len(result[0]["content"]) == 1
        assert result[0]["content"][0]["type"] == "tool_use"

    def test_all_empty_text_blocks_replaced_with_placeholder(self):
        """
        If all content blocks are empty text, replace with a placeholder
        to avoid sending an empty content array.
        """
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": ""},
                ],
            },
        ]

        result = _sanitize_anthropic_messages_empty_text_blocks(messages)

        assert len(result[0]["content"]) == 1
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][0]["text"].strip()  # must be non-empty

    def test_string_content_untouched(self):
        """Messages with string content should pass through unchanged."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        result = _sanitize_anthropic_messages_empty_text_blocks(messages)

        assert result == messages

    def test_no_content_key_untouched(self):
        """Messages without a content key should pass through."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant"},
        ]

        result = _sanitize_anthropic_messages_empty_text_blocks(messages)

        assert result == messages

    def test_user_message_content_list_also_sanitized(self):
        """
        Empty text blocks should be stripped from user messages too,
        not just assistant messages.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ""},
                    {"type": "text", "text": "actual question"},
                ],
            },
        ]

        result = _sanitize_anthropic_messages_empty_text_blocks(messages)

        assert len(result[0]["content"]) == 1
        assert result[0]["content"][0]["text"] == "actual question"

    def test_tool_result_content_blocks_untouched(self):
        """
        tool_result content blocks should not be affected — only
        {"type": "text", "text": ""} blocks are stripped.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_xxx",
                        "content": "",
                    },
                ],
            },
        ]

        result = _sanitize_anthropic_messages_empty_text_blocks(messages)

        assert result == messages

    def test_multiple_messages_mixed(self):
        """End-to-end scenario with multiple messages, some needing sanitization."""
        messages = [
            {"role": "user", "content": "Run ls"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": ""},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "Bash",
                        "input": {"command": "ls"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "file1.txt\nfile2.txt",
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Here are the files:"},
                ],
            },
        ]

        result = _sanitize_anthropic_messages_empty_text_blocks(messages)

        # First message: string content, unchanged
        assert result[0] == messages[0]
        # Second message: empty text stripped, only tool_use remains
        assert len(result[1]["content"]) == 1
        assert result[1]["content"][0]["type"] == "tool_use"
        # Third message: tool_result, unchanged
        assert result[2] == messages[2]
        # Fourth message: non-empty text, unchanged
        assert result[3] == messages[3]

    def test_does_not_mutate_original_messages(self):
        """The function should not modify the input list or its dicts."""
        original_content = [
            {"type": "text", "text": ""},
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "Bash",
                "input": {},
            },
        ]
        messages = [
            {
                "role": "assistant",
                "content": original_content,
            },
        ]

        _sanitize_anthropic_messages_empty_text_blocks(messages)

        # Original message content should be unchanged
        assert len(messages[0]["content"]) == 2
        assert messages[0]["content"][0] == {"type": "text", "text": ""}
