"""
Tests for _strip_empty_text_blocks_from_anthropic_messages.

Covers the fix for https://github.com/BerriAI/litellm/issues/22930:
The /v1/messages endpoint must strip empty text content blocks that Anthropic
returns in assistant responses but rejects on subsequent requests.
"""

import copy

from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
    _strip_empty_text_blocks_from_anthropic_messages,
)


def test_removes_empty_text_block_alongside_tool_use():
    """Assistant message with tool_use + empty text block → text block removed."""
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "What is the weather?"}],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": ""},
                {
                    "type": "tool_use",
                    "id": "toolu_01A",
                    "name": "get_weather",
                    "input": {"location": "SF"},
                },
            ],
        },
    ]
    result = _strip_empty_text_blocks_from_anthropic_messages(messages)
    assistant = result[1]
    assert len(assistant["content"]) == 1
    assert assistant["content"][0]["type"] == "tool_use"


def test_preserves_nonempty_text_block():
    """Non-empty text blocks must not be removed."""
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I'll look that up."},
                {
                    "type": "tool_use",
                    "id": "toolu_01B",
                    "name": "search",
                    "input": {"q": "test"},
                },
            ],
        },
    ]
    original = copy.deepcopy(messages)
    result = _strip_empty_text_blocks_from_anthropic_messages(messages)
    assert result == original


def test_string_content_untouched():
    """Messages with string content (not list) must pass through unchanged."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    original = copy.deepcopy(messages)
    result = _strip_empty_text_blocks_from_anthropic_messages(messages)
    assert result == original


def test_does_not_empty_content_array():
    """If removing empty text blocks would leave content empty, keep them."""
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": ""}],
        },
    ]
    result = _strip_empty_text_blocks_from_anthropic_messages(messages)
    # Content should remain untouched — can't leave it empty
    assert len(result[0]["content"]) == 1


def test_multiple_empty_text_blocks_removed():
    """Multiple empty text blocks are all removed when other blocks exist."""
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": ""},
                {
                    "type": "tool_use",
                    "id": "toolu_01C",
                    "name": "calc",
                    "input": {"expr": "1+1"},
                },
                {"type": "text", "text": ""},
            ],
        },
    ]
    result = _strip_empty_text_blocks_from_anthropic_messages(messages)
    assert len(result[0]["content"]) == 1
    assert result[0]["content"][0]["type"] == "tool_use"


def test_user_messages_also_sanitized():
    """Empty text blocks in user messages are also stripped (defensive)."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": ""},
                {"type": "image", "source": {"type": "base64", "data": "..."}},
            ],
        },
    ]
    result = _strip_empty_text_blocks_from_anthropic_messages(messages)
    assert len(result[0]["content"]) == 1
    assert result[0]["content"][0]["type"] == "image"


def test_whitespace_only_text_blocks_removed():
    """Whitespace-only text blocks (e.g. ' ') are treated as empty and removed."""
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": " "},
                {
                    "type": "tool_use",
                    "id": "toolu_01D",
                    "name": "search",
                    "input": {"q": "test"},
                },
            ],
        },
    ]
    result = _strip_empty_text_blocks_from_anthropic_messages(messages)
    assert len(result[0]["content"]) == 1
    assert result[0]["content"][0]["type"] == "tool_use"


def test_empty_messages_list():
    """Empty messages list returns empty list."""
    assert _strip_empty_text_blocks_from_anthropic_messages([]) == []


def test_real_world_multi_turn_with_tool_result():
    """
    Real-world multi-turn scenario from issue #22930:
    User → Assistant (tool_use + empty text) → tool_result → Assistant
    """
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "What's 2+2?"}],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": ""},
                {
                    "type": "tool_use",
                    "id": "toolu_calc",
                    "name": "calculator",
                    "input": {"expression": "2+2"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_calc",
                    "content": "4",
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "The answer is 4."}],
        },
    ]
    result = _strip_empty_text_blocks_from_anthropic_messages(messages)

    # The second message (assistant with tool_use) should have empty text removed
    assert len(result[1]["content"]) == 1
    assert result[1]["content"][0]["type"] == "tool_use"

    # All other messages remain unchanged
    assert len(result[0]["content"]) == 1  # user
    assert len(result[2]["content"]) == 1  # tool_result
    assert len(result[3]["content"]) == 1  # final assistant
