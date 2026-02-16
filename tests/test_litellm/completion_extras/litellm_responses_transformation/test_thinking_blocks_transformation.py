"""
Test suite for thinking_blocks transformation in OpenAI Responses API.

Tests the conversion of thinking_blocks from Chat Completion format to
OpenAI Responses API reasoning items format.
"""

import json
import os
import sys
from typing import List
from unittest.mock import MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system-path

import litellm
from litellm.completion_extras.litellm_responses_transformation.transformation import (
    LiteLLMResponsesTransformationHandler,
)


def test_regular_thinking_block_conversion():
    """
    Test that regular thinking blocks (type: 'thinking') are converted to reasoning items
    with summary generated from thinking text.
    """
    handler = LiteLLMResponsesTransformationHandler()

    messages = [
        {
            "role": "assistant",
            "content": "The answer is 42",
            "thinking_blocks": [
                {
                    "type": "thinking",
                    "thinking": "Let me think about this problem step by step...",
                    "signature": "abc123",
                }
            ],
        }
    ]

    input_items, _ = handler.convert_chat_completion_messages_to_responses_api(messages)

    # Find reasoning item
    reasoning_items = [item for item in input_items if item.get("type") == "reasoning"]
    assert len(reasoning_items) == 1, "Should have 1 reasoning item"

    reasoning_item = reasoning_items[0]
    assert reasoning_item["type"] == "reasoning"
    assert reasoning_item["role"] == "assistant"
    assert "summary" in reasoning_item
    assert len(reasoning_item["summary"]) == 1
    assert reasoning_item["summary"][0]["type"] == "summary_text"
    assert (
        "Let me think about this problem" in reasoning_item["summary"][0]["text"]
    ), "Summary should contain thinking text"

    # Verify message comes after reasoning
    message_items = [item for item in input_items if item.get("type") == "message"]
    assert len(message_items) == 1, "Should have 1 message item"
    reasoning_index = input_items.index(reasoning_item)
    message_index = input_items.index(message_items[0])
    assert (
        reasoning_index < message_index
    ), "Reasoning item should come before message"


def test_thinking_block_truncation():
    """
    Test that thinking text longer than 100 chars is truncated with '...' in summary.
    """
    handler = LiteLLMResponsesTransformationHandler()

    long_thinking = "A" * 150  # 150 character string

    messages = [
        {
            "role": "assistant",
            "content": "Answer",
            "thinking_blocks": [{"type": "thinking", "thinking": long_thinking}],
        }
    ]

    input_items, _ = handler.convert_chat_completion_messages_to_responses_api(messages)

    reasoning_items = [item for item in input_items if item.get("type") == "reasoning"]
    assert len(reasoning_items) == 1

    summary_text = reasoning_items[0]["summary"][0]["text"]
    assert len(summary_text) == 103, "Should be 100 chars + '...'"
    assert summary_text.endswith("..."), "Should end with '...'"
    assert summary_text.startswith("A" * 100), "Should start with first 100 chars"


def test_redacted_thinking_block_conversion():
    """
    Test that redacted thinking blocks (type: 'redacted_thinking') are converted
    to reasoning items with encrypted_content preserved.
    """
    handler = LiteLLMResponsesTransformationHandler()

    encrypted_blob = "encrypted_blob_xyz789"

    messages = [
        {
            "role": "assistant",
            "content": "The answer is 42",
            "thinking_blocks": [
                {"type": "redacted_thinking", "data": encrypted_blob}
            ],
        }
    ]

    input_items, _ = handler.convert_chat_completion_messages_to_responses_api(messages)

    # Find reasoning item
    reasoning_items = [item for item in input_items if item.get("type") == "reasoning"]
    assert len(reasoning_items) == 1, "Should have 1 reasoning item"

    reasoning_item = reasoning_items[0]
    assert reasoning_item["type"] == "reasoning"
    assert reasoning_item["role"] == "assistant"
    assert "encrypted_content" in reasoning_item
    assert (
        reasoning_item["encrypted_content"] == encrypted_blob
    ), "encrypted_content should match data field"
    assert (
        "summary" not in reasoning_item
    ), "Redacted thinking should not have summary"


def test_multiple_thinking_blocks():
    """
    Test that multiple thinking blocks (mixed types) are all converted in correct order.
    """
    handler = LiteLLMResponsesTransformationHandler()

    messages = [
        {
            "role": "assistant",
            "content": "Final answer",
            "thinking_blocks": [
                {"type": "thinking", "thinking": "First thought"},
                {"type": "redacted_thinking", "data": "encrypted_1"},
                {"type": "thinking", "thinking": "Second thought"},
                {"type": "redacted_thinking", "data": "encrypted_2"},
            ],
        }
    ]

    input_items, _ = handler.convert_chat_completion_messages_to_responses_api(messages)

    # Find all reasoning items
    reasoning_items = [item for item in input_items if item.get("type") == "reasoning"]
    assert len(reasoning_items) == 4, "Should have 4 reasoning items"

    # Verify order and content
    assert "summary" in reasoning_items[0]
    assert "First thought" in reasoning_items[0]["summary"][0]["text"]

    assert "encrypted_content" in reasoning_items[1]
    assert reasoning_items[1]["encrypted_content"] == "encrypted_1"

    assert "summary" in reasoning_items[2]
    assert "Second thought" in reasoning_items[2]["summary"][0]["text"]

    assert "encrypted_content" in reasoning_items[3]
    assert reasoning_items[3]["encrypted_content"] == "encrypted_2"

    # Verify all reasoning items come before message
    message_items = [item for item in input_items if item.get("type") == "message"]
    assert len(message_items) == 1
    message_index = input_items.index(message_items[0])
    for reasoning_item in reasoning_items:
        reasoning_index = input_items.index(reasoning_item)
        assert reasoning_index < message_index, "All reasoning before message"


def test_thinking_blocks_with_tool_calls():
    """
    Test that thinking blocks are added BEFORE tool_calls in the output.
    """
    handler = LiteLLMResponsesTransformationHandler()

    messages = [
        {
            "role": "assistant",
            "thinking_blocks": [
                {"type": "thinking", "thinking": "I need to call a function"}
            ],
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "SF"}',
                    },
                }
            ],
        }
    ]

    input_items, _ = handler.convert_chat_completion_messages_to_responses_api(messages)

    # Find reasoning and function_call items
    reasoning_items = [item for item in input_items if item.get("type") == "reasoning"]
    function_call_items = [
        item for item in input_items if item.get("type") == "function_call"
    ]

    assert len(reasoning_items) == 1, "Should have 1 reasoning item"
    assert len(function_call_items) == 1, "Should have 1 function_call item"

    # Verify order: reasoning BEFORE function_call
    reasoning_index = input_items.index(reasoning_items[0])
    function_call_index = input_items.index(function_call_items[0])
    assert (
        reasoning_index < function_call_index
    ), "Reasoning should come before function call"

    # Verify reasoning content
    assert "summary" in reasoning_items[0]
    assert "I need to call a function" in reasoning_items[0]["summary"][0]["text"]

    # Verify function_call content
    assert function_call_items[0]["call_id"] == "call_123"
    assert function_call_items[0]["name"] == "get_weather"


def test_empty_and_none_thinking_blocks():
    """
    Test that empty or None thinking_blocks are handled gracefully without errors.
    """
    handler = LiteLLMResponsesTransformationHandler()

    # Test with None thinking_blocks
    messages_none = [
        {"role": "assistant", "content": "Answer", "thinking_blocks": None}
    ]

    input_items_none, _ = handler.convert_chat_completion_messages_to_responses_api(
        messages_none
    )
    reasoning_items_none = [
        item for item in input_items_none if item.get("type") == "reasoning"
    ]
    assert len(reasoning_items_none) == 0, "Should have no reasoning items for None"

    # Test with empty list thinking_blocks
    messages_empty = [{"role": "assistant", "content": "Answer", "thinking_blocks": []}]

    input_items_empty, _ = handler.convert_chat_completion_messages_to_responses_api(
        messages_empty
    )
    reasoning_items_empty = [
        item for item in input_items_empty if item.get("type") == "reasoning"
    ]
    assert len(reasoning_items_empty) == 0, "Should have no reasoning items for empty"

    # Test without thinking_blocks key
    messages_no_key = [{"role": "assistant", "content": "Answer"}]

    input_items_no_key, _ = handler.convert_chat_completion_messages_to_responses_api(
        messages_no_key
    )
    reasoning_items_no_key = [
        item for item in input_items_no_key if item.get("type") == "reasoning"
    ]
    assert (
        len(reasoning_items_no_key) == 0
    ), "Should have no reasoning items without key"

    # All should have message items
    assert len([item for item in input_items_none if item.get("type") == "message"]) == 1
    assert (
        len([item for item in input_items_empty if item.get("type") == "message"]) == 1
    )
    assert (
        len([item for item in input_items_no_key if item.get("type") == "message"]) == 1
    )


def test_multi_turn_conversation_with_encrypted_content():
    """
    Test that encrypted_content is preserved across multi-turn conversations.
    This simulates passing thinking_blocks from a previous response back in the next request.
    """
    handler = LiteLLMResponsesTransformationHandler()

    # Simulate a multi-turn conversation
    encrypted_content_from_previous = "encrypted_response_abc123"

    messages = [
        {"role": "user", "content": "What's 2+2?"},
        {
            "role": "assistant",
            "content": "4",
            "thinking_blocks": [
                {"type": "redacted_thinking", "data": encrypted_content_from_previous}
            ],
        },
        {"role": "user", "content": "Now multiply that by 3"},
    ]

    input_items, _ = handler.convert_chat_completion_messages_to_responses_api(messages)

    # Find reasoning item from assistant message
    reasoning_items = [item for item in input_items if item.get("type") == "reasoning"]
    assert len(reasoning_items) == 1, "Should have 1 reasoning item"

    # Verify encrypted_content is preserved exactly
    assert "encrypted_content" in reasoning_items[0]
    assert (
        reasoning_items[0]["encrypted_content"] == encrypted_content_from_previous
    ), "encrypted_content must be preserved exactly for multi-turn"

    # Verify message order: user -> reasoning -> assistant message -> user
    assert input_items[0]["type"] == "message"
    assert input_items[0]["role"] == "user"
    assert input_items[1]["type"] == "reasoning"
    assert input_items[2]["type"] == "message"
    assert input_items[2]["role"] == "assistant"
    assert input_items[3]["type"] == "message"
    assert input_items[3]["role"] == "user"


def test_unknown_thinking_block_type():
    """
    Test that unknown thinking block types are skipped gracefully without errors.
    """
    handler = LiteLLMResponsesTransformationHandler()

    messages = [
        {
            "role": "assistant",
            "content": "Answer",
            "thinking_blocks": [
                {"type": "unknown_type", "some_field": "some_value"},
                {"type": "thinking", "thinking": "Valid thinking"},
            ],
        }
    ]

    input_items, _ = handler.convert_chat_completion_messages_to_responses_api(messages)

    # Should only have 1 reasoning item (the valid one)
    reasoning_items = [item for item in input_items if item.get("type") == "reasoning"]
    assert (
        len(reasoning_items) == 1
    ), "Should skip unknown type and only convert valid one"
    assert "Valid thinking" in reasoning_items[0]["summary"][0]["text"]


def test_empty_thinking_text():
    """
    Test that thinking blocks with empty thinking text still create reasoning items.
    """
    handler = LiteLLMResponsesTransformationHandler()

    messages = [
        {
            "role": "assistant",
            "content": "Answer",
            "thinking_blocks": [{"type": "thinking", "thinking": ""}],
        }
    ]

    input_items, _ = handler.convert_chat_completion_messages_to_responses_api(messages)

    reasoning_items = [item for item in input_items if item.get("type") == "reasoning"]
    assert len(reasoning_items) == 1, "Should create reasoning item even with empty text"
    assert reasoning_items[0]["type"] == "reasoning"
    assert reasoning_items[0]["role"] == "assistant"
    # Empty thinking should not add summary
    assert "summary" not in reasoning_items[0], "Empty thinking should not have summary"


def test_thinking_blocks_only_no_content():
    """
    Test that assistant messages with only thinking_blocks (no content) work correctly.
    The transformation sets content to "" by default, so a message item is created.
    """
    handler = LiteLLMResponsesTransformationHandler()

    messages = [
        {
            "role": "assistant",
            "thinking_blocks": [{"type": "thinking", "thinking": "Just thinking..."}],
            # Note: no content field (defaults to "")
        }
    ]

    input_items, _ = handler.convert_chat_completion_messages_to_responses_api(messages)

    # Should have reasoning item and message item (with empty content)
    reasoning_items = [item for item in input_items if item.get("type") == "reasoning"]
    message_items = [item for item in input_items if item.get("type") == "message"]

    assert len(reasoning_items) == 1, "Should have reasoning item"
    assert len(message_items) == 1, "Should have message item (with empty content)"
    assert "Just thinking..." in reasoning_items[0]["summary"][0]["text"]

    # Verify reasoning comes before message
    reasoning_index = input_items.index(reasoning_items[0])
    message_index = input_items.index(message_items[0])
    assert reasoning_index < message_index, "Reasoning should come before message"
