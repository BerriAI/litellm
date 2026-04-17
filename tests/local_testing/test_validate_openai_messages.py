"""
Tests for validate_and_fix_openai_messages in litellm/utils.py

Covers the fix for https://github.com/BerriAI/litellm/issues/25880:
Pydantic Message objects with tool_calls should not trigger
PydanticSerializationUnexpectedValue warnings.
"""

import warnings

import pytest

from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Function,
    Message,
)
from litellm.utils import validate_and_fix_openai_messages


def _make_tool_call_message():
    """Helper: build an assistant Message with a single tool_call."""
    tc = ChatCompletionMessageToolCall(
        id="toolu_1",
        type="function",
        function=Function(name="add", arguments='{"a":1,"b":2}'),
    )
    return Message(content="", role="assistant", tool_calls=[tc])


def test_no_pydantic_serialization_warning_with_tool_calls():
    """Regression: pydantic Messages with tool_calls must not emit warnings."""
    msg = _make_tool_call_message()

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Pydantic serializer warnings")
        # Should not raise
        result = validate_and_fix_openai_messages(
            [
                {"role": "user", "content": "hi"},
                msg,
                {"role": "tool", "tool_call_id": "toolu_1", "content": "3"},
            ]
        )

    # Basic sanity: all messages come back as dicts
    assert isinstance(result, list)
    assert len(result) == 3
    for m in result:
        assert isinstance(m, dict)

    # The assistant message should have tool_calls serialized as dicts
    assistant_msg = result[1]
    assert assistant_msg["role"] == "assistant"
    assert isinstance(assistant_msg["tool_calls"], list)
    assert isinstance(assistant_msg["tool_calls"][0], dict)
    assert assistant_msg["tool_calls"][0]["function"]["name"] == "add"


def test_original_message_not_mutated():
    """The original pydantic Message should not be mutated by validation."""
    msg = _make_tool_call_message()
    original_tc = msg.tool_calls[0]

    validate_and_fix_openai_messages(
        [
            {"role": "user", "content": "hi"},
            msg,
            {"role": "tool", "tool_call_id": "toolu_1", "content": "3"},
        ]
    )

    # tool_calls on the original message should still be the pydantic model
    assert isinstance(msg.tool_calls[0], ChatCompletionMessageToolCall)
    assert msg.tool_calls[0] is original_tc


def test_dict_messages_still_work():
    """Plain dict messages should still be processed correctly."""
    messages = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "greet", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "hi"},
    ]

    result = validate_and_fix_openai_messages(messages)
    assert len(result) == 3
    assert result[1]["tool_calls"][0]["function"]["name"] == "greet"


def test_missing_role_gets_default():
    """Messages without a role should default to 'assistant'."""
    result = validate_and_fix_openai_messages(
        [{"content": "some content"}]
    )
    assert result[0]["role"] == "assistant"


def test_dict_messages_not_mutated_in_place():
    """Original dict messages should not be mutated."""
    original = {"role": "user", "content": "test"}
    validate_and_fix_openai_messages([original])
    # Should still be the same dict content
    assert original == {"role": "user", "content": "test"}
