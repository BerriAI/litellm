"""Tests for validate_and_fix_openai_messages (issue #25880)."""

import warnings

from litellm.types.utils import ChatCompletionMessageToolCall, Function, Message
from litellm.utils import validate_and_fix_openai_messages


def _make_tool_call_message() -> Message:
    tc = ChatCompletionMessageToolCall(
        id="toolu_1",
        type="function",
        function=Function(name="add", arguments='{"a":1,"b":2}'),
    )
    return Message(content="", role="assistant", tool_calls=[tc])


def test_no_pydantic_serialization_warning_with_tool_calls():
    msg = _make_tool_call_message()

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message=".*serialized value may not be as expected.*",
            category=UserWarning,
        )
        result = validate_and_fix_openai_messages(
            [
                {"role": "user", "content": "hi"},
                msg,
                {"role": "tool", "tool_call_id": "toolu_1", "content": "3"},
            ]
        )

    assert len(result) == 3
    assert all(isinstance(m, dict) for m in result)
    assistant_msg = result[1]
    assert assistant_msg["role"] == "assistant"
    assert isinstance(assistant_msg["tool_calls"][0], dict)
    assert assistant_msg["tool_calls"][0]["function"]["name"] == "add"


def test_original_message_not_mutated():
    msg = _make_tool_call_message()
    original_tc = msg.tool_calls[0]

    validate_and_fix_openai_messages([{"role": "user", "content": "hi"}, msg])

    assert isinstance(msg.tool_calls[0], ChatCompletionMessageToolCall)
    assert msg.tool_calls[0] is original_tc


def test_dict_messages_with_tool_calls_still_work():
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


def test_missing_role_defaults_to_assistant_without_mutating_input():
    original = {"content": "some content"}

    result = validate_and_fix_openai_messages([original])

    assert result[0]["role"] == "assistant"
    assert original == {"content": "some content"}
