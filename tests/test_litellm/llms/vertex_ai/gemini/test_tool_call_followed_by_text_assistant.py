"""
Regression test for tool-call / tool-result matching in the Gemini message converter.

When an assistant message that contains tool_calls is followed by a *second* assistant
message that has no tool_calls (e.g. the model emits a short narration turn after the
tool call but before the tool result), the converter used to overwrite its
`last_message_with_tool_calls` reference with the text-only assistant message. The
subsequent tool result could then no longer be matched to its tool call, and conversion
failed with:

    Exception: Missing corresponding tool call for tool response message.

This happens for any OpenAI-style history with that shape, independent of provider/model.
"""

import pytest

from litellm.llms.vertex_ai.gemini.transformation import (
    _gemini_convert_messages_with_history,
)


def _messages_with_text_assistant_between_tool_call_and_result():
    return [
        {"role": "user", "content": "list the files"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {"name": "shell", "arguments": '{"command": ["ls"]}'},
                }
            ],
        },
        # text-only assistant message in between (no tool_calls)
        {"role": "assistant", "content": "Running the command now."},
        {"role": "tool", "tool_call_id": "call_abc123", "content": "math.py"},
    ]


def test_tool_result_matches_tool_call_with_text_assistant_in_between():
    messages = _messages_with_text_assistant_between_tool_call_and_result()

    # Should not raise "Missing corresponding tool call for tool response message".
    contents = _gemini_convert_messages_with_history(messages=messages)

    # The function response must be present and carry the correct tool name.
    function_responses = [
        part["function_response"]
        for content in contents
        for part in content["parts"]
        if isinstance(part, dict) and part.get("function_response")
    ]
    assert function_responses, f"expected a functionResponse part, got: {contents}"
    assert function_responses[0]["name"] == "shell"
