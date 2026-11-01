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
    gemini_convert_messages_with_history,
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
    contents = gemini_convert_messages_with_history(messages=messages)

    # The function response must be present and carry the correct tool name.
    function_responses = [
        part["function_response"]
        for content in contents
        for part in content["parts"]
        if isinstance(part, dict) and part.get("function_response")
    ]
    assert function_responses, f"expected a functionResponse part, got: {contents}"
    assert function_responses[0]["name"] == "shell"


def _function_response_names(contents):
    return [
        part["function_response"]["name"]
        for content in contents
        for part in content["parts"]
        if isinstance(part, dict) and part.get("function_response")
    ]


def test_tool_results_match_tool_calls_split_across_assistant_messages():
    """Each tool result is matched against every tool call of the merged model turn.

    Gemini streams `[functionCall, thought, functionCall]` when it reasons between two
    calls. Replayed as chat messages this becomes two assistant messages with one tool
    call each; the result for the first call used to be looked up only in the second
    message and failed with "Missing corresponding tool call" (#49063).
    """
    messages = [
        {"role": "user", "content": "Weather in Paris and the time in Rome?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_a",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                }
            ],
        },
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "Now the time in Rome.",
            "tool_calls": [
                {
                    "id": "call_b",
                    "type": "function",
                    "function": {"name": "get_time", "arguments": '{"city": "Rome"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_a", "content": "sunny"},
        {"role": "tool", "tool_call_id": "call_b", "content": "noon"},
    ]

    contents = gemini_convert_messages_with_history(messages=messages)

    function_calls = [
        part["function_call"]["name"]
        for content in contents
        for part in content["parts"]
        if isinstance(part, dict) and part.get("function_call")
    ]
    assert function_calls == ["get_weather", "get_time"]
    assert _function_response_names(contents) == ["get_weather", "get_time"]
    # The original assistant messages are not modified.
    assert [tc["id"] for tc in messages[2]["tool_calls"]] == ["call_b"]


def test_responses_input_with_reasoning_between_function_calls_converts_for_gemini():
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )

    chat_messages = LiteLLMCompletionResponsesConfig._transform_response_input_param_to_chat_completion_message(
        input=[
            {"role": "user", "content": "Weather in Paris and Rome?"},
            {
                "type": "function_call",
                "call_id": "call_a",
                "name": "get_weather",
                "arguments": '{"city": "Paris"}',
            },
            {
                "type": "reasoning",
                "id": "rs_1",
                "content": [{"type": "output_text", "text": "Now Rome too."}],
            },
            {
                "type": "function_call",
                "call_id": "call_b",
                "name": "get_weather",
                "arguments": '{"city": "Rome"}',
            },
            {"type": "function_call_output", "call_id": "call_a", "output": "sunny"},
            {"type": "function_call_output", "call_id": "call_b", "output": "cloudy"},
        ],
        replay_reasoning=True,
    )

    contents = gemini_convert_messages_with_history(messages=chat_messages)

    assert _function_response_names(contents) == ["get_weather", "get_weather"]
