"""
Regression: preserve function_call_output ordering.

Gemini/Vertex requires tool outputs to immediately follow the assistant tool call.
The ResponsesAPI->Chat conversion must not move tool outputs to the end.
"""

from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)


def test_function_call_output_stays_adjacent_to_tool_call():
    msgs = LiteLLMCompletionResponsesConfig._transform_response_input_param_to_chat_completion_message(
        input=[
            {
                "role": "user",
                "type": "message",
                "content": [{"type": "input_text", "text": "Call echo with 'hello'."}],
            },
            {
                "type": "function_call",
                "name": "echo",
                "call_id": "call_123",
                "arguments": '{"text":"hello"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": '{"text":"hello"}',
            },
            {
                "role": "assistant",
                "type": "message",
                "content": [{"type": "output_text", "text": "Done."}],
            },
            {
                "role": "user",
                "type": "message",
                "content": [{"type": "input_text", "text": "Now say hi."}],
            },
        ]
    )

    # Find the assistant message that contains tool_calls
    tool_call_idx = None
    tool_msg_idx = None
    assistant_ok_idx = None

    for i, m in enumerate(msgs):
        if isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls"):
            tool_call_idx = i
        if isinstance(m, dict) and m.get("role") == "tool":
            tool_msg_idx = i

        # Assistant "Done." can be either a plain string or a structured content list
        if isinstance(m, dict) and m.get("role") == "assistant":
            content = m.get("content")
            if content == "Done.":
                assistant_ok_idx = i
            elif isinstance(content, list):
                for block in content:
                    if (
                        isinstance(block, dict)
                        and block.get("type") == "text"
                        and block.get("text") == "Done."
                    ):
                        assistant_ok_idx = i
                        break

    assert tool_call_idx is not None
    assert tool_msg_idx is not None
    assert assistant_ok_idx is not None

    # Tool output must be right after tool call, and before the assistant "Done." message.
    assert tool_msg_idx == tool_call_idx + 1
    assert assistant_ok_idx > tool_msg_idx


def test_assistant_message_after_tool_call_is_folded_into_it():
    """Codex echoes history as [function_call, assistant message, function_call_output].
    The assistant message must fold into the tool_calls message so the tool result
    stays immediately after it (DeepSeek and Anthropic reject it otherwise)."""
    msgs = LiteLLMCompletionResponsesConfig._transform_response_input_param_to_chat_completion_message(
        input=[
            {
                "role": "user",
                "type": "message",
                "content": [{"type": "input_text", "text": "Add 21 and 21."}],
            },
            {
                "type": "function_call",
                "name": "get_sum",
                "namespace": "mcp__everything",
                "call_id": "call_1",
                "arguments": '{"a":21,"b":21}',
            },
            {
                "role": "assistant",
                "type": "message",
                "content": [{"type": "output_text", "text": ""}],
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "42",
            },
        ]
    )

    roles = [m.get("role") for m in msgs if isinstance(m, dict)]
    assert roles.count("assistant") == 1

    tool_call_idx = next(i for i, m in enumerate(msgs) if isinstance(m, dict) and m.get("tool_calls"))
    assert msgs[tool_call_idx].get("role") == "assistant"
    assert msgs[tool_call_idx + 1].get("role") == "tool"


def test_assistant_message_before_function_call_keeps_one_assistant_turn():
    """The chat->responses bridge emits an assistant message ahead of its function_call.

    Round-tripping that order back to chat must fold both into a single assistant
    turn, so the tool result still follows the message that made the call.
    """
    msgs = LiteLLMCompletionResponsesConfig._transform_response_input_param_to_chat_completion_message(
        input=[
            {
                "role": "user",
                "type": "message",
                "content": [{"type": "input_text", "text": "What is the weather?"}],
            },
            {
                "role": "assistant",
                "type": "message",
                "content": [{"type": "output_text", "text": "Let me check."}],
            },
            {
                "type": "function_call",
                "name": "get_weather",
                "call_id": "call_1",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "sunny",
            },
        ]
    )

    assistant_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") == "assistant"]
    assert len(assistant_msgs) == 1
    assistant = assistant_msgs[0]
    assert assistant["content"] == [{"type": "text", "text": "Let me check."}]
    assert [tc["function"]["name"] for tc in assistant["tool_calls"]] == ["get_weather"]

    assistant_idx = msgs.index(assistant)
    assert msgs[assistant_idx + 1].get("role") == "tool"
    assert msgs[assistant_idx + 1].get("tool_call_id") == "call_1"
