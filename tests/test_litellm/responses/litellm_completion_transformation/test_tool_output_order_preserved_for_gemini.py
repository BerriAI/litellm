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
            {"role": "user", "type": "message", "content": [{"type": "input_text", "text": "Turn on button"}]},
            {
                "type": "function_call",
                "name": "HassTurnOn",
                "call_id": "call_123",
                "arguments": '{"name":"button"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": '{"ok":true}',
            },
            {"role": "assistant", "type": "message", "content": [{"type": "output_text", "text": "OK"}]},
            {"role": "user", "type": "message", "content": [{"type": "input_text", "text": "Turn off button"}]},
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

        # Assistant "OK" can be either a plain string or a structured content list
        if isinstance(m, dict) and m.get("role") == "assistant":
            content = m.get("content")
            if content == "OK":
                assistant_ok_idx = i
            elif isinstance(content, list):
                for block in content:
                    if (
                        isinstance(block, dict)
                        and block.get("type") == "text"
                        and block.get("text") == "OK"
                    ):
                        assistant_ok_idx = i
                        break

    assert tool_call_idx is not None
    assert tool_msg_idx is not None
    assert assistant_ok_idx is not None

    # Tool output must be right after tool call, and before the assistant "OK" message.
    assert tool_msg_idx == tool_call_idx + 1
    assert assistant_ok_idx > tool_msg_idx

