

import litellm
from litellm.integrations.lunary import parse_messages, parse_tool_calls
from litellm.types.utils import (
    ChatCompletionMessageCustomToolCall,
    ChatCompletionMessageToolCall,
    Function,
)


def test_parse_tool_calls_serializes_custom_tool_calls():
    custom_call = ChatCompletionMessageCustomToolCall(
        id="call_c",
        custom={"name": "ApplyPatch", "input": "*** Begin Patch"},
    )
    function_call = ChatCompletionMessageToolCall(
        id="call_f",
        type="function",
        function=Function(name="read_file", arguments='{"path": "a.py"}'),
    )
    parsed = parse_tool_calls([custom_call, function_call])
    assert parsed == [
        {
            "type": "custom",
            "id": "call_c",
            "function": {"name": "ApplyPatch", "arguments": "*** Begin Patch"},
        },
        {
            "type": "function",
            "id": "call_f",
            "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
        },
    ]


def test_parse_tool_calls_none_passthrough():
    assert parse_tool_calls(None) is None


def test_parse_messages_lists_server_tool_calls_before_client_calls():
    response = litellm.ModelResponse(
        choices=[
            litellm.Choices(
                index=0,
                finish_reason="tool_calls",
                message=litellm.Message(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "SF"}'},
                        }
                    ],
                    provider_specific_fields={
                        "server_tool_calls": [
                            {
                                "id": "srvtoolu_01",
                                "type": "function",
                                "function": {"name": "web_search", "arguments": '{"query": "persimmon"}'},
                            }
                        ]
                    },
                ),
            )
        ],
    )
    assert parse_messages(response.choices) == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "type": "function",
                "id": "srvtoolu_01",
                "function": {"name": "web_search", "arguments": '{"query": "persimmon"}'},
            },
            {"type": "function", "id": "call_1", "function": {"name": "get_weather", "arguments": '{"city": "SF"}'}},
        ],
    }
