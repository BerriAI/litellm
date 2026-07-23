import os
import sys

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.integrations.lunary import parse_tool_calls
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
