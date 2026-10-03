"""The hallucinated multi_tool_use.parallel expansion must not fail the response."""

import json

import pytest

from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    _handle_invalid_parallel_tool_calls,
)
from litellm.types.utils import ChatCompletionMessageToolCall, Function


def _parallel_instance(arguments: str):
    return [
        ChatCompletionMessageToolCall(
            id="call_1",
            type="function",
            function=Function(name="multi_tool_use.parallel", arguments=arguments),
        )
    ]


@pytest.mark.parametrize(
    "arguments",
    [
        '{"tool_uses": [{"recipient_name": "functions.get_weather"}]}',
        '{"tool_uses": "nope"}',
        '{"tool_uses": [42]}',
        "{}",
    ],
)
def test_malformed_tool_uses_returns_original_calls(arguments):
    """A hallucinated payload we cannot expand must come back untouched."""
    tool_calls = _parallel_instance(arguments)

    result = _handle_invalid_parallel_tool_calls(tool_calls)

    assert len(result) == 1
    assert result[0].function.name == "multi_tool_use.parallel"
    assert result[0].id == "call_1"
    assert result[0].function.arguments == arguments


def test_valid_parallel_payload_still_expands():
    """The guard must not swallow well-formed payloads."""
    tool_calls = _parallel_instance(
        json.dumps(
            {
                "tool_uses": [
                    {
                        "recipient_name": "functions.get_weather",
                        "parameters": {"city": "NYC"},
                    },
                    {"recipient_name": "functions.get_time", "parameters": {"tz": "EST"}},
                ]
            }
        )
    )

    result = _handle_invalid_parallel_tool_calls(tool_calls)

    assert [c.function.name for c in result] == ["get_weather", "get_time"]
