"""
Tests for ``_handle_invalid_parallel_tool_calls`` in
``litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response``.

The function replaces each hallucinated ``multi_tool_use.parallel`` tool call
with the real tool calls it encoded. These tests focus on the case where a
single message carries more than one ``multi_tool_use.parallel`` call, which
exercises the running-offset bookkeeping used to splice replacements in place.
"""

import json

from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    _handle_invalid_parallel_tool_calls,
)
from litellm.types.utils import ChatCompletionMessageToolCall, Function


def _multi_tool_use(call_id, tool_uses):
    return ChatCompletionMessageToolCall(
        id=call_id,
        type="function",
        function=Function(
            name="multi_tool_use.parallel",
            arguments=json.dumps({"tool_uses": tool_uses}),
        ),
    )


def test_multiple_multi_tool_use_parallel_all_expanded():
    """Two multi_tool_use.parallel calls in one message must both be expanded."""
    tool_calls = [
        _multi_tool_use(
            "call_1",
            [
                {"recipient_name": "functions.get_weather", "parameters": {"city": "NYC"}},
                {"recipient_name": "functions.get_time", "parameters": {"tz": "EST"}},
            ],
        ),
        _multi_tool_use(
            "call_2",
            [
                {"recipient_name": "functions.get_stock", "parameters": {"ticker": "AAPL"}},
            ],
        ),
    ]

    result = _handle_invalid_parallel_tool_calls(tool_calls)

    # No hallucinated wrapper call should survive.
    assert all(tc.function.name != "multi_tool_use.parallel" for tc in result)

    names = [tc.function.name for tc in result]
    assert names == ["get_weather", "get_time", "get_stock"]

    ids = [tc.id for tc in result]
    assert ids == ["call_1_0", "call_1_1", "call_2_0"]


def test_multi_tool_use_parallel_interleaved_with_real_call():
    """A real tool call between two parallel calls keeps its position and value."""
    tool_calls = [
        _multi_tool_use(
            "call_1",
            [
                {"recipient_name": "functions.get_weather", "parameters": {"city": "NYC"}},
            ],
        ),
        ChatCompletionMessageToolCall(
            id="call_real",
            type="function",
            function=Function(name="lookup", arguments='{"q": "x"}'),
        ),
        _multi_tool_use(
            "call_3",
            [
                {"recipient_name": "functions.get_stock", "parameters": {"ticker": "AAPL"}},
            ],
        ),
    ]

    result = _handle_invalid_parallel_tool_calls(tool_calls)

    assert all(tc.function.name != "multi_tool_use.parallel" for tc in result)
    assert [tc.function.name for tc in result] == ["get_weather", "lookup", "get_stock"]
    assert [tc.id for tc in result] == ["call_1_0", "call_real", "call_3_0"]
