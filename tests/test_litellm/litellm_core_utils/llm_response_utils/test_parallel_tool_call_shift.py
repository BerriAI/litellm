"""Every hallucinated multi_tool_use.parallel call in one message must expand.

Replacing one entry with len(expansions) entries moves everything after it by
len(expansions) - 1; advancing by the full length skipped one entry per
expansion, so with two such calls the second stayed in place (and the call
after it was overwritten).
"""

import json

from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    convert_to_model_response_object,
)
from litellm.types.utils import ModelResponse


def _parallel(call_id: str, *calls):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "multi_tool_use.parallel",
            "arguments": json.dumps(
                {
                    "tool_uses": [
                        {"recipient_name": name, "parameters": params} for name, params in calls
                    ]
                }
            ),
        },
    }


def _plain(call_id: str, name: str):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }


def test_every_parallel_tool_call_expands():
    response_object = {
        "id": "chatcmpl-parallel",
        "object": "chat.completion",
        "created": 1728933352,
        "model": "gpt-4o-2024-08-06",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        _plain("0", "get_weather"),
                        # Two expansions here and one below, so the first splice
                        # changes the list length and the second has to land at
                        # the shifted offset.
                        _parallel(
                            "m1",
                            ("functions.get_time", {"tz": "UTC"}),
                            ("functions.get_date", {"tz": "UTC"}),
                        ),
                        _plain("2", "get_news"),
                        _parallel("m2", ("functions.get_quote", {"sym": "AAPL"})),
                        _plain("4", "get_forecast"),
                    ],
                },
            }
        ],
    }

    result = convert_to_model_response_object(
        response_object=response_object,
        model_response_object=ModelResponse(),
        response_type="completion",
    )

    names = [tc.function.name for tc in result.choices[0].message.tool_calls]
    assert names == [
        "get_weather",
        "get_time",
        "get_date",
        "get_news",
        "get_quote",
        "get_forecast",
    ]
