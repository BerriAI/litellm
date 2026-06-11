"""Differential parity: v2 IR translation vs the v1 AnthropicConfig chain.

For each OpenAI chat request in the corpus, the v1 body is produced exactly as
``litellm.completion`` would (``map_openai_params`` then ``transform_request``)
and compared, as normalized JSON, to the v2 body from ``translate_chat_request``.
The corpus is the covered surface: the anthropic v2 flag only turns on for the
shapes pinned here. A divergence on any shape fails the build, which is what
keeps the rewrite honest while it grows.
"""

import copy
import json

import pytest

from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.translation import translate_chat_request

MODEL = "claude-3-5-sonnet-20241022"

_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def _assistant_tool_call(call_id, city):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"city": city}),
                },
            }
        ],
    }


CORPUS = {
    "text": {
        "model": MODEL,
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Hello, world"}],
    },
    "system_and_sampling": {
        "model": MODEL,
        "max_tokens": 50,
        "temperature": 0.5,
        "top_p": 0.9,
        "messages": [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ],
    },
    "multiturn_stop_stream": {
        "model": MODEL,
        "max_tokens": 64,
        "stop": ["END", "STOP"],
        "stream": True,
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ],
    },
    "no_max_tokens": {
        "model": MODEL,
        "messages": [{"role": "user", "content": "no max tokens here"}],
    },
    "tools_auto": {
        "model": MODEL,
        "max_tokens": 200,
        "tools": [_WEATHER_TOOL],
        "tool_choice": "auto",
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
    },
    "tool_choice_required": {
        "model": MODEL,
        "max_tokens": 200,
        "tools": [_WEATHER_TOOL],
        "tool_choice": "required",
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
    },
    "tool_choice_none": {
        "model": MODEL,
        "max_tokens": 200,
        "tools": [_WEATHER_TOOL],
        "tool_choice": "none",
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
    },
    "tool_choice_specific": {
        "model": MODEL,
        "max_tokens": 200,
        "tools": [_WEATHER_TOOL],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
    },
    "tool_call_roundtrip": {
        "model": MODEL,
        "max_tokens": 200,
        "messages": [
            {"role": "user", "content": "Weather in Paris?"},
            _assistant_tool_call("call_1", "Paris"),
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 20C"},
        ],
    },
    "parallel_tool_results_merge": {
        "model": MODEL,
        "max_tokens": 200,
        "messages": [
            {"role": "user", "content": "Weather in Paris and Rome?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"city": "Paris"}),
                        },
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"city": "Rome"}),
                        },
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 20C"},
            {"role": "tool", "tool_call_id": "call_2", "content": "Cloudy, 18C"},
        ],
    },
    "assistant_text_and_tool_call": {
        "model": MODEL,
        "max_tokens": 200,
        "messages": [
            {"role": "user", "content": "Weather in Paris?"},
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"city": "Paris"}),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 20C"},
        ],
    },
    "image_data_uri": {
        "model": MODEL,
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANS"},
                    },
                ],
            }
        ],
    },
}


def _v1_body(request: dict) -> dict:
    config = AnthropicConfig()
    model = request["model"]
    params = {
        key: value for key, value in request.items() if key not in ("model", "messages")
    }
    optional = config.map_openai_params(
        copy.deepcopy(params), {}, model, drop_params=False
    )
    return config.transform_request(
        model, copy.deepcopy(request["messages"]), optional, {}, {}
    )


def _v2_body(request: dict) -> dict:
    result = translate_chat_request(copy.deepcopy(request), "anthropic")
    assert result.is_ok(), result.error.summary
    return result.ok


def _norm(body: dict) -> str:
    return json.dumps(body, sort_keys=True, default=str)


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_v2_matches_v1(name: str) -> None:
    request = CORPUS[name]
    assert _norm(_v2_body(request)) == _norm(_v1_body(request))
