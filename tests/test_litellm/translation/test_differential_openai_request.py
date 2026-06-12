"""Differential parity: v2 IR translation vs the v1 OpenAIGPTConfig chain.

For each OpenAI chat request in the corpus, the v1 body is produced exactly as
``litellm.completion`` would for provider "openai" (``map_openai_params`` then
``transform_request``) and compared, as normalized JSON, to the v2 body from
``translate_chat_request``. The corpus is the covered surface: the
openai_compat v2 flag only turns on for the shapes pinned here. v1 is a
near-passthrough (five touches), so byte parity is only possible for shapes
the inbound parse round-trips losslessly; everything else must be a TYPED
fallback (the raw guard / parse / serializer reasons asserted below), never a
silent divergence.
"""

import copy
import json

import pytest

from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.translation import translate_chat_request

from .conftest import build_real_deps

MODEL = "gpt-4o"

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

_STRICT_TOOL = {
    "type": "function",
    "function": {
        "name": "report",
        "parameters": {
            "type": "object",
            "properties": {"body": {"type": "string"}},
            "required": ["body"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


def _assistant_tool_call(call_id, city, name="get_weather"):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps({"city": city})},
            }
        ],
    }


CORPUS = {
    "text": {
        "model": MODEL,
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
    "multiturn_stop_list_stream": {
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
    "max_completion_tokens": {
        "model": MODEL,
        "max_completion_tokens": 128,
        "messages": [{"role": "user", "content": "hi"}],
    },
    "temperature_int_stays_int": {
        "model": MODEL,
        "temperature": 1,
        "messages": [{"role": "user", "content": "hi"}],
    },
    "tools_auto": {
        "model": MODEL,
        "tools": [_WEATHER_TOOL],
        "tool_choice": "auto",
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
    },
    "tools_strict": {
        "model": MODEL,
        "tools": [_STRICT_TOOL],
        "messages": [{"role": "user", "content": "report this"}],
    },
    "tool_choice_required": {
        "model": MODEL,
        "tools": [_WEATHER_TOOL],
        "tool_choice": "required",
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
    },
    "tool_choice_none": {
        "model": MODEL,
        "tools": [_WEATHER_TOOL],
        "tool_choice": "none",
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
    },
    "tool_choice_specific": {
        "model": MODEL,
        "tools": [_WEATHER_TOOL],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
    },
    "parallel_tool_calls_false": {
        "model": MODEL,
        "tools": [_WEATHER_TOOL],
        "parallel_tool_calls": False,
        "messages": [{"role": "user", "content": "Weather in Paris and Rome?"}],
    },
    "tool_call_roundtrip": {
        "model": MODEL,
        "tools": [_WEATHER_TOOL],
        "messages": [
            {"role": "user", "content": "Weather in Paris?"},
            _assistant_tool_call("call_1", "Paris"),
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 20C"},
        ],
    },
    "tool_call_compact_arguments_roundtrip": {
        # real OpenAI responses carry compact argument JSON; the IR rides the
        # verbatim wire bytes (ToolUse.arguments_raw), so the dominant
        # replayed-history shape is served byte-identically, not fallback
        "model": MODEL,
        "tools": [_WEATHER_TOOL],
        "messages": [
            {"role": "user", "content": "w?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"Paris"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        ],
    },
    "tool_call_odd_spacing_and_blank_arguments_roundtrip": {
        # arbitrary spacing and blank argument strings ride through verbatim
        # (v1 forwards the original messages untouched)
        "model": MODEL,
        "tools": [_WEATHER_TOOL],
        "messages": [
            {"role": "user", "content": "w?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{ "city" :  "Paris" }',
                        },
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": ""},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
            {"role": "tool", "tool_call_id": "call_2", "content": "ok"},
        ],
    },
    "image_url_string_to_object": {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this"},
                    {"type": "image_url", "image_url": "https://e.test/a.png"},
                ],
            }
        ],
    },
    "image_base64": {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "and this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                    },
                ],
            }
        ],
    },
    "response_format_json_object": {
        "model": MODEL,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": "json please"}],
    },
    "response_format_json_schema_strict": {
        "model": MODEL,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {
                    "type": "object",
                    "properties": {"capital": {"type": "string"}},
                    "required": ["capital"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        },
        "messages": [{"role": "user", "content": "capital of France?"}],
    },
    "cache_control_stripped_everywhere": {
        # v1 strips cache_control recursively from messages and tools; the IR
        # carries it as typed metadata and the serializer drops it the same way.
        "model": MODEL,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object", "properties": {}},
                    "cache_control": {"type": "ephemeral"},
                },
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "cached context",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"type": "text", "text": "question"},
                ],
            }
        ],
    },
}

# Typed fallbacks: each row must return a TranslationError whose summary
# carries the reason fragment, so the seam serves the request through v1.
# (v1 is NOT invoked for these rows: several would perform I/O, raise, or
# depend on get_optional_params interplay outside map_openai_params.)
EXPECTED_FALLBACKS = {
    "o_series_model": (
        {"model": "o3-mini", "messages": [{"role": "user", "content": "hi"}]},
        "OpenAIOSeriesConfig",
    ),
    "gpt5_model": (
        {"model": "gpt-5", "messages": [{"role": "user", "content": "hi"}]},
        "OpenAIGPT5Config",
    ),
    "http_pdf_file_id": (
        {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "file",
                            "file": {"file_id": "https://e.test/doc.pdf"},
                        }
                    ],
                }
            ],
        },
        # the inbound schema has no file part at all, so v1's in-transform
        # pdf download (gpt_transformation.py:236-257) can never be reached
        "messages",
    ),
    "stop_string_form": (
        {"model": MODEL, "stop": "END", "messages": [{"role": "user", "content": "x"}]},
        "string-form stop",
    ),
    "both_max_tokens_keys": (
        {
            "model": MODEL,
            "max_tokens": 5,
            "max_completion_tokens": 6,
            "messages": [{"role": "user", "content": "x"}],
        },
        "both max_tokens and max_completion_tokens",
    ),
    "message_name_field": (
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "x", "name": "alice"}],
        },
        "message name field",
    ),
    "consecutive_user_messages": (
        {
            "model": MODEL,
            "messages": [
                {"role": "user", "content": "a"},
                {"role": "user", "content": "b"},
            ],
        },
        "consecutive user messages",
    ),
    "image_detail_key": (
        {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "see"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://e.test/a.png",
                                "detail": "low",
                            },
                        },
                    ],
                }
            ],
        },
        "image_url detail/format",
    ),
    "single_text_content_list": (
        {
            "model": MODEL,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "only"}]}
            ],
        },
        "single-text content list",
    ),
    "empty_tools_list": (
        {"model": MODEL, "tools": [], "messages": [{"role": "user", "content": "x"}]},
        "empty tools list",
    ),
    "stream_options_unsupported": (
        {
            "model": MODEL,
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": "x"}],
        },
        "stream_options",
    ),
    "user_param_model_list_gate": (
        {"model": MODEL, "user": "u-1", "messages": [{"role": "user", "content": "x"}]},
        "open_ai_chat_completion_models",
    ),
    "top_k_not_openai": (
        {"model": MODEL, "top_k": 40, "messages": [{"role": "user", "content": "x"}]},
        "top_k",
    ),
    "reasoning_effort_plain_gpt": (
        {
            "model": MODEL,
            "reasoning_effort": "high",
            "messages": [{"role": "user", "content": "x"}],
        },
        "reasoning_effort",
    ),
    "response_format_on_gpt4": (
        {
            "model": "gpt-4",
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": "x"}],
        },
        "outside v1's supported set",
    ),
    "legacy_function_call": (
        {
            "model": MODEL,
            "messages": [
                {"role": "user", "content": "x"},
                {"role": "assistant", "content": "", "function_call": {"name": "f"}},
            ],
        },
        "function_call",
    ),
}


def _v1_body(case: dict) -> dict:
    request = copy.deepcopy(case)
    config = OpenAIGPTConfig()
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


def _v2_body(case: dict):
    return translate_chat_request(
        copy.deepcopy(case), "openai_compat", build_real_deps()
    )


def _norm(body: dict) -> str:
    return json.dumps(body, sort_keys=True, default=str)


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_v2_request_matches_v1(name: str) -> None:
    result = _v2_body(CORPUS[name])
    assert result.is_ok(), result.error.summary
    assert _norm(result.ok) == _norm(_v1_body(CORPUS[name]))


@pytest.mark.parametrize("name", sorted(EXPECTED_FALLBACKS))
def test_unsupported_shape_is_a_typed_fallback(name: str) -> None:
    case, reason_fragment = EXPECTED_FALLBACKS[name]
    result = _v2_body(case)
    assert result.is_error(), f"{name} unexpectedly translated: {result.ok!r}"
    assert reason_fragment in result.error.summary, result.error.summary
