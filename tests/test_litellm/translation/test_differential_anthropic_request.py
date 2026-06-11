"""Differential parity: v2 IR translation vs the v1 AnthropicConfig chain.

For each OpenAI chat request in the corpus, the v1 body is produced exactly as
``litellm.completion`` would (``map_openai_params`` then ``transform_request``)
and compared, as normalized JSON, to the v2 body from ``translate_chat_request``.
The corpus is the covered surface: the anthropic v2 flag only turns on for the
shapes pinned here. A divergence on any shape fails the build.

The corpus deliberately spans the four feature surfaces Claude Code lives on
(prompt caching, extended thinking, tool use, structured outputs) on
current-generation model ids (audit F3: the old corpus model was absent from
the model map, hiding the max_tokens default divergence).
"""

import copy
import json

import pytest

from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.translation import translate_chat_request

from .conftest import build_real_deps

LEGACY = "claude-3-5-sonnet-20241022"  # absent from model map: env default path
CURRENT = "claude-sonnet-4-5"  # model-map max_tokens; output_format strategy
REASONING = "claude-3-7-sonnet-20250219"  # budget-token thinking strategy
JSON_TOOL = "claude-3-5-haiku-20241022"  # json_tool_call response_format strategy

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

_PYDANTIC_STYLE_TOOL = {
    "type": "function",
    "function": {
        "name": "report",
        "description": "Submit a report",
        "parameters": {
            "title": "ReportArgs",
            "type": "object",
            "properties": {"body": {"title": "Body", "type": "string"}},
            "required": ["body"],
            "additionalProperties": False,
        },
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
    # ---- the tracer's original shapes (legacy model) ----
    "text": {
        "model": LEGACY,
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Hello, world"}],
    },
    "system_and_sampling": {
        "model": LEGACY,
        "max_tokens": 50,
        "temperature": 0.5,
        "top_p": 0.9,
        "messages": [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ],
    },
    "multiturn_stop_stream": {
        "model": LEGACY,
        "max_tokens": 64,
        "stop": ["END", "STOP"],
        "stream": True,
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ],
    },
    "no_max_tokens_legacy_model": {
        "model": LEGACY,
        "messages": [{"role": "user", "content": "no max tokens here"}],
    },
    "tools_auto": {
        "model": LEGACY,
        "max_tokens": 200,
        "tools": [_WEATHER_TOOL],
        "tool_choice": "auto",
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
    },
    "tool_choice_required": {
        "model": LEGACY,
        "max_tokens": 200,
        "tools": [_WEATHER_TOOL],
        "tool_choice": "required",
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
    },
    "tool_choice_none": {
        "model": LEGACY,
        "max_tokens": 200,
        "tools": [_WEATHER_TOOL],
        "tool_choice": "none",
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
    },
    "tool_choice_specific": {
        "model": LEGACY,
        "max_tokens": 200,
        "tools": [_WEATHER_TOOL],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
    },
    "tool_call_roundtrip": {
        "model": LEGACY,
        "max_tokens": 200,
        "messages": [
            {"role": "user", "content": "Weather in Paris?"},
            _assistant_tool_call("call_1", "Paris"),
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 20C"},
        ],
    },
    "parallel_tool_results_merge": {
        "model": LEGACY,
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
        "model": LEGACY,
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
    "system_as_array": {
        "model": LEGACY,
        "max_tokens": 30,
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are helpful"},
                    {"type": "text", "text": "Be concise"},
                ],
            },
            {"role": "user", "content": "Hi"},
        ],
    },
    "stop_as_string": {
        "model": LEGACY,
        "max_tokens": 40,
        "stop": "STOP",
        "messages": [{"role": "user", "content": "Hello"}],
    },
    "max_completion_tokens": {
        "model": LEGACY,
        "max_completion_tokens": 77,
        "messages": [{"role": "user", "content": "Hello"}],
    },
    "tool_without_description": {
        "model": LEGACY,
        "max_tokens": 50,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "ping",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "messages": [{"role": "user", "content": "ping"}],
    },
    "image_data_uri": {
        "model": LEGACY,
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
    # ---- audit-driven additions ----
    "current_model_default_max_tokens": {
        "model": CURRENT,
        "messages": [{"role": "user", "content": "model-map default, not 4096"}],
    },
    "https_image_url": {
        "model": CURRENT,
        "max_tokens": 64,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "see"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/cat.png"},
                    },
                ],
            }
        ],
    },
    "image_format_override": {
        "model": CURRENT,
        "max_tokens": 64,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:application/octet-stream;base64,QUJD",
                            "format": "image/webp",
                        },
                    }
                ],
            }
        ],
    },
    "cache_control_everywhere": {
        "model": CURRENT,
        "max_tokens": 64,
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "rules",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "hi",
                        "cache_control": {"type": "ephemeral", "ttl": "5m"},
                    }
                ],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "parameters": {"type": "object", "properties": {}},
                },
                "cache_control": {"type": "ephemeral"},
            }
        ],
    },
    "cache_control_on_string_message": {
        "model": CURRENT,
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "sys", "cache_control": {"type": "ephemeral"}},
            {"role": "user", "content": "hi", "cache_control": {"type": "ephemeral"}},
        ],
    },
    "cache_control_on_tool_result": {
        "model": CURRENT,
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": "go"},
            _assistant_tool_call("call_1", "Paris"),
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": [
                    {
                        "type": "text",
                        "text": "Sunny",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
        ],
    },
    "reasoning_effort_low": {
        "model": REASONING,
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": "think"}],
        "reasoning_effort": "low",
    },
    "reasoning_effort_high_no_max_tokens": {
        "model": REASONING,
        "messages": [{"role": "user", "content": "think hard"}],
        "reasoning_effort": "high",
    },
    "reasoning_effort_none": {
        "model": REASONING,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "no think"}],
        "reasoning_effort": "none",
    },
    "thinking_explicit": {
        "model": REASONING,
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": "think"}],
        "thinking": {"type": "enabled", "budget_tokens": 1024},
    },
    "thinking_no_max_tokens_bumps": {
        "model": REASONING,
        "messages": [{"role": "user", "content": "think"}],
        "thinking": {"type": "enabled", "budget_tokens": 1024},
    },
    "thinking_history_blocks": {
        "model": REASONING,
        "max_tokens": 2048,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "messages": [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": None,
                "thinking_blocks": [
                    {
                        "type": "thinking",
                        "thinking": "I should call the tool",
                        "signature": "sig==",
                    }
                ],
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
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"},
        ],
        "tools": [_WEATHER_TOOL],
    },
    "response_format_json_schema_current": {
        "model": CURRENT,
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "capital of France?"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {
                    "title": "Answer",
                    "type": "object",
                    "properties": {
                        "capital": {"type": "string", "minLength": 2, "title": "Capital"}
                    },
                    "required": ["capital"],
                },
            },
        },
    },
    "response_format_json_object_current": {
        "model": CURRENT,
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "json please"}],
        "response_format": {"type": "json_object"},
    },
    "response_format_json_object_legacy_noop": {
        "model": JSON_TOOL,
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "json please"}],
        "response_format": {"type": "json_object"},
    },
    "response_format_json_schema_legacy_tool": {
        "model": JSON_TOOL,
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "capital of France?"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {
                    "type": "object",
                    "properties": {"capital": {"type": "string"}},
                    "required": ["capital"],
                },
            },
        },
    },
    "response_format_with_thinking_no_forced_choice": {
        "model": REASONING,
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": "json"}],
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "a",
                "schema": {"type": "object", "properties": {"x": {"type": "string"}}},
            },
        },
    },
    "user_metadata": {
        "model": CURRENT,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
        "user": "customer-123",
    },
    "user_email_skipped": {
        "model": CURRENT,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
        "user": "someone@example.com",
    },
    "parallel_tool_calls_false": {
        "model": CURRENT,
        "max_tokens": 64,
        "tools": [_WEATHER_TOOL],
        "parallel_tool_calls": False,
        "messages": [{"role": "user", "content": "Weather in Paris and Rome?"}],
    },
    "parallel_with_tool_choice": {
        "model": CURRENT,
        "max_tokens": 64,
        "tools": [_WEATHER_TOOL],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "messages": [{"role": "user", "content": "weather"}],
    },
    "parallel_with_string_none_choice": {
        "model": CURRENT,
        "max_tokens": 64,
        "tools": [_WEATHER_TOOL],
        "tool_choice": "none",
        "parallel_tool_calls": False,
        "messages": [{"role": "user", "content": "weather"}],
    },
    "tool_choice_dict_forms": {
        "model": CURRENT,
        "max_tokens": 64,
        "tools": [_WEATHER_TOOL],
        "tool_choice": {"type": "any"},
        "messages": [{"role": "user", "content": "weather"}],
    },
    "top_k": {
        "model": CURRENT,
        "max_tokens": 64,
        "top_k": 5,
        "messages": [{"role": "user", "content": "hi"}],
    },
    "temperature_int_stays_int": {
        "model": CURRENT,
        "max_tokens": 64,
        "temperature": 1,
        "messages": [{"role": "user", "content": "hi"}],
    },
    "tool_name_sanitization_with_history": {
        "model": CURRENT,
        "max_tokens": 64,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "mcp.server/get_weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "mcp_server_get_weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ],
        "tool_choice": {
            "type": "function",
            "function": {"name": "mcp.server/get_weather"},
        },
        "messages": [
            {"role": "user", "content": "go"},
            _assistant_tool_call("call.1", "Paris", name="mcp.server/get_weather"),
            {"role": "tool", "tool_call_id": "call.1", "content": "Sunny"},
        ],
    },
    "pydantic_style_schema_filtered": {
        "model": CURRENT,
        "max_tokens": 64,
        "tools": [_PYDANTIC_STYLE_TOOL],
        "messages": [{"role": "user", "content": "report"}],
    },
    "tool_schema_type_coerced": {
        "model": CURRENT,
        "max_tokens": 64,
        "tools": [
            {
                "type": "function",
                "function": {"name": "f", "parameters": {"properties": {}}},
            }
        ],
        "messages": [{"role": "user", "content": "hi"}],
    },
    "tool_schema_missing_parameters": {
        "model": CURRENT,
        "max_tokens": 64,
        "tools": [{"type": "function", "function": {"name": "f"}}],
        "messages": [{"role": "user", "content": "hi"}],
    },
    "empty_user_content_placeholder": {
        "model": CURRENT,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": ""}],
    },
    "whitespace_text_part_placeholder": {
        "model": CURRENT,
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "   "}]},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "continue"},
        ],
    },
    "tool_result_parts_not_placeholdered": {
        "model": CURRENT,
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": "go"},
            _assistant_tool_call("call_1", "Paris"),
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": [{"type": "text", "text": "ok"}],
            },
        ],
    },
    "duplicate_tool_call_ids_dedupe": {
        "model": CURRENT,
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": "go"},
            _assistant_tool_call("call_1", "Paris"),
            _assistant_tool_call("call_1", "Paris"),
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"},
        ],
    },
    "tool_use_id_sanitized": {
        "model": CURRENT,
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": "go"},
            _assistant_tool_call("call/1:x", "Paris"),
            {"role": "tool", "tool_call_id": "call/1:x", "content": "Sunny"},
        ],
    },
    "final_assistant_text_rstripped": {
        "model": CURRENT,
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "prefill...   "},
        ],
    },
    "stop_whitespace_kept_without_drop_params": {
        "model": CURRENT,
        "max_tokens": 64,
        "stop": ["END", "  "],
        "messages": [{"role": "user", "content": "hi"}],
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
    if "top_k" in params:
        # Provider-specific kwargs bypass map_openai_params in production:
        # get_optional_params copies them into optional_params verbatim.
        optional = {**optional, "top_k": params["top_k"]}
    return config.transform_request(
        model, copy.deepcopy(request["messages"]), optional, {}, {}
    )


def _v2_body(request: dict) -> dict:
    result = translate_chat_request(
        copy.deepcopy(request), "anthropic", build_real_deps()
    )
    assert result.is_ok(), result.error.summary
    return result.ok


def _norm(body: dict) -> str:
    return json.dumps(body, sort_keys=True, default=str)


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_v2_matches_v1(name: str) -> None:
    request = CORPUS[name]
    assert _norm(_v2_body(request)) == _norm(_v1_body(request))


def test_gated_sampling_falls_back_where_v1_raises() -> None:
    """opus-4.7-class models reject sampling params: v1 raises a client error,
    v2 returns unsupported so the seam lands on the same v1 behavior."""
    request = {
        "model": "claude-opus-4-7",
        "max_tokens": 64,
        "temperature": 0.2,
        "messages": [{"role": "user", "content": "hi"}],
    }
    result = translate_chat_request(request, "anthropic", build_real_deps())
    assert result.is_error()
    assert result.error.tag == "unsupported"
    with pytest.raises(Exception):
        _v1_body(request)


def test_drop_params_drops_gated_sampling_like_v1(monkeypatch) -> None:
    import litellm

    request = {
        "model": "claude-opus-4-7",
        "max_tokens": 64,
        "temperature": 0.2,
        "top_p": 0.5,
        "messages": [{"role": "user", "content": "hi"}],
    }
    deps = build_real_deps(drop_params=True, drop_params_global=True)
    result = translate_chat_request(copy.deepcopy(request), "anthropic", deps)
    assert result.is_ok(), result.error.summary if result.is_error() else None

    monkeypatch.setattr(litellm, "drop_params", True)
    config = AnthropicConfig()
    params = {k: v for k, v in request.items() if k not in ("model", "messages")}
    optional = config.map_openai_params(
        copy.deepcopy(params), {}, request["model"], drop_params=True
    )
    v1 = config.transform_request(
        request["model"], copy.deepcopy(request["messages"]), optional, {}, {}
    )
    assert _norm(result.ok) == _norm(v1)
