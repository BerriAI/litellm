"""Differential parity for the two azure_ai request routes.

Override set (openai-compat over httpx): v1 is
``AzureAIStudioConfig().map_openai_params`` (= OpenAIConfig's per-family
dispatch, plain GPT here) then ``transform_request`` (extra_body merge +
max_retries pop + ``{model, messages, **optional_params}`` with the
content-list flatten). v2 is ``translate_chat_request(..., "azure_ai")``.
Flatten-prone shapes (text-only content lists), model-map-gated tool_choice,
grok models and the user param are typed fallbacks.

Claude route: v1 is ``AzureAnthropicConfig().map_openai_params`` +
``transform_request`` -- AnthropicConfig's chain with the REAL model name (no
RESPONSE_FORMAT_SPOOF_MODEL: unlike bedrock_invoke/vertex, v1 azure_ai never
spoofs, so response_format picks output_format vs json-tool by the model
itself). v2 is ``translate_chat_request(..., "azure_ai_anthropic")``.
``x-anthropic-billing-header`` system blocks fall back (v1 strips them via
``should_strip_billing_metadata``).
"""

import copy
import json

import pytest

from litellm.llms.azure_ai.anthropic.transformation import AzureAnthropicConfig
from litellm.llms.azure_ai.chat.transformation import AzureAIStudioConfig
from litellm.translation import translate_chat_request

from .conftest import build_real_deps

FOUNDRY_MODEL = "Mistral-large-2407"
CLAUDE_OUTPUT_FORMAT = "claude-sonnet-4-5"  # output_format response strategy
CLAUDE_JSON_TOOL = "claude-3-5-haiku-20241022"  # json_tool_call strategy

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

_JSON_SCHEMA_RF = {
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
}

FOUNDRY_CORPUS = {
    "text": {
        "model": FOUNDRY_MODEL,
        "messages": [{"role": "user", "content": "Hello"}],
    },
    "system_and_sampling": {
        "model": FOUNDRY_MODEL,
        "max_tokens": 50,
        "temperature": 0.5,
        "top_p": 0.9,
        "stop": ["END"],
        "messages": [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ],
    },
    "max_completion_tokens": {
        "model": FOUNDRY_MODEL,
        "max_completion_tokens": 128,
        "messages": [{"role": "user", "content": "hi"}],
    },
    "stream_true": {
        "model": FOUNDRY_MODEL,
        "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
    },
    "tools_without_tool_choice": {
        "model": FOUNDRY_MODEL,
        "tools": [_WEATHER_TOOL],
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
    },
    "tool_call_roundtrip": {
        "model": FOUNDRY_MODEL,
        "tools": [_WEATHER_TOOL],
        "messages": [
            {"role": "user", "content": "Weather in Paris?"},
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
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 20C"},
        ],
    },
    "response_format_json_schema": {
        "model": FOUNDRY_MODEL,
        "response_format": _JSON_SCHEMA_RF,
        "messages": [{"role": "user", "content": "capital of France?"}],
    },
    "image_content_list_not_flattened": {
        # _transform_messages skips the flatten when an image part is present
        "model": FOUNDRY_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                    },
                ],
            }
        ],
    },
}

FOUNDRY_EXPECTED_FALLBACKS = {
    "tool_choice_model_map_gated": (
        {
            "model": FOUNDRY_MODEL,
            "tools": [_WEATHER_TOOL],
            "tool_choice": "auto",
            "messages": [{"role": "user", "content": "x"}],
        },
        "tool_choice on azure_ai is model-map gated",
    ),
    "grok_model": (
        {"model": "grok-3", "messages": [{"role": "user", "content": "x"}]},
        "XAIChatConfig",
    ),
    "text_only_content_list_flatten": (
        {
            "model": FOUNDRY_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "part one"},
                        {"type": "text", "text": "part two"},
                    ],
                }
            ],
        },
        "flattens it to a string",
    ),
    "user_param": (
        {
            "model": FOUNDRY_MODEL,
            "user": "u-1",
            "messages": [{"role": "user", "content": "x"}],
        },
        "user param",
    ),
    "cache_control_forwarded": (
        {
            "model": FOUNDRY_MODEL,
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
            "messages": [{"role": "user", "content": "x"}],
        },
        "cache_control inside tools",
    ),
    "o_series_name": (
        {"model": "o3-mini", "messages": [{"role": "user", "content": "x"}]},
        "OpenAIOSeriesConfig",
    ),
}


def _v1_foundry_body(case: dict) -> dict:
    request = copy.deepcopy(case)
    config = AzureAIStudioConfig()
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


def _v2_body(case: dict, provider: str):
    return translate_chat_request(copy.deepcopy(case), provider, build_real_deps())


def _norm(body: dict) -> str:
    return json.dumps(body, sort_keys=True, default=str)


@pytest.mark.parametrize("name", sorted(FOUNDRY_CORPUS))
def test_v2_foundry_request_matches_v1(name: str) -> None:
    result = _v2_body(FOUNDRY_CORPUS[name], "azure_ai")
    assert result.is_ok(), result.error.summary
    assert _norm(result.ok) == _norm(_v1_foundry_body(FOUNDRY_CORPUS[name]))


@pytest.mark.parametrize("name", sorted(FOUNDRY_EXPECTED_FALLBACKS))
def test_foundry_unsupported_shape_is_a_typed_fallback(name: str) -> None:
    case, reason_fragment = FOUNDRY_EXPECTED_FALLBACKS[name]
    result = _v2_body(case, "azure_ai")
    assert result.is_error(), f"{name} unexpectedly translated: {result.ok!r}"
    assert reason_fragment in result.error.summary, result.error.summary


# ---------------------------------------------------------------------------
# Claude route.
# ---------------------------------------------------------------------------

CLAUDE_CORPUS = {
    "text": {
        "model": CLAUDE_OUTPUT_FORMAT,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Hello"}],
    },
    "system_and_sampling": {
        "model": CLAUDE_OUTPUT_FORMAT,
        "max_tokens": 64,
        "temperature": 0.5,
        "messages": [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ],
    },
    "tools": {
        "model": CLAUDE_OUTPUT_FORMAT,
        "max_tokens": 64,
        "tools": [_WEATHER_TOOL],
        "tool_choice": "auto",
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
    },
    "tool_roundtrip": {
        "model": CLAUDE_OUTPUT_FORMAT,
        "max_tokens": 64,
        "tools": [_WEATHER_TOOL],
        "messages": [
            {"role": "user", "content": "Weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "toolu_01",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"city": "Paris"}),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "toolu_01", "content": "Sunny"},
        ],
    },
    "response_format_output_format_model": {
        # claude-sonnet-4-5: v1 maps response_format onto the native
        # output_format API with the real model name (no spoof)
        "model": CLAUDE_OUTPUT_FORMAT,
        "max_tokens": 64,
        "response_format": _JSON_SCHEMA_RF,
        "messages": [{"role": "user", "content": "capital of France?"}],
    },
    "response_format_json_tool_model": {
        "model": CLAUDE_JSON_TOOL,
        "max_tokens": 64,
        "response_format": _JSON_SCHEMA_RF,
        "messages": [{"role": "user", "content": "capital of France?"}],
    },
    "thinking_enabled": {
        "model": "claude-3-7-sonnet-20250219",
        "max_tokens": 2048,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "messages": [{"role": "user", "content": "Think hard"}],
    },
}

CLAUDE_EXPECTED_FALLBACKS = {
    "billing_header_system_block": (
        {
            "model": CLAUDE_OUTPUT_FORMAT,
            "max_tokens": 64,
            "messages": [
                {"role": "system", "content": "x-anthropic-billing-header: cc-1"},
                {"role": "user", "content": "hi"},
            ],
        },
        "x-anthropic-billing-header",
    ),
    "non_claude_model": (
        {
            "model": FOUNDRY_MODEL,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
        },
        "Claude models only",
    ),
}


def _v1_claude_body(case: dict) -> dict:
    request = copy.deepcopy(case)
    config = AzureAnthropicConfig()
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


@pytest.mark.parametrize("name", sorted(CLAUDE_CORPUS))
def test_v2_claude_request_matches_v1(name: str) -> None:
    result = _v2_body(CLAUDE_CORPUS[name], "azure_ai_anthropic")
    assert result.is_ok(), result.error.summary
    assert _norm(result.ok) == _norm(_v1_claude_body(CLAUDE_CORPUS[name]))


@pytest.mark.parametrize("name", sorted(CLAUDE_EXPECTED_FALLBACKS))
def test_claude_unsupported_shape_is_a_typed_fallback(name: str) -> None:
    case, reason_fragment = CLAUDE_EXPECTED_FALLBACKS[name]
    result = _v2_body(case, "azure_ai_anthropic")
    assert result.is_error(), f"{name} unexpectedly translated: {result.ok!r}"
    assert reason_fragment in result.error.summary, result.error.summary
