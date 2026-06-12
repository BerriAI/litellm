"""Differential parity for the google request transforms.

Two-sided gate over the characterization corpus (cases copied verbatim from
mateo/translation-characterization-providers):

1. v1-at-HEAD must still equal the committed snapshot (drift guard);
2. v2 must equal the snapshot BYTE-FOR-BYTE (canonical JSON).

A quirk corpus pins the drift list and the 3-way structured-output fork
against v1 in-process: responseJsonSchema (2.x regex) vs responseSchema +
propertyOrdering (model-map capability) vs schema-as-user-message, AI-Studio
top_k dropping, gemini-3 default temperature / thinkingLevel / function-call
id forwarding, multi-message systems, and the function_response name
recovery. Shapes outside the proven surface must return a typed error (the
seam falls back to v1), never a divergent body.
"""

import copy
import json

import pytest

from litellm.translation import translate_chat_request
from litellm.translation_seam_google import build_google_deps

from . import _google_corpus as corpus

CASES = corpus.cases()

EXPECTED_FALLBACKS = {
    "pdf_base64": "file/document parts are outside the v2 inbound surface",
}


def _v2_raw(provider_key: str, case: dict) -> dict:
    model, _, _ = corpus.resolve(provider_key)
    return {
        "model": model,
        "messages": copy.deepcopy(case["messages"]),
        **copy.deepcopy(case["params"]),
    }


def _v2_translate(provider_key: str, raw: dict, drop_params: bool = False):
    v2_provider = corpus.V2_PROVIDERS[provider_key]
    deps = build_google_deps(v2_provider, request_drop_params=drop_params)
    return translate_chat_request(raw, v2_provider, deps)


@pytest.mark.parametrize("case_id", sorted(CASES))
@pytest.mark.parametrize("provider_key", sorted(corpus.PROVIDERS))
def test_v1_still_matches_snapshot(
    provider_key: str, case_id: str, vertex_token_stub
) -> None:
    case = CASES[case_id]
    if provider_key in case["skip"]:
        pytest.skip(case["skip"][provider_key])
    snapshot = corpus.SNAPSHOTS_DIR / "requests" / provider_key / f"{case_id}.json"
    body = corpus.run_v1_request_transform(provider_key, case)
    assert corpus.canonical_json(body) == snapshot.read_text(), (
        f"v1 drifted from the characterization snapshot for {case_id}; "
        "regenerate the corpus and ship the diff as its own PR"
    )


@pytest.mark.parametrize("case_id", sorted(CASES))
@pytest.mark.parametrize("provider_key", sorted(corpus.PROVIDERS))
def test_v2_matches_snapshot_or_falls_back(provider_key: str, case_id: str) -> None:
    case = CASES[case_id]
    if provider_key in case["skip"]:
        pytest.skip(case["skip"][provider_key])
    result = _v2_translate(provider_key, _v2_raw(provider_key, case))
    if case_id in EXPECTED_FALLBACKS:
        assert result.is_error(), EXPECTED_FALLBACKS[case_id]
        return
    assert result.is_ok(), result.error.summary
    snapshot = corpus.SNAPSHOTS_DIR / "requests" / provider_key / f"{case_id}.json"
    assert corpus.canonical_json(result.ok) == snapshot.read_text()


# ---------------------------------------------------------------------------
# google-only quirk corpus: reference is v1 in-process (the same invocation
# as the characterization seam), asserted JSON-equal.
# ---------------------------------------------------------------------------

_USER = {"role": "user", "content": "What is the capital of France?"}

_JSON_SCHEMA_RF = {
    "type": "json_schema",
    "json_schema": {
        "name": "capital",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"capital": {"type": "string"}},
            "required": ["capital"],
            "additionalProperties": False,
        },
    },
}

_TOOL_HISTORY = [
    {"role": "user", "content": "Weather in Paris?"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_g3_001",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"city": "Paris"}),
                },
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_g3_001", "content": "18C"},
]

_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

# (alias, case, drop_params) quirks; every row references v1 in-process.
QUIRKS = {
    "studio_response_schema_property_ordering": (
        # supports_response_schema=True but fails the 2.x regex -> the
        # responseSchema + propertyOrdering tier of the 3-way fork.
        "gemini/gemini-exp-1206",
        {"messages": [_USER], "params": {"response_format": _JSON_SCHEMA_RF}},
        False,
    ),
    "vertex_schema_prompt_for_unsupported_capability": (
        "vertex_ai/gemini-pro-latest",
        {"messages": [_USER], "params": {"response_format": _JSON_SCHEMA_RF}},
        False,
    ),
    "studio_schema_prompt_for_unsupported_model": (
        "gemini/gemini-1.5-flash",
        {"messages": [_USER], "params": {"response_format": _JSON_SCHEMA_RF}},
        False,
    ),
    "studio_top_k_passthrough": (
        # top_k is not an OpenAI param; v1 forwards it on BOTH google routes.
        "gemini/gemini-2.5-flash",
        {"messages": [_USER], "params": {"max_tokens": 64, "top_k": 5}},
        False,
    ),
    "vertex_top_k_passthrough": (
        "vertex_ai/gemini-2.5-pro",
        {"messages": [_USER], "params": {"max_tokens": 64, "top_k": 5}},
        False,
    ),
    "multi_system_messages_two_parts": (
        "vertex_ai/gemini-2.5-pro",
        {
            "messages": [
                {"role": "system", "content": "You are terse."},
                {"role": "system", "content": "Answer in French."},
                _USER,
            ],
            "params": {"max_tokens": 64},
        },
        False,
    ),
    "system_only_blank_user_message": (
        "vertex_ai/gemini-2.5-pro",
        {
            "messages": [{"role": "system", "content": "You are terse."}],
            "params": {},
        },
        False,
    ),
    "stop_as_string": (
        "vertex_ai/gemini-2.5-pro",
        {"messages": [_USER], "params": {"stop": "END", "max_tokens": 32}},
        False,
    ),
    "tool_choice_none_mode": (
        "vertex_ai/gemini-2.5-pro",
        {
            "messages": [_USER],
            "params": {"tools": [copy.deepcopy(_WEATHER_TOOL)], "tool_choice": "none"},
        },
        False,
    ),
    "tool_without_parameters": (
        "vertex_ai/gemini-2.5-pro",
        {
            "messages": [_USER],
            "params": {
                "tools": [{"type": "function", "function": {"name": "ping"}}]
            },
        },
        False,
    ),
    "parallel_tool_calls_never_reaches_wire": (
        "vertex_ai/gemini-2.5-pro",
        {
            "messages": [_USER],
            "params": {
                "tools": [copy.deepcopy(_WEATHER_TOOL)],
                "parallel_tool_calls": False,
            },
        },
        False,
    ),
    "reasoning_effort_minimal_model_budget": (
        "vertex_ai/gemini-2.5-pro",
        {"messages": [_USER], "params": {"reasoning_effort": "minimal"}},
        False,
    ),
    "thinking_budget_zero": (
        "vertex_ai/gemini-2.5-pro",
        {
            "messages": [_USER],
            "params": {"thinking": {"type": "enabled", "budget_tokens": 0}},
        },
        False,
    ),
    "image_url_format_override": (
        "vertex_ai/gemini-2.5-pro",
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Look."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://example.com/cat.png",
                                "format": "image/webp",
                            },
                        },
                    ],
                }
            ],
            "params": {"max_tokens": 32},
        },
        False,
    ),
    "gemini3_default_temperature_and_level": (
        "vertex_ai/gemini-3-pro-preview",
        {"messages": [_USER], "params": {"reasoning_effort": "low"}},
        False,
    ),
    "gemini3_studio_forwards_function_call_ids": (
        "gemini/gemini-3-pro-preview",
        {"messages": copy.deepcopy(_TOOL_HISTORY), "params": {"max_tokens": 64}},
        False,
    ),
}


@pytest.mark.parametrize("name", sorted(QUIRKS))
def test_quirks_match_v1(name: str, vertex_token_stub) -> None:
    alias, case, drop_params = QUIRKS[name]
    v1 = corpus.run_v1_request_transform_for_model(
        alias, copy.deepcopy(case), drop_params=drop_params
    )
    model, custom_llm_provider, _ = corpus.resolve_model(alias)
    raw = {
        "model": model,
        "messages": copy.deepcopy(case["messages"]),
        **copy.deepcopy(case["params"]),
    }
    provider_key = {"vertex_ai": "vertex_gemini", "gemini": "gemini"}[
        custom_llm_provider
    ]
    result = _v2_translate(provider_key, raw, drop_params=drop_params)
    assert result.is_ok(), result.error.summary
    assert corpus.canonical_json(result.ok) == corpus.canonical_json(v1)


def test_studio_https_image_falls_back() -> None:
    raw = {
        "model": "gemini-2.5-flash",
        "max_tokens": 64,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/cat.png"},
                    },
                ],
            }
        ],
    }
    result = _v2_translate("gemini", raw)
    assert result.is_error()
    assert result.error.tag == "unsupported"


def test_large_cache_marker_falls_back() -> None:
    raw = {
        "model": "gemini-2.5-pro",
        "max_tokens": 64,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "x" * 5000,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ],
    }
    result = _v2_translate("vertex_gemini", raw)
    assert result.is_error()
    assert result.error.tag == "unsupported"


def test_reasoning_effort_xhigh_falls_back() -> None:
    raw = {
        "model": "gemini-2.5-pro",
        "reasoning_effort": "xhigh",
        "messages": [_USER],
    }
    result = _v2_translate("vertex_gemini", raw)
    assert result.is_error()
    assert result.error.tag == "unsupported"


def test_vertex_anthropic_response_format_with_thinking_falls_back() -> None:
    raw = {
        "model": "claude-sonnet-4@20250514",
        "max_tokens": 2048,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "response_format": {"type": "json_object"},
        "messages": [_USER],
    }
    result = _v2_translate("vertex_anthropic", raw)
    assert result.is_error()
    assert result.error.tag == "unsupported"
