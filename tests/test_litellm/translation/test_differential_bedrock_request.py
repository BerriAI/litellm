"""Differential parity for the bedrock request transforms.

Two-sided gate over the characterization corpus (cases copied verbatim from
mateo/translation-characterization):

1. v1-at-HEAD must still equal the committed snapshot (drift guard: the
   corpus stays an honest reference);
2. v2 must equal the snapshot BYTE-FOR-BYTE (canonical JSON), which makes
   v2 == v1 transitively.

Shapes outside the proven surface must return a typed error (the seam then
falls back to v1), never a divergent body. A second corpus of bedrock-only
quirk shapes (4.5 cache ttl, parallel tool config, name normalization,
tool_choice gates, top_k) runs v1 in-process as its own reference.
"""

import copy
import json

import pytest

from litellm.translation import translate_chat_request

from . import _bedrock_corpus as corpus
from .conftest import build_real_deps

CASES = corpus.cases()

# Shapes the corpus skips per provider (v1 downloads URL media: network
# inside the transform) or that v2 deliberately routes back to v1.
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


@pytest.mark.parametrize("case_id", sorted(CASES))
@pytest.mark.parametrize("provider_key", sorted(corpus.PROVIDERS))
def test_v1_still_matches_snapshot(provider_key: str, case_id: str) -> None:
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
    result = translate_chat_request(
        _v2_raw(provider_key, case), provider_key, build_real_deps()
    )
    if case_id in EXPECTED_FALLBACKS:
        assert result.is_error(), EXPECTED_FALLBACKS[case_id]
        return
    assert result.is_ok(), result.error.summary
    snapshot = corpus.SNAPSHOTS_DIR / "requests" / provider_key / f"{case_id}.json"
    assert corpus.canonical_json(result.ok) == snapshot.read_text()


# ---------------------------------------------------------------------------
# bedrock-only quirk corpus: reference is v1 in-process (same invocation as
# the characterization seam), asserted JSON-equal.
# ---------------------------------------------------------------------------

_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

_USER = {"role": "user", "content": "What is the capital of France?"}

QUIRKS = {
    "no_max_tokens_no_default": {"messages": [_USER], "params": {}},
    "stop_whitespace_kept": {
        "messages": [_USER],
        "params": {"max_tokens": 64, "stop": ["END", "  "]},
    },
    "top_k_additional_field": {
        "messages": [_USER],
        "params": {"max_tokens": 64, "top_k": 5},
    },
    "tool_name_normalized": {
        "messages": [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call/1:x",
                        "type": "function",
                        "function": {
                            "name": "mcp.server/get_weather",
                            "arguments": json.dumps({"city": "Paris"}),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call/1:x", "content": "Sunny"},
        ],
        "params": {
            "max_tokens": 64,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "mcp.server/get_weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": "mcp.server/get_weather"},
            },
        },
    },
    "thinking_budget_clamped_to_bedrock_min": {
        "messages": [_USER],
        "params": {"thinking": {"type": "enabled", "budget_tokens": 256}},
    },
    "thinking_rewrites_forced_choice_to_auto": {
        "messages": [_USER],
        "params": {
            "max_tokens": 2048,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "tools": [_WEATHER_TOOL],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        },
    },
}

# converse-only: the invoke route deliberately falls back for the
# response_format x reasoning model-spoof crossing.
QUIRKS_CONVERSE_ONLY = {
    "json_schema_with_effort_drops_forced_choice": {
        "messages": [_USER],
        "params": {
            "max_tokens": 2048,
            "reasoning_effort": "low",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "capital",
                    "schema": {
                        "type": "object",
                        "properties": {"capital": {"type": "string"}},
                    },
                },
            },
        },
    },
    "assistant_blank_text_dropped": {
        "messages": [
            _USER,
            {"role": "assistant", "content": "   "},
            {"role": "user", "content": "And of Italy?"},
        ],
        "params": {"max_tokens": 64},
    },
}

_CLAUDE_45 = "bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0"

QUIRKS_45 = {
    "cache_ttl_kept_on_claude_4_5": {
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "rules",
                        "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    }
                ],
            },
            _USER,
        ],
        "params": {
            "max_tokens": 64,
            "tools": [
                {**_WEATHER_TOOL, "cache_control": {"type": "ephemeral", "ttl": "5m"}}
            ],
        },
    },
    "parallel_tool_config_on_claude_4_5": {
        "messages": [_USER],
        "params": {
            "max_tokens": 64,
            "tools": [_WEATHER_TOOL],
            "parallel_tool_calls": False,
        },
    },
}


@pytest.mark.parametrize("name", sorted(QUIRKS))
@pytest.mark.parametrize("provider_key", sorted(corpus.PROVIDERS))
def test_quirks_match_v1(provider_key: str, name: str) -> None:
    case = QUIRKS[name]
    v1 = corpus.run_v1_request_transform(provider_key, case)
    result = translate_chat_request(
        _v2_raw(provider_key, case), provider_key, build_real_deps()
    )
    assert result.is_ok(), result.error.summary
    assert corpus.canonical_json(result.ok) == corpus.canonical_json(v1)


@pytest.mark.parametrize("name", sorted(QUIRKS_CONVERSE_ONLY))
def test_converse_only_quirks_match_v1(name: str) -> None:
    case = QUIRKS_CONVERSE_ONLY[name]
    v1 = corpus.run_v1_request_transform("bedrock_converse", case)
    result = translate_chat_request(
        _v2_raw("bedrock_converse", case), "bedrock_converse", build_real_deps()
    )
    assert result.is_ok(), result.error.summary
    assert corpus.canonical_json(result.ok) == corpus.canonical_json(v1)


def test_invoke_response_format_with_reasoning_falls_back() -> None:
    case = QUIRKS_CONVERSE_ONLY["json_schema_with_effort_drops_forced_choice"]
    result = translate_chat_request(
        _v2_raw("bedrock_invoke", case), "bedrock_invoke", build_real_deps()
    )
    assert result.is_error()
    assert result.error.tag == "unsupported"


@pytest.mark.parametrize("name", sorted(QUIRKS_45))
def test_claude_4_5_quirks_match_v1_on_converse(name: str) -> None:
    case = QUIRKS_45[name]
    v1 = corpus.run_v1_request_transform_for_model(_CLAUDE_45, case)
    model, _, _ = corpus.resolve_model(_CLAUDE_45)
    raw = {
        "model": model,
        "messages": copy.deepcopy(case["messages"]),
        **copy.deepcopy(case["params"]),
    }
    result = translate_chat_request(raw, "bedrock_converse", build_real_deps())
    assert result.is_ok(), result.error.summary
    assert corpus.canonical_json(result.ok) == corpus.canonical_json(v1)


def test_tool_choice_none_falls_back_where_v1_raises() -> None:
    raw = {
        "model": "anthropic.claude-sonnet-4-20250514-v1:0",
        "max_tokens": 64,
        "tools": [_WEATHER_TOOL],
        "tool_choice": "none",
        "messages": [_USER],
    }
    result = translate_chat_request(raw, "bedrock_converse", build_real_deps())
    assert result.is_error()
    assert result.error.tag == "unsupported"


def test_tool_choice_none_dropped_with_drop_params() -> None:
    case = {
        "messages": [_USER],
        "params": {"max_tokens": 64, "tools": [_WEATHER_TOOL], "tool_choice": "none"},
    }
    deps = build_real_deps(drop_params=True)
    raw = {
        "model": "anthropic.claude-sonnet-4-20250514-v1:0",
        "messages": copy.deepcopy(case["messages"]),
        **copy.deepcopy(case["params"]),
    }
    result = translate_chat_request(raw, "bedrock_converse", deps)
    assert result.is_ok(), result.error.summary
    v1 = corpus.run_v1_request_transform_for_model(
        corpus.PROVIDERS["bedrock_converse"], case, drop_params=True
    )
    assert corpus.canonical_json(result.ok) == corpus.canonical_json(v1)


def test_tool_history_without_tools_falls_back_where_v1_raises() -> None:
    raw = {
        "model": "anthropic.claude-sonnet-4-20250514-v1:0",
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        ],
    }
    result = translate_chat_request(raw, "bedrock_converse", build_real_deps())
    assert result.is_error()
    assert result.error.tag == "unsupported"


def test_non_claude_bedrock_model_falls_back() -> None:
    raw = {
        "model": "amazon.nova-pro-v1:0",
        "max_tokens": 64,
        "messages": [_USER],
    }
    for provider_key in ("bedrock_converse", "bedrock_invoke"):
        result = translate_chat_request(raw, provider_key, build_real_deps())
        assert result.is_error()
        assert result.error.tag == "unsupported"
