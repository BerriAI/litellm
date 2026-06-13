"""Differential parity for the hosted_vllm request path (httpx dedicated
elif, main.py:2619; transform LIVE; no model prefix).

Two-sided: every served row is byte-identical (normalized JSON) between v1
in-process at HEAD (get_optional_params + HostedVLLMChatConfig
.transform_request, the _own_module_corpus invoker) and v2; every fallback
row asserts BOTH the typed v2 error and v1's own behavior (serve or
rewrite, asserted in-process). The thinking budget bands and the recursive
tools cleaning are v1 facts re-verified here, not dossier trust.
"""

import copy
import json

import pytest

from litellm.translation.dispatch import NEVER_PORT
from litellm.translation.engine import pipeline
from litellm.translation.engine.pipeline import translate_chat_request
from litellm.utils import get_optional_params

from ._own_module_corpus import capture_v1_wire_body, run_v1_request_transform
from .conftest import build_real_deps

MODEL = "qwen2.5-7b-instruct"
_USER = [{"role": "user", "content": "Hello, world"}]

DIRTY_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather",
        "strict": True,
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "city": {"type": "string"},
                "nested": {
                    "type": "object",
                    "additionalProperties": False,
                    "strict": True,
                    "properties": {"zip": {"type": "string"}},
                },
            },
            "required": ["city"],
        },
    },
}

CASES: dict[str, dict] = {
    "text": {"model": MODEL, "messages": _USER},
    "system_and_sampling": {
        "model": MODEL,
        "max_tokens": 64,
        "temperature": 0.5,
        "top_p": 0.9,
        "messages": [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ],
    },
    "stream_true": {"model": MODEL, "stream": True, "messages": _USER},
    "stop_list": {"model": MODEL, "stop": ["END"], "messages": _USER},
    "max_completion_tokens_verbatim": {
        "model": MODEL,
        "max_completion_tokens": 128,
        "messages": _USER,
    },
    "temperature_int_stays_int": {"model": MODEL, "temperature": 1, "messages": _USER},
    # the recursive cleaning: strict (any depth, any value) and
    # additionalProperties:false (any depth) removed
    "tools_cleaned_recursively": {
        "model": MODEL,
        "tools": [DIRTY_TOOL],
        "tool_choice": "auto",
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
    },
    "tool_choice_specific": {
        "model": MODEL,
        "tools": [DIRTY_TOOL],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
    },
    "tool_call_compact_roundtrip": {
        "model": MODEL,
        "tools": [DIRTY_TOOL],
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
    "parallel_tool_calls_false": {
        "model": MODEL,
        "tools": [DIRTY_TOOL],
        "parallel_tool_calls": False,
        "messages": [{"role": "user", "content": "Weather?"}],
    },
    "response_format_json_object": {
        "model": MODEL,
        "response_format": {"type": "json_object"},
        "messages": _USER,
    },
    "response_format_json_schema_strict": {
        "model": MODEL,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "a", "schema": {"type": "object"}, "strict": True},
        },
        "messages": _USER,
    },
    # reasoning_effort is UNCONDITIONALLY supported, emitted verbatim
    "reasoning_effort_verbatim": {
        "model": MODEL,
        "reasoning_effort": "high",
        "messages": _USER,
    },
    # the budget-band rewrite (map_openai_params; in-process re-verified):
    "thinking_band_high": {
        "model": MODEL,
        "thinking": {"type": "enabled", "budget_tokens": 12000},
        "messages": _USER,
    },
    "thinking_band_medium": {
        "model": MODEL,
        "thinking": {"type": "enabled", "budget_tokens": 6000},
        "messages": _USER,
    },
    "thinking_band_low": {
        "model": MODEL,
        "thinking": {"type": "enabled", "budget_tokens": 3000},
        "messages": _USER,
    },
    "thinking_band_minimal_no_budget": {
        "model": MODEL,
        "thinking": {"type": "enabled"},
        "messages": _USER,
    },
    # the EXACT >= boundaries (verifier-wave2b-alpha F4: the interior-point
    # rows above let a >= -> > band mutation survive; these three kill it —
    # 10000 -> high, 5000 -> medium, 2000 -> low on BOTH sides)
    "thinking_band_high_exact_boundary": {
        "model": MODEL,
        "thinking": {"type": "enabled", "budget_tokens": 10000},
        "messages": _USER,
    },
    "thinking_band_medium_exact_boundary": {
        "model": MODEL,
        "thinking": {"type": "enabled", "budget_tokens": 5000},
        "messages": _USER,
    },
    "thinking_band_low_exact_boundary": {
        "model": MODEL,
        "thinking": {"type": "enabled", "budget_tokens": 2000},
        "messages": _USER,
    },
    "thinking_disabled_dropped": {
        "model": MODEL,
        "thinking": {"type": "disabled"},
        "messages": _USER,
    },
    "thinking_adaptive_dropped": {
        "model": MODEL,
        "thinking": {"type": "adaptive"},
        "messages": _USER,
    },
    # explicit reasoning_effort WINS over the thinking rewrite
    "reasoning_effort_wins_over_thinking": {
        "model": MODEL,
        "thinking": {"type": "enabled", "budget_tokens": 12000},
        "reasoning_effort": "low",
        "messages": _USER,
    },
}

# name -> (case, v2 reason fragment); generator emits FALLBACK (v1 serves it)
EXPECTED_FALLBACKS: dict[str, tuple[dict, str]] = {
    # v1 synthesizes a function tool from a custom-type tool; the shared
    # openai guard falls back
    "custom_tool": (
        {
            "model": MODEL,
            "messages": _USER,
            "tools": [
                {
                    "type": "custom",
                    "custom": {
                        "name": "ct",
                        "description": "d",
                        "input_schema": {"type": "object", "properties": {}},
                    },
                }
            ],
        },
        "custom-type tool",
    ),
    # v1 prepends assistant thinking_blocks as content blocks; the shared
    # openai guard falls back
    "assistant_thinking_blocks": (
        {
            "model": MODEL,
            "messages": [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "ans",
                    "thinking_blocks": [
                        {"type": "thinking", "thinking": "hmm", "signature": "s"}
                    ],
                },
                {"role": "user", "content": "n"},
            ],
        },
        "thinking_blocks",
    ),
    # v1 converts video file parts to video_url blocks; the inbound boundary
    # falls back on every file part
    "video_file_part": (
        {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "watch"},
                        {"type": "file", "file": {"file_id": "https://x/v.mp4"}},
                    ],
                }
            ],
        },
        "not yet supported",
    ),
    "top_k": ({"model": MODEL, "top_k": 5, "messages": _USER}, "extra_body"),
    "user": ({"model": MODEL, "user": "u1", "messages": _USER}, "user"),
    "stream_false": ({"model": MODEL, "stream": False, "messages": _USER}, "stream"),
}


def _v2(case: dict):
    return translate_chat_request(copy.deepcopy(case), "hosted_vllm", build_real_deps())


def _norm(body: dict) -> str:
    return json.dumps(body, sort_keys=True, default=str)


@pytest.mark.parametrize("name", sorted(CASES))
def test_v2_request_matches_v1(name: str) -> None:
    case = CASES[name]
    result = _v2(case)
    assert result.is_ok(), result.error.summary
    assert _norm(result.ok) == _norm(run_v1_request_transform("hosted_vllm", case))


@pytest.mark.parametrize("name", sorted(EXPECTED_FALLBACKS))
def test_expected_fallbacks_are_typed(name: str) -> None:
    case, reason = EXPECTED_FALLBACKS[name]
    result = _v2(case)
    assert result.is_error(), f"{name} unexpectedly served"
    assert reason in result.error.summary, result.error.summary


def test_thinking_bands_semantic_pins() -> None:
    """The wire reasoning_effort value per band, the disabled/adaptive
    drops, and the thinking key never reaching the wire."""
    expectations = {
        "thinking_band_high": "high",
        "thinking_band_medium": "medium",
        "thinking_band_low": "low",
        "thinking_band_minimal_no_budget": "minimal",
        "reasoning_effort_wins_over_thinking": "low",
    }
    for name, effort in expectations.items():
        result = _v2(CASES[name])
        assert result.is_ok(), (name, result.error.summary)
        assert result.ok["reasoning_effort"] == effort, name
        assert "thinking" not in result.ok, name
    for name in ("thinking_disabled_dropped", "thinking_adaptive_dropped"):
        result = _v2(CASES[name])
        assert result.is_ok()
        assert "reasoning_effort" not in result.ok, name
        assert "thinking" not in result.ok, name


def test_tools_cleaning_semantic_pins() -> None:
    """strict gone at every depth; additionalProperties:false gone at every
    depth; everything else intact."""
    result = _v2(CASES["tools_cleaned_recursively"])
    assert result.is_ok()
    dumped = json.dumps(result.ok["tools"])
    assert "strict" not in dumped
    assert "additionalProperties" not in dumped
    assert result.ok["tools"][0]["function"]["parameters"]["properties"]["nested"][
        "properties"
    ] == {"zip": {"type": "string"}}


def test_custom_tool_v1_serves_its_conversion() -> None:
    case, _ = EXPECTED_FALLBACKS["custom_tool"]
    v1 = run_v1_request_transform("hosted_vllm", case)
    assert v1["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "ct",
                "parameters": {"type": "object", "properties": {}},
                "description": "d",
            },
        }
    ]


def test_thinking_blocks_v1_serves_its_prepend() -> None:
    case, _ = EXPECTED_FALLBACKS["assistant_thinking_blocks"]
    v1 = run_v1_request_transform("hosted_vllm", case)
    assert v1["messages"][1]["content"] == [
        {"type": "thinking", "thinking": "hmm"},
        {"type": "text", "text": "ans"},
    ]
    assert "thinking_blocks" not in v1["messages"][1]


def test_video_file_v1_serves_its_conversion() -> None:
    case, _ = EXPECTED_FALLBACKS["video_file_part"]
    v1 = run_v1_request_transform("hosted_vllm", case)
    assert v1["messages"][0]["content"][1] == {
        "type": "video_url",
        "video_url": {"url": "https://x/v.mp4"},
    }


def test_top_k_fallback_reason_matches_the_wire_proven_extra_body_merge() -> None:
    """hosted_vllm IS in openai_compatible_providers: top_k rides the
    extra_body packing and hh merges it onto the wire (the wave-2b
    wire-prove rule — never inherit the reason text without a pinned
    row)."""
    case, _ = EXPECTED_FALLBACKS["top_k"]
    packed = get_optional_params(
        model=MODEL, custom_llm_provider="hosted_vllm", stream=None, top_k=5
    )
    assert packed["extra_body"] == {"top_k": 5}
    assert "top_k" not in run_v1_request_transform("hosted_vllm", copy.deepcopy(case))
    wire = capture_v1_wire_body(
        f"hosted_vllm/{MODEL}", api_base="https://vllm.example.test/v1", top_k=5
    )
    assert wire["top_k"] == 5
    assert "extra_body" not in wire


def test_user_v1_drops_it() -> None:
    case, _ = EXPECTED_FALLBACKS["user"]
    assert "user" not in run_v1_request_transform("hosted_vllm", case)


def test_supported_list_mirror() -> None:
    """v1's list is the base list + thinking/reasoning_effort appended
    UNCONDITIONALLY (no capability fork, no model map needed — fixed name
    samples, the lm_studio precedent)."""
    from litellm.translation.providers.hosted_vllm.params import _HOSTED_VLLM_LIST

    from ._own_module_corpus import provider_config

    mirror_keys = (
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "stream",
        "stop",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "response_format",
    )
    assert _HOSTED_VLLM_LIST <= set(mirror_keys) | {"thinking", "reasoning_effort"}
    for model in (MODEL, "meta-llama/Llama-3.1-8B-Instruct", "gpt-4"):
        supported = set(
            provider_config("hosted_vllm", model).get_supported_openai_params(model)
        )
        # the inherited base list's gpt-4/gpt-3.5-turbo-16k response_format
        # name gate; v2 composes openai_compat.unsupported_response_format
        # for exactly these names
        allowed = (
            _HOSTED_VLLM_LIST - {"response_format"}
            if model in ("gpt-4", "gpt-3.5-turbo-16k")
            else _HOSTED_VLLM_LIST
        )
        for key in mirror_keys:
            assert (key in allowed) == (key in supported), (model, key)
        assert {"thinking", "reasoning_effort"} <= supported, model


def test_registration_facts() -> None:
    assert "hosted_vllm" in pipeline._SERIALIZERS
    assert "hosted_vllm" in pipeline._RESPONSE_PARSERS
    assert "hosted_vllm" in pipeline._RAW_GUARDS
    assert pipeline.response_dialect("hosted_vllm") == "openai"
    assert "hosted_vllm" not in NEVER_PORT
