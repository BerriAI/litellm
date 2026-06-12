"""Differential parity for the openrouter request path (httpx dedicated
elif, main.py:3354; transform LIVE; no model prefix).

Two-sided: every served row is byte-identical (normalized JSON) between v1
in-process at HEAD (get_optional_params + OpenrouterConfig.transform_request,
the _own_module_corpus invoker) and v2; every fallback row asserts BOTH the
typed v2 error and v1's own behavior (serve, rewrite, or raise — asserted
in-process). Dossier-drift pins recorded in wave2b-alpha-port.md: openrouter
is NOT in openai_compatible_providers, so top_k (and transforms/models/route)
ride the TOP-LEVEL passthrough — researcher-4 read the map's extra_body
packing, which is DEAD at runtime (those kwargs never reach
non_default_params); top_k is therefore SERVED verbatim, wire-proven below.
"""

import copy
import json

import pytest

from litellm.exceptions import UnsupportedParamsError

from litellm.translation.dispatch import NEVER_PORT
from litellm.translation.engine import pipeline
from litellm.translation.engine.pipeline import translate_chat_request

from ._own_module_corpus import capture_v1_wire_body, run_v1_request_transform
from .conftest import build_real_deps

MODEL = "openai/gpt-4o"  # not reasoning-capable on the openrouter map
REASONER = "anthropic/claude-3.7-sonnet"  # openrouter/{m} supports_reasoning
_USER = [{"role": "user", "content": "Hello, world"}]

WEATHER_TOOL = {
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
    "stop_list": {"model": MODEL, "stop": ["END", "STOP"], "messages": _USER},
    # mct passes VERBATIM (no rename arm on the openrouter chain)
    "max_completion_tokens_verbatim": {
        "model": MODEL,
        "max_completion_tokens": 128,
        "messages": _USER,
    },
    "temperature_int_stays_int": {"model": MODEL, "temperature": 1, "messages": _USER},
    # the drift pin: top_k rides the TOP-LEVEL passthrough (openrouter is
    # not a compat provider), so v2 serves it as a delta
    "top_k_top_level": {"model": MODEL, "top_k": 5, "messages": _USER},
    "tools_auto": {
        "model": MODEL,
        "tools": [WEATHER_TOOL],
        "tool_choice": "auto",
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
    },
    "tool_choice_specific": {
        "model": MODEL,
        "tools": [WEATHER_TOOL],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
    },
    "tool_call_compact_roundtrip": {
        "model": MODEL,
        "tools": [WEATHER_TOOL],
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
        "tools": [WEATHER_TOOL],
        "parallel_tool_calls": False,
        "messages": [{"role": "user", "content": "Weather in Paris and Rome?"}],
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
    # reasoning_effort SERVED verbatim on dual-key reasoning-capable models
    "reasoning_effort_on_reasoner": {
        "model": REASONER,
        "reasoning_effort": "high",
        "messages": _USER,
    },
    # cache_control on a NON-cache-capable model: v1's base recursive strip
    # == the IR's silent drop, byte-identical (block + message level)
    "cache_control_stripped_on_non_cache_model": {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "a",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"type": "text", "text": "b"},
                ],
                "cache_control": {"type": "ephemeral"},
            }
        ],
    },
}

# name -> (case, v2 reason fragment); generator emits FALLBACK (v1 serves it)
EXPECTED_FALLBACKS: dict[str, tuple[dict, str]] = {
    "thinking_on_reasoner": (
        {
            "model": REASONER,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "messages": _USER,
        },
        "VERBATIM",
    ),
    "cache_control_on_claude": (
        {
            "model": REASONER,
            "messages": [
                {
                    "role": "user",
                    "content": "plain",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        "cache-capable",
    ),
    "user": ({"model": MODEL, "user": "u1", "messages": _USER}, "user"),
    "stream_false": ({"model": MODEL, "stream": False, "messages": _USER}, "stream"),
    # openrouter-native routing params ride the top-level passthrough in v1;
    # inbound unknowns in v2
    "transforms": (
        {"model": MODEL, "transforms": ["middle-out"], "messages": _USER},
        "transforms",
    ),
    "route": ({"model": MODEL, "route": "fallback", "messages": _USER}, "route"),
}

# name -> (case, v2 reason fragment); generator emits FALLBACK (v1 raises UPE)
V1_RAISES: dict[str, tuple[dict, str]] = {
    "reasoning_effort_on_non_reasoning_model": (
        {"model": MODEL, "reasoning_effort": "high", "messages": _USER},
        "reasoning_effort",
    ),
    "thinking_on_non_reasoning_model": (
        {"model": MODEL, "thinking": {"type": "enabled"}, "messages": _USER},
        "thinking",
    ),
}


def _v2(case: dict):
    return translate_chat_request(copy.deepcopy(case), "openrouter", build_real_deps())


def _norm(body: dict) -> str:
    return json.dumps(body, sort_keys=True, default=str)


@pytest.mark.parametrize("name", sorted(CASES))
def test_v2_request_matches_v1(name: str) -> None:
    case = CASES[name]
    result = _v2(case)
    assert result.is_ok(), result.error.summary
    assert _norm(result.ok) == _norm(run_v1_request_transform("openrouter", case))


@pytest.mark.parametrize("name", sorted(EXPECTED_FALLBACKS))
def test_expected_fallbacks_are_typed(name: str) -> None:
    case, reason = EXPECTED_FALLBACKS[name]
    result = _v2(case)
    assert result.is_error(), f"{name} unexpectedly served"
    assert reason in result.error.summary, result.error.summary


@pytest.mark.parametrize("name", sorted(V1_RAISES))
def test_v1_raise_rows_are_two_sided(name: str) -> None:
    case, reason = V1_RAISES[name]
    result = _v2(case)
    assert result.is_error(), f"{name} unexpectedly served"
    assert reason in result.error.summary, result.error.summary
    with pytest.raises(UnsupportedParamsError):
        run_v1_request_transform("openrouter", case)


def test_usage_include_is_injected_on_every_body() -> None:
    """The always-on delta: v1 injects usage:{include:true} into EVERY body
    (its 'if "usage" not in response' arm is unreachable for v2 shapes)."""
    for name in ("text", "stream_true", "tools_auto"):
        result = _v2(CASES[name])
        assert result.is_ok()
        assert result.ok["usage"] == {"include": True}, name


def test_top_k_wire_proven_top_level() -> None:
    """openrouter is NOT in openai_compatible_providers: top_k rides the
    top-level passthrough (never extra_body) and reaches the wire verbatim —
    pinned at the transform seam AND the wire via the full completion()
    stack against a mock transport (the wave-2b wire-prove rule)."""
    v1 = run_v1_request_transform("openrouter", CASES["top_k_top_level"])
    assert v1["top_k"] == 5
    wire = capture_v1_wire_body("openrouter/openai/gpt-4o", top_k=5)
    assert wire["top_k"] == 5
    assert "extra_body" not in wire
    assert wire["usage"] == {"include": True}


def test_thinking_on_reasoner_v1_serves_verbatim() -> None:
    case, _ = EXPECTED_FALLBACKS["thinking_on_reasoner"]
    v1 = run_v1_request_transform("openrouter", case)
    assert v1["thinking"] == {"type": "enabled", "budget_tokens": 1024}


def test_cache_control_move_v1_serves_its_rewrite() -> None:
    """Cache-capable models: v1 moves the message-level marker into the last
    content block (string content becomes a one-block list)."""
    case, _ = EXPECTED_FALLBACKS["cache_control_on_claude"]
    v1 = run_v1_request_transform("openrouter", case)
    assert v1["messages"][0]["content"] == [
        {"type": "text", "text": "plain", "cache_control": {"type": "ephemeral"}}
    ]
    assert "cache_control" not in v1["messages"][0]


def test_user_v1_drops_it() -> None:
    case, _ = EXPECTED_FALLBACKS["user"]
    assert "user" not in run_v1_request_transform("openrouter", case)


def test_transforms_route_v1_serves_top_level() -> None:
    for name in ("transforms", "route"):
        case, key = EXPECTED_FALLBACKS[name]
        v1 = run_v1_request_transform("openrouter", case)
        assert key in v1, name


def test_supported_list_mirror_over_model_map() -> None:
    """The hand-copied gate must track v1's get_supported_openai_params at
    HEAD for every openrouter chat map row: base keys row-for-row, and
    thinking/reasoning_effort exactly on the DUAL-key reasoning-capable
    models (incl. the openrouter/-prefixed ids v1's prefix-strip makes
    UNREACHABLE in the map — they answer False)."""
    import litellm

    from litellm.translation.providers.openrouter.params import (
        _OPENROUTER_LIST,
        supports_openrouter_reasoning,
    )

    from ._own_module_corpus import provider_config

    deps = build_real_deps()
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
    assert _OPENROUTER_LIST <= set(mirror_keys) | {"top_k", "reasoning_effort"}
    models = [MODEL, REASONER, "openrouter/auto", "openrouter/free"] + sorted(
        key.split("/", 1)[1]
        for key, info in litellm.model_cost.items()
        if key.startswith("openrouter/")
        and isinstance(info, dict)
        and info.get("mode") == "chat"
    )
    assert len(models) > 50
    for model in models:
        supported = set(
            provider_config("openrouter", model).get_supported_openai_params(model)
        )
        for key in mirror_keys:
            assert (key in _OPENROUTER_LIST) == (key in supported), (model, key)
        capable = supports_openrouter_reasoning(model, deps)
        assert capable == ("reasoning_effort" in supported), model
        assert capable == ("thinking" in supported), model


def test_registration_facts() -> None:
    assert "openrouter" in pipeline._SERIALIZERS
    assert "openrouter" in pipeline._RESPONSE_PARSERS
    assert "openrouter" in pipeline._RAW_GUARDS
    assert pipeline.response_dialect("openrouter") == "openai"
    assert "openrouter" not in NEVER_PORT
