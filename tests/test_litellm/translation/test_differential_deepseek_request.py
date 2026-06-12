"""Differential parity for the deepseek request path (httpx dedicated elif,
main.py:1942; transform LIVE; no model prefix).

Two-sided: every served row is byte-identical (normalized JSON) between v1
in-process at HEAD (get_optional_params + DeepSeekChatConfig.transform_request,
the _own_module_corpus invoker) and v2; every fallback row asserts BOTH the
typed v2 error and v1's own behavior (serve or rewrite, asserted in-process).
The thinking rewrite and the content-list flatten are v1 facts re-verified
here, not dossier trust. Two dossier-drift pins recorded in
wave2b-alpha-port.md: (1) deepseek does NOT drop tool_choice auto/none (that
fact was deepinfra's); tool_choice rides the base list verbatim, pinned by
the tools rows below; (2) web_search_options is SERVED by v1's base list at
HEAD (researcher-4 listed it as a raise) — the inbound boundary falls back
either way, and the row pins v1's serve so the reason stays honest.
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

MODEL = "deepseek-chat"
REASONER = "deepseek-reasoner"
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
    # mct passes VERBATIM: no rename arm anywhere on the deepseek chain
    "max_completion_tokens_verbatim": {
        "model": MODEL,
        "max_completion_tokens": 128,
        "messages": _USER,
    },
    "temperature_int_stays_int": {"model": MODEL, "temperature": 1, "messages": _USER},
    "tools_auto": {
        "model": MODEL,
        "tools": [WEATHER_TOOL],
        "tool_choice": "auto",
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
    },
    # the dossier-drift pin: tool_choice "none" rides the base list verbatim
    # (the claimed deepseek auto/none drop was deepinfra's fact)
    "tool_choice_none_verbatim": {
        "model": MODEL,
        "tools": [WEATHER_TOOL],
        "tool_choice": "none",
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
    # the always-on flatten: text-only content lists join ("ab")
    "content_list_flattened": {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "a"},
                    {"type": "text", "text": "b"},
                ],
            }
        ],
    },
    # thinking rewrite laws (map_openai_params:49-63), not model-gated:
    "thinking_enabled_budget_discarded": {
        "model": MODEL,
        "thinking": {"type": "enabled", "budget_tokens": 5000},
        "messages": _USER,
    },
    "thinking_disabled_dropped": {
        "model": MODEL,
        "thinking": {"type": "disabled"},
        "messages": _USER,
    },
    "reasoning_effort_rewrites_to_thinking": {
        "model": MODEL,
        "reasoning_effort": "high",
        "messages": _USER,
    },
    "reasoning_effort_none_dropped": {
        "model": MODEL,
        "reasoning_effort": "none",
        "messages": _USER,
    },
    # the elif shadow: a present-but-disabled thinking dict suppresses the
    # reasoning_effort rewrite entirely
    "thinking_disabled_shadows_reasoning_effort": {
        "model": MODEL,
        "thinking": {"type": "disabled"},
        "reasoning_effort": "high",
        "messages": _USER,
    },
    # thinking mode on a reasoning model with NO assistant history: the
    # fill has nothing to patch, so the request serves
    "reasoner_thinking_without_history": {
        "model": REASONER,
        "thinking": {"type": "enabled"},
        "messages": _USER,
    },
    # the fill is capability-gated: deepseek-chat (supports_reasoning False)
    # with thinking enabled and assistant history SERVES untouched
    "thinking_history_inert_on_non_reasoning_model": {
        "model": MODEL,
        "thinking": {"type": "enabled"},
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "prev"},
            {"role": "user", "content": "next"},
        ],
    },
}

# name -> (case, v2 reason fragment). Every row's v1 side (serve or rewrite)
# is asserted in-process by the dedicated tests below; the generator emits
# these as FALLBACK (v1 serves it) rows.
EXPECTED_FALLBACKS: dict[str, tuple[dict, str]] = {
    "thinking_fill_history": (
        {
            "model": REASONER,
            "thinking": {"type": "enabled"},
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "prev"},
                {"role": "user", "content": "next"},
            ],
        },
        "_fill_reasoning_content",
    ),
    "reasoning_effort_fill_history": (
        # the rewrite path arms the fill too: reasoning_effort -> thinking
        # enabled happens at map time, BEFORE transform reads it
        {
            "model": REASONER,
            "reasoning_effort": "low",
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "prev"},
                {"role": "user", "content": "next"},
            ],
        },
        "_fill_reasoning_content",
    ),
    "non_text_content": (
        {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://x/i.png"},
                        },
                    ],
                }
            ],
        },
        "lossy flatten",
    ),
    "top_k": ({"model": MODEL, "top_k": 5, "messages": _USER}, "extra_body"),
    "user": ({"model": MODEL, "user": "u1", "messages": _USER}, "user"),
    "stream_false": ({"model": MODEL, "stream": False, "messages": _USER}, "stream"),
    # dossier drift: v1 SERVES web_search_options at HEAD (base list grew
    # it); v2's inbound boundary falls back typed like every parse-level
    # unknown — v1 serves, so the fallback is safe
    "web_search_options": (
        {
            "model": MODEL,
            "web_search_options": {"search_context_size": "medium"},
            "messages": _USER,
        },
        "web_search_options",
    ),
}


def _v2(case: dict):
    return translate_chat_request(copy.deepcopy(case), "deepseek", build_real_deps())


def _norm(body: dict) -> str:
    return json.dumps(body, sort_keys=True, default=str)


@pytest.mark.parametrize("name", sorted(CASES))
def test_v2_request_matches_v1(name: str) -> None:
    case = CASES[name]
    result = _v2(case)
    assert result.is_ok(), result.error.summary
    assert _norm(result.ok) == _norm(run_v1_request_transform("deepseek", case))


@pytest.mark.parametrize("name", sorted(EXPECTED_FALLBACKS))
def test_expected_fallbacks_are_typed(name: str) -> None:
    case, reason = EXPECTED_FALLBACKS[name]
    result = _v2(case)
    assert result.is_error(), f"{name} unexpectedly served"
    assert reason in result.error.summary, result.error.summary


def test_thinking_rewrite_emits_enabled_only() -> None:
    """Semantic pin on top of the byte rows: the wire thinking value is
    exactly {"type": "enabled"} (budget_tokens discarded)."""
    result = _v2(CASES["thinking_enabled_budget_discarded"])
    assert result.is_ok()
    assert result.ok["thinking"] == {"type": "enabled"}
    result = _v2(CASES["reasoning_effort_rewrites_to_thinking"])
    assert result.is_ok()
    assert result.ok["thinking"] == {"type": "enabled"}
    assert "reasoning_effort" not in result.ok


def test_thinking_fill_v1_serves_its_rewrite() -> None:
    """Thinking mode + assistant history on a reasoning model: v1 injects
    reasoning_content ' ' into every assistant message missing it (asserted
    in-process) — for BOTH arming paths (verbatim thinking dict and the
    reasoning_effort rewrite, which lands before transform reads it)."""
    for name in ("thinking_fill_history", "reasoning_effort_fill_history"):
        case, _ = EXPECTED_FALLBACKS[name]
        v1 = run_v1_request_transform("deepseek", case)
        assert v1["messages"][1]["reasoning_content"] == " ", name


def test_thinking_fill_inert_on_non_reasoning_models() -> None:
    """The fill is capability-gated: deepseek-chat with thinking enabled and
    assistant history SERVES (v1 leaves the history untouched)."""
    case = CASES["thinking_history_inert_on_non_reasoning_model"]
    result = _v2(case)
    assert result.is_ok(), result.error.summary
    v1 = run_v1_request_transform("deepseek", case)
    assert "reasoning_content" not in v1["messages"][1]
    assert _norm(result.ok) == _norm(v1)


def test_non_text_content_v1_flatten_is_lossy() -> None:
    case, _ = EXPECTED_FALLBACKS["non_text_content"]
    v1 = run_v1_request_transform("deepseek", case)
    assert v1["messages"][0]["content"] == "look"  # the image part is GONE


def test_top_k_fallback_reason_matches_the_wire_proven_extra_body_merge() -> None:
    """deepseek is in litellm.openai_compatible_providers, so top_k rides
    the extra_body packing (never the top-level passthrough); hh merges the
    contents onto the wire AFTER transform (hh:448). All three claims
    pinned: the packing, the transform-output absence, and the WIRE merge
    via the full completion() stack against a mock transport (the wave-2b
    wire-prove rule: never inherit the reason text without a pinned row)."""
    case, _ = EXPECTED_FALLBACKS["top_k"]
    packed = get_optional_params(
        model=MODEL, custom_llm_provider="deepseek", stream=None, top_k=5
    )
    assert packed["extra_body"] == {"top_k": 5}
    assert "top_k" not in run_v1_request_transform("deepseek", copy.deepcopy(case))
    wire = capture_v1_wire_body("deepseek/deepseek-chat", top_k=5)
    assert wire["top_k"] == 5
    assert "extra_body" not in wire


def test_user_v1_drops_it() -> None:
    case, _ = EXPECTED_FALLBACKS["user"]
    assert "user" not in run_v1_request_transform("deepseek", case)


def test_web_search_options_v1_serves_it() -> None:
    """The drift pin: researcher-4 listed web_search_options as a deepseek
    RAISE; at HEAD the base list carries it and v1 serves it verbatim. The
    v2 fallback (inbound boundary) is therefore the v1-serves kind."""
    case, _ = EXPECTED_FALLBACKS["web_search_options"]
    v1 = run_v1_request_transform("deepseek", copy.deepcopy(case))
    assert v1["web_search_options"] == {"search_context_size": "medium"}


def test_supported_list_mirror_over_model_map() -> None:
    """The hand-copied _DEEPSEEK_LIST must track v1's
    get_supported_openai_params at HEAD for every deepseek chat map row
    (critic-grok M4 / compat_sdk mirror shape): base keys row-for-row, and
    thinking/reasoning_effort UNCONDITIONAL on every model. Direction
    argument: every key v2 can serve is in the mirror set; a v1 list
    growing a key v2 lacks only widens the typed fallback."""
    import litellm

    from litellm.translation.providers.deepseek.params import _DEEPSEEK_LIST

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
    assert _DEEPSEEK_LIST <= set(mirror_keys) | {"thinking", "reasoning_effort"}
    models = [MODEL, REASONER] + sorted(
        key.split("/", 1)[1]
        for key, info in litellm.model_cost.items()
        if key.startswith("deepseek/")
        and isinstance(info, dict)
        and info.get("mode") == "chat"
    )
    assert len(models) >= 2
    for model in models:
        supported = set(
            provider_config("deepseek", model).get_supported_openai_params(model)
        )
        for key in mirror_keys:
            assert (key in _DEEPSEEK_LIST) == (key in supported), (model, key)
        assert {"thinking", "reasoning_effort"} <= supported, model


def test_reasoning_capability_mirror_over_model_map() -> None:
    """The fill gate reads deps over deepseek/{m}: assert the deps read
    agrees with v1's supports_reasoning for every deepseek chat map row
    (drift gate, the wave-2a mirror shape)."""
    import litellm
    from litellm.utils import supports_reasoning

    deps = build_real_deps()
    rows = [
        key.split("/", 1)[1]
        for key, info in litellm.model_cost.items()
        if key.startswith("deepseek/")
        and isinstance(info, dict)
        and info.get("mode") == "chat"
    ]
    assert len(rows) > 0
    for short in rows:
        assert deps.supports_capability(
            f"deepseek/{short}", "supports_reasoning"
        ) == supports_reasoning(short, custom_llm_provider="deepseek"), short


def test_registration_facts() -> None:
    assert "deepseek" in pipeline._SERIALIZERS
    assert "deepseek" in pipeline._RESPONSE_PARSERS
    assert "deepseek" in pipeline._RAW_GUARDS
    assert pipeline.response_dialect("deepseek") == "openai"
    assert "deepseek" not in NEVER_PORT
