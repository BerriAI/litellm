"""Differential parity: v2 xai translation vs the v1 XAIChatConfig chain.

Two-sided over the generated characterization corpus (provenance:
characterization_xai/README.md): v1 AT HEAD must still equal the committed
snapshot (drift guard) AND v2 must equal the snapshot byte-for-byte. The v1
invoker is ``get_optional_params(custom_llm_provider="xai")`` -> extra_body
pop -> ``transform_request`` — v1 AS EXECUTED on the httpx path, never bare
``map_openai_params`` (whose max_completion_tokens rename arm is dead code:
``_check_valid_arg`` raises first; researcher-3 R2).

The R2 gate rows therefore pin the RAISE: v1 must raise
UnsupportedParamsError in-process AND v2 must return a typed fallback, so
flag-on traffic gets v1's own error, never a remap.
"""

import copy
import json

import pytest

from litellm.exceptions import UnsupportedParamsError

from litellm.translation import translate_chat_request

from ._xai_corpus import (
    SNAPSHOTS_DIR,
    canonical_json,
    corpus,
    load_json,
    run_v1_request_transform,
)
from .conftest import build_real_deps

CASES = corpus("cases")

_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}

_USER = [{"role": "user", "content": "x"}]

# Rows where v1 RAISES UnsupportedParamsError (the R2 supported-list gate);
# v2 must be a typed fallback so v1 serves its own raise. The reason fragment
# is asserted on the v2 error; the raise is asserted on v1 in-process.
V1_RAISES = {
    "max_completion_tokens_any_grok": (
        {"model": "grok-4-0709", "max_completion_tokens": 128, "messages": _USER},
        "max_completion_tokens",
    ),
    "max_completion_tokens_grok3mini": (
        {"model": "grok-3-mini", "max_completion_tokens": 128, "messages": _USER},
        "max_completion_tokens",
    ),
    "stop_on_grok4": (
        {"model": "grok-4-0709", "stop": ["END"], "messages": _USER},
        "stop on grok-4-0709",
    ),
    "stop_on_grok3mini": (
        {"model": "grok-3-mini", "stop": ["END"], "messages": _USER},
        "stop on grok-3-mini",
    ),
    "stop_on_grok_code_fast": (
        {"model": "grok-code-fast-1", "stop": ["END"], "messages": _USER},
        "stop on grok-code-fast-1",
    ),
    "reasoning_effort_on_non_reasoning_grok4": (
        {"model": "grok-4-0709", "reasoning_effort": "high", "messages": _USER},
        "reasoning_effort on non-reasoning xai model",
    ),
    "frequency_penalty_on_grok4": (
        # parse-level fallback (penalties are outside the IR); v1's gate
        # raises for this family, so the fallback serves v1's own error
        {"model": "grok-4-0709", "frequency_penalty": 0.5, "messages": _USER},
        "frequency_penalty",
    ),
}

# Typed fallbacks where v1 SERVES the request (v1 is not invoked: the seam
# routes these to v1 untouched, so v1's behavior is by-construction v1's).
EXPECTED_FALLBACKS = {
    "web_search_options_responses_bridge": (
        {
            "model": "grok-4-0709",
            "web_search_options": {"search_context_size": "high"},
            "messages": _USER,
        },
        "Responses-API bridge",
    ),
    "use_xai_oauth_pkce_flow": (
        {"model": "grok-4-0709", "use_xai_oauth": True, "messages": _USER},
        "PKCE",
    ),
    "explicit_stream_false_reaches_wire": (
        {"model": "grok-4-0709", "stream": False, "messages": _USER},
        "explicit stream: false",
    ),
    "user_message_name_forwarded_by_v1": (
        {
            "model": "grok-4-0709",
            "messages": [{"role": "user", "content": "x", "name": "alice"}],
        },
        "message name field",
    ),
    "nested_tool_strict_below_function": (
        {
            "model": "grok-4-0709",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "f",
                        "parameters": {
                            "type": "object",
                            "properties": {"strict": {"type": "boolean"}},
                        },
                    },
                }
            ],
            "messages": _USER,
        },
        "nested 'strict' key",
    ),
    "presence_penalty_outside_ir": (
        # v1 passes presence_penalty through for every grok model; the hub
        # keeps penalties parse-level fallbacks (integrator's unified
        # _raw_openai_body semantics), so v1 serves it
        {"model": "grok-4-0709", "presence_penalty": 0.5, "messages": _USER},
        "presence_penalty",
    ),
    "frequency_penalty_supported_family_outside_ir": (
        # supported on grok-3-mini in v1; still a parse-level fallback
        {"model": "grok-3-mini", "frequency_penalty": 0.5, "messages": _USER},
        "frequency_penalty",
    ),
    "seed_outside_ir": (
        {"model": "grok-4-0709", "seed": 42, "messages": _USER},
        "seed",
    ),
    "logprobs_outside_ir": (
        {"model": "grok-4-0709", "logprobs": True, "messages": _USER},
        "logprobs",
    ),
    "stream_options_outside_ir": (
        {
            "model": "grok-4-0709",
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": _USER,
        },
        "stream_options",
    ),
    "string_form_stop_supported_family": (
        {"model": "grok-3", "stop": "END", "messages": _USER},
        "string-form stop",
    ),
    "both_max_tokens_keys": (
        {
            "model": "grok-4-0709",
            "max_tokens": 5,
            "max_completion_tokens": 6,
            "messages": _USER,
        },
        "both max_tokens and max_completion_tokens",
    ),
    "top_k_not_an_xai_param": (
        {"model": "grok-4-0709", "top_k": 40, "messages": _USER},
        "top_k",
    ),
    "image_detail_key": (
        {
            "model": "grok-4-0709",
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
    "consecutive_user_messages": (
        {
            "model": "grok-4-0709",
            "messages": [
                {"role": "user", "content": "a"},
                {"role": "user", "content": "b"},
            ],
        },
        "consecutive user messages",
    ),
    "empty_tools_list": (
        {"model": "grok-4-0709", "tools": [], "messages": _USER},
        "empty tools list",
    ),
}


def _v2(case: dict):
    return translate_chat_request(copy.deepcopy(case), "xai", build_real_deps())


def _norm(body: dict) -> str:
    return json.dumps(body, sort_keys=True, default=str)


@pytest.mark.parametrize("name", sorted(CASES))
def test_v1_at_head_still_matches_the_snapshot(name: str) -> None:
    snapshot = (SNAPSHOTS_DIR / "requests" / f"{name}.json").read_text()
    assert canonical_json(run_v1_request_transform(CASES[name])) == snapshot


@pytest.mark.parametrize("name", sorted(CASES))
def test_v2_request_matches_the_snapshot(name: str) -> None:
    result = _v2(CASES[name])
    assert result.is_ok(), result.error.summary
    snapshot = load_json(SNAPSHOTS_DIR / "requests" / f"{name}.json")
    assert _norm(result.ok) == _norm(snapshot)


@pytest.mark.parametrize("name", sorted(V1_RAISES))
def test_v1_raise_rows_fall_back_typed(name: str) -> None:
    case, reason_fragment = V1_RAISES[name]
    result = _v2(case)
    assert result.is_error(), f"{name} unexpectedly translated: {result.ok!r}"
    assert reason_fragment in result.error.summary, result.error.summary
    with pytest.raises(UnsupportedParamsError):
        run_v1_request_transform(case)


@pytest.mark.parametrize("name", sorted(EXPECTED_FALLBACKS))
def test_unsupported_shape_is_a_typed_fallback(name: str) -> None:
    case, reason_fragment = EXPECTED_FALLBACKS[name]
    result = _v2(case)
    assert result.is_error(), f"{name} unexpectedly translated: {result.ok!r}"
    assert reason_fragment in result.error.summary, result.error.summary


def test_reasoning_gate_matches_v1_capability_read() -> None:
    """The v2 gate reads deps.supports_capability over the ``xai/{model}``
    map key; it must agree with v1's litellm.supports_reasoning on the
    models the corpus serves and falls back."""
    import litellm

    from litellm.translation.providers.xai.params import supports_reasoning

    deps = build_real_deps()
    for model in (
        "grok-3",
        "grok-3-mini",
        "grok-4-0709",
        "grok-code-fast-1",
        "grok-2-1212",
    ):
        assert supports_reasoning(model, deps) == litellm.supports_reasoning(
            model=model, custom_llm_provider="xai"
        ), model
