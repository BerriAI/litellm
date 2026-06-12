"""Differential parity for the groq chat request path (wave-2b-beta).

v1 side, invoked the way main.py's dedicated groq elif runs:
``get_optional_params("groq")`` then ``GroqChatConfig.transform_request``
(assistant None-strip + the base GPT assembly), with hh's envelope pops
mirrored (fake_stream/json_mode are routing keys; ``extra_body`` is popped
and MERGED into the wire body — groq's map packs ``top_k`` there, unlike
the cohere/mistral top-level passthrough).

The json_schema three-way fork (the wave's cross-plane feature, ported as
researcher-4 prescribed): native-schema models (the ``groq/{m}``
``supports_response_schema`` map flag) serve response_format VERBATIM;
non-native models fall back (v1 serves its json_tool_call workaround);
non-native + tools falls back where v1 raises ``litellm.BadRequestError``
(the exact raise class pinned — NOT UnsupportedParamsError).

NOTE for the fork wirer (main.py:2338-2344): the groq elif merges
``GroqChatConfig.get_config()`` class attrs into optional_params — ambient
module state v2 cannot see; the seam must fall back when that config is
non-empty (the ambient-globals rule; CLAUDE.md fork obligations).
"""

import copy
import json

import pytest

import litellm
from litellm.exceptions import BadRequestError, UnsupportedParamsError
from litellm.llms.groq.chat.transformation import GroqChatConfig
from litellm.utils import get_optional_params

from litellm.translation.engine.pipeline import translate_chat_request
from litellm.translation_seam import build_translation_deps

MODEL = "llama-3.3-70b-versatile"
NATIVE_SCHEMA_MODEL = "openai/gpt-oss-120b"
REASONING_MODEL = "qwen/qwen3-32b"
_U = [{"role": "user", "content": "hi"}]

_SCHEMA_RF = {
    "type": "json_schema",
    "json_schema": {
        "name": "s",
        "schema": {"type": "object", "properties": {}},
        "strict": True,
    },
}

CASES = {
    "plain": {"model": MODEL, "messages": _U},
    "sampling": {
        "model": MODEL,
        "messages": _U,
        "temperature": 0.5,
        "top_p": 0.9,
        "max_tokens": 10,
        "stop": ["x"],
    },
    "mct_renamed": {"model": MODEL, "messages": _U, "max_completion_tokens": 50},
    "top_k_extra_body_merged": {"model": MODEL, "messages": _U, "top_k": 7},
    "stream_true": {"model": MODEL, "messages": _U, "stream": True},
    "tools": {
        "model": MODEL,
        "messages": _U,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "tool_choice": "auto",
    },
    "parallel_tool_calls": {
        "model": MODEL,
        "messages": _U,
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
        "parallel_tool_calls": True,
    },
    "response_format_json_object": {
        "model": MODEL,
        "messages": _U,
        "response_format": {"type": "json_object"},
    },
    "response_format_schema_native_passthrough": {
        "model": NATIVE_SCHEMA_MODEL,
        "messages": _U,
        "response_format": copy.deepcopy(_SCHEMA_RF),
    },
    "reasoning_effort_capable_model": {
        "model": REASONING_MODEL,
        "messages": _U,
        "reasoning_effort": "high",
    },
    "assistant_none_strip": {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "r"},
        ],
    },
}

V1_RAISES = {
    "thinking": (
        {"model": MODEL, "messages": _U, "thinking": {"type": "enabled"}},
        "thinking",
    ),
    "reasoning_effort_non_reasoning_model": (
        {"model": MODEL, "messages": _U, "reasoning_effort": "high"},
        "reasoning_effort",
    ),
}

V1_SERVES_FALLBACKS = {
    "user_silent_drop": ({"model": MODEL, "messages": _U, "user": "u1"}, "user"),
    "response_format_schema_workaround": (
        {
            "model": MODEL,
            "messages": _U,
            "response_format": copy.deepcopy(_SCHEMA_RF),
        },
        "json_tool_call workaround",
    ),
    "seed_parse_level": ({"model": MODEL, "messages": _U, "seed": 42}, "seed"),
    "n_parse_level": ({"model": MODEL, "messages": _U, "n": 2}, "n"),
    "service_tier_parse_level": (
        {"model": MODEL, "messages": _U, "service_tier": "flex"},
        "service_tier",
    ),
    "logit_bias_parse_level": (
        {"model": MODEL, "messages": _U, "logit_bias": {"1": 1}},
        "logit_bias",
    ),
    "explicit_stream_false": (
        {"model": MODEL, "messages": _U, "stream": False},
        "stream",
    ),
    "message_name_forwarded": (
        {"model": MODEL, "messages": [{"role": "user", "content": "hi", "name": "b"}]},
        "name",
    ),
}


def run_v1_request_transform(case: dict) -> dict:
    """May RAISE (UnsupportedParamsError / BadRequestError): pinned v1
    behavior for the raise rows."""
    request = copy.deepcopy(case)
    model = request.pop("model")
    messages = request.pop("messages")
    optional_params = get_optional_params(
        model=model,
        custom_llm_provider="groq",
        messages=copy.deepcopy(messages),
        stream=request.pop("stream", None),
        **request,
    )
    optional_params.pop("fake_stream", None)
    optional_params.pop("json_mode", None)
    extra_body = optional_params.pop("extra_body", {}) or {}
    body = GroqChatConfig().transform_request(
        model=model,
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )
    return {**body, **extra_body}


def _v2(case: dict):
    return translate_chat_request(copy.deepcopy(case), "groq", build_translation_deps())


def _norm(body: dict) -> str:
    return json.dumps(body, sort_keys=True, default=str)


@pytest.mark.parametrize("name", sorted(CASES))
def test_v2_request_matches_v1(name: str) -> None:
    case = CASES[name]
    result = _v2(case)
    assert result.is_ok(), result.error.summary
    assert _norm(result.ok) == _norm(run_v1_request_transform(case))


@pytest.mark.parametrize("name", sorted(V1_RAISES))
def test_v1_raise_rows_fall_back_typed(name: str) -> None:
    case, fragment = V1_RAISES[name]
    result = _v2(case)
    assert result.is_error(), f"{name} unexpectedly translated: {result.ok!r}"
    assert fragment in result.error.summary, result.error.summary
    with pytest.raises(UnsupportedParamsError):
        run_v1_request_transform(case)


def test_schema_with_tools_falls_back_where_v1_raises_bad_request() -> None:
    """The wave's one non-UnsupportedParamsError request raise: pinned exact
    type (litellm.BadRequestError)."""
    case = {
        "model": MODEL,
        "messages": _U,
        "response_format": copy.deepcopy(_SCHEMA_RF),
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
    }
    result = _v2(case)
    assert result.is_error()
    assert "BadRequestError" in result.error.summary
    with pytest.raises(BadRequestError):
        run_v1_request_transform(case)


@pytest.mark.parametrize("name", sorted(V1_SERVES_FALLBACKS))
def test_v1_serves_fallback_rows(name: str) -> None:
    case, fragment = V1_SERVES_FALLBACKS[name]
    result = _v2(case)
    assert result.is_error(), f"{name} unexpectedly translated: {result.ok!r}"
    assert fragment in result.error.summary, result.error.summary
    run_v1_request_transform(case)


def test_native_schema_capability_is_the_fork_truth() -> None:
    """The three-way fork keys off the groq/{m} supports_response_schema map
    flag — pin that the chosen models disagree so the fork rows stay live."""
    assert litellm.supports_response_schema(
        model=NATIVE_SCHEMA_MODEL, custom_llm_provider="groq"
    )
    assert not litellm.supports_response_schema(model=MODEL, custom_llm_provider="groq")
    native = _v2(CASES["response_format_schema_native_passthrough"])
    assert native.is_ok()
    assert native.ok["response_format"]["type"] == "json_schema"


def test_top_k_rides_extra_body_then_the_wire() -> None:
    """groq's map packs top_k into extra_body (NOT the cohere/mistral
    top-level passthrough); hh merges extra_body into the wire body, so the
    emission is wire-equivalent — the wire-prove rule."""
    request = copy.deepcopy(CASES["top_k_extra_body_merged"])
    model = request.pop("model")
    optional_params = get_optional_params(
        model=model, custom_llm_provider="groq", stream=None, top_k=7
    )
    assert optional_params["extra_body"] == {"top_k": 7}
    result = _v2(CASES["top_k_extra_body_merged"])
    assert result.is_ok() and result.ok["top_k"] == 7
