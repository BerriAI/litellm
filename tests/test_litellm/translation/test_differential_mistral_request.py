"""Differential parity for the mistral chat request path (wave-2b-beta).

v1 side, invoked the way main.py's mistral elif runs:
``get_optional_params(custom_llm_provider="mistral")`` (explicit map arms,
drop_params threaded) with completion()'s ``stream=None`` default, the
handler's ``extra_body`` pop (seed -> random_seed is envelope-merged there,
but ``seed`` is a parse-level fallback in v2 anyway), then
``MistralConfig.transform_request`` — the reasoning-prompt injection for
magistral models and the two-branch ``_transform_messages``.

Served deltas pinned IDENTICAL: mct -> max_tokens, tool_choice required ->
any with the DICT form silently dropped, tools ``$id``/``$schema`` strip
(depth-capped; additionalProperties/strict KEPT — the v1 docstring lies),
top_k verbatim top-level (wire-proven), the text-list flatten, the
MistralToolCallMessage shape, empty-assistant removal, per-message None
strip, the name matrix (user names dropped, tool names kept -> fallback),
and the image branch's verbatim base-transform passthrough.
"""

import copy
import json

import pytest

from litellm.exceptions import UnsupportedParamsError
from litellm.llms.mistral.chat.transformation import MistralConfig
from litellm.utils import get_optional_params

from litellm.translation.engine.pipeline import translate_chat_request
from litellm.translation_seam import build_translation_deps

MODEL = "mistral-large-latest"
_U = [{"role": "user", "content": "hi"}]

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
    "top_k_wire_proven": {"model": MODEL, "messages": _U, "top_k": 7},
    "stream_true": {"model": MODEL, "messages": _U, "stream": True},
    "explicit_stream_false_dropped": {"model": MODEL, "messages": _U, "stream": False},
    "tools_auto": {
        "model": MODEL,
        "messages": _U,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "d",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "tool_choice": "auto",
    },
    "tool_choice_required_to_any": {
        "model": MODEL,
        "messages": _U,
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
        "tool_choice": "required",
    },
    "tool_choice_dict_dropped": {
        "model": MODEL,
        "messages": _U,
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
        "tool_choice": {"type": "function", "function": {"name": "f"}},
    },
    "tools_schema_refs_stripped": {
        "model": MODEL,
        "messages": _U,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "strict": True,
                    "parameters": {
                        "$id": "root",
                        "$schema": "draft-07",
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"a": {"$id": "nested", "type": "string"}},
                    },
                },
            }
        ],
    },
    "tools_schema_refs_stripped_4_levels_deep": {
        # verifier-wave2b-beta F2: an ordinary $id four object levels inside
        # parameters — v1's call site passes
        # max_depth=DEFAULT_MAX_RECURSE_DEPTH (=100), NOT the signature
        # default 10 the old _REFS_MAX_DEPTH pinned; both sides must strip.
        "model": MODEL,
        "messages": _U,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "parameters": {
                        "$id": "root",
                        "type": "object",
                        "properties": {
                            "a": {
                                "$id": "l1",
                                "type": "object",
                                "properties": {
                                    "b": {
                                        "$id": "l2",
                                        "type": "object",
                                        "properties": {
                                            "c": {"$id": "l3", "type": "string"}
                                        },
                                    }
                                },
                            }
                        },
                    },
                },
            }
        ],
    },
    "response_format_json_object": {
        "model": MODEL,
        "messages": _U,
        "response_format": {"type": "json_object"},
    },
    "parallel_tool_calls": {
        "model": MODEL,
        "messages": _U,
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
        "parallel_tool_calls": True,
    },
    "flatten_multi_text_list": {
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
    "assistant_tool_history": {
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
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
        ],
    },
    "empty_assistant_removed": {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "again"},
        ],
    },
    "user_name_dropped_both_sides": {
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi", "name": "bob"}],
    },
    "system_first": {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hi"},
        ],
    },
    "image_branch_verbatim": {
        "model": "pixtral-12b",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what"},
                    {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
                ],
            }
        ],
    },
    "image_branch_with_tool_history": {
        "model": "pixtral-12b",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what"},
                    {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
                ],
            },
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
            {"role": "tool", "tool_call_id": "c1", "content": "res"},
        ],
    },
}

# v1 RAISES UnsupportedParamsError; v2 must be a typed fallback.
V1_RAISES = {
    "frequency_penalty": (
        {"model": MODEL, "messages": _U, "frequency_penalty": 0.1},
        "frequency_penalty",
    ),
    "presence_penalty": (
        {"model": MODEL, "messages": _U, "presence_penalty": 0.1},
        "presence_penalty",
    ),
    "n": ({"model": MODEL, "messages": _U, "n": 2}, "n"),
    "logprobs": ({"model": MODEL, "messages": _U, "logprobs": True}, "logprobs"),
    "reasoning_effort_non_magistral": (
        {"model": MODEL, "messages": _U, "reasoning_effort": "high"},
        "reasoning_effort",
    ),
    "thinking_non_magistral": (
        {"model": MODEL, "messages": _U, "thinking": {"type": "enabled"}},
        "thinking",
    ),
}

# v1 SERVES these; v2 falls back typed so v1 keeps serving its own behavior.
V1_SERVES_FALLBACKS = {
    "user_silent_drop": (
        {"model": MODEL, "messages": _U, "user": "u1"},
        "user",
    ),
    "seed_random_seed_extra_body": (
        {"model": MODEL, "messages": _U, "seed": 42},
        "seed",
    ),
    "magistral_reasoning_prompt_injection": (
        {
            "model": "magistral-medium-latest",
            "messages": [{"role": "user", "content": "solve"}],
            "reasoning_effort": "high",
        },
        "reasoning",
    ),
    "magistral_thinking_prompt_injection": (
        {
            "model": "magistral-medium-latest",
            "messages": [{"role": "user", "content": "solve"}],
            "thinking": {"type": "enabled", "budget_tokens": 1024},
        },
        "thinking",
    ),
    "tool_name_kept_by_v1": (
        {
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
                {"role": "tool", "tool_call_id": "c1", "name": "f", "content": "r"},
            ],
        },
        "tool-message name",
    ),
    "image_branch_name_forwarded": (
        {
            "model": "pixtral-12b",
            "messages": [
                {
                    "role": "user",
                    "name": "bob",
                    "content": [
                        {"type": "text", "text": "what"},
                        {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
                    ],
                }
            ],
        },
        "image/file",
    ),
    "string_stop_verbatim": (
        {"model": MODEL, "messages": _U, "stop": "x"},
        "string-form stop",
    ),
    "single_text_list_flatten": (
        # v1 flattens it and the IR collapse + flatten WOULD equal v1, but
        # the shared single-text-list arm is conservative (the sambanova
        # precedent); v1 serves its flatten.
        {
            "model": MODEL,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "only"}]}
            ],
        },
        "single-text content list",
    ),
}


def run_v1_request_transform(case: dict) -> dict:
    request = copy.deepcopy(case)
    model = request.pop("model")
    messages = request.pop("messages")
    optional_params = get_optional_params(
        model=model,
        custom_llm_provider="mistral",
        messages=copy.deepcopy(messages),
        stream=request.pop("stream", None),
        **request,
    )
    optional_params.pop("extra_body", None)
    return MistralConfig().transform_request(
        model=model,
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )


def _v2(case: dict):
    return translate_chat_request(
        copy.deepcopy(case), "mistral", build_translation_deps()
    )


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


@pytest.mark.parametrize("name", sorted(V1_SERVES_FALLBACKS))
def test_v1_serves_fallback_rows(name: str) -> None:
    case, fragment = V1_SERVES_FALLBACKS[name]
    result = _v2(case)
    assert result.is_error(), f"{name} unexpectedly translated: {result.ok!r}"
    assert fragment in result.error.summary, result.error.summary
    run_v1_request_transform(case)


def test_top_k_rides_the_wire_top_level() -> None:
    """The wire-prove rule: mistral's top_k is NOT the extra_body arm — the
    generic passthrough places it top-level and the body carries it
    (researcher-4 listed top_k as a RAISE; the probe refutes that)."""
    v1_body = run_v1_request_transform(CASES["top_k_wire_proven"])
    assert v1_body["top_k"] == 7
    assert "extra_body" not in v1_body
    result = _v2(CASES["top_k_wire_proven"])
    assert result.is_ok() and result.ok["top_k"] == 7


def test_explicit_stream_false_never_reaches_the_wire() -> None:
    """Unlike the compat families, mistral's map only copies stream=True —
    explicit False is dropped by v1 itself, so v2 SERVES the shape (the IR's
    absent-vs-false collapse IS v1's behavior; no guard arm)."""
    v1_body = run_v1_request_transform(CASES["explicit_stream_false_dropped"])
    assert "stream" not in v1_body
    result = _v2(CASES["explicit_stream_false_dropped"])
    assert result.is_ok() and "stream" not in result.ok


def test_schema_refs_strip_keeps_additional_properties_and_strict() -> None:
    """The code-not-docstring pin: only $id/$schema are removed."""
    result = _v2(CASES["tools_schema_refs_stripped"])
    assert result.is_ok()
    function = result.ok["tools"][0]["function"]
    assert function["strict"] is True
    assert function["parameters"]["additionalProperties"] is False
    assert "$id" not in function["parameters"]
    assert "$id" not in function["parameters"]["properties"]["a"]


def _nested_id_schema(levels: int) -> dict:
    nest: dict = {"$id": "deep", "type": "string"}
    for _ in range(levels):
        nest = {"type": "object", "child": nest}
    return {"type": "object", "outer": nest}


@pytest.mark.parametrize("levels", [40, 101])
def test_refs_depth_cap_is_v1s_default_max_recurse_depth(levels: int) -> None:
    """verifier-wave2b-beta F2 + the drift gate: v1 strips at
    _remove_json_schema_refs(max_depth=DEFAULT_MAX_RECURSE_DEPTH) — 40
    levels strip BOTH sides (the old cap of 10 left v2 keeping them), and
    past the constant both sides keep the key. v2 imports the same
    litellm.constants symbol, pinned here against v1's constant at HEAD so
    neither the call-site argument nor the constant can drift unseen."""
    from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH

    from litellm.translation.providers.mistral import serialize as mistral_serialize

    assert DEFAULT_MAX_RECURSE_DEPTH == 100
    assert (
        mistral_serialize.DEFAULT_MAX_RECURSE_DEPTH is DEFAULT_MAX_RECURSE_DEPTH
    ), "the serializer stopped reading v1's constant; re-pin the cap"
    case = {
        "model": MODEL,
        "messages": _U,
        "tools": [
            {
                "type": "function",
                "function": {"name": "f", "parameters": _nested_id_schema(levels)},
            }
        ],
    }
    v1_body = run_v1_request_transform(case)
    result = _v2(case)
    assert result.is_ok(), result.error.summary
    assert _norm(result.ok) == _norm(v1_body)
    deep = result.ok["tools"][0]["function"]["parameters"]
    for _ in range(levels + 1):
        deep = deep["outer"] if "outer" in deep else deep["child"]
    stripped = "$id" not in deep
    assert stripped == (levels < 96), "the cap boundary moved; re-derive"


def test_empty_text_list_falls_back_v1_keeps_list_form() -> None:
    """v1's flatten only assigns a TRUTHY join: a list flattening to ""
    keeps its LIST form on the wire, which the IR cannot reproduce."""
    case = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ""},
                    {"type": "text", "text": ""},
                ],
            }
        ],
    }
    result = _v2(case)
    assert result.is_error()
    assert "empty" in result.error.summary
    v1_body = run_v1_request_transform(case)
    assert isinstance(v1_body["messages"][0]["content"], list)
