"""Differential parity for the ollama_chat response path (wave 3).

v1 side: ``OllamaChatConfig.transform_response`` mutating a pre-allocated
``ModelResponse`` — ``model = ollama_chat/{REQUEST model}`` (the wire model
key is IGNORED), finish from ``map_finish_reason(done_reason or "stop")``
with the tool_calls override, the message riding VERBATIM into
``Message(**message)`` (extra keys included) after the thinking remap /
think-tag split, usage from prompt_eval_count/eval_count.

v2 side: the ollama_chat parser builds the normalized body on
``ChatResponse.wire``; the seam's ``openai`` construction arm reproduces
v1's assembly. Both sides get the SAME pre-allocated ``ModelResponse(id=...)``
(v1 keeps the ambient chatcmpl id — no wire id exists on this wire).

Minted tool-call ids are normalized in ``_norm``: BOTH sides mint them
inside ``Message(**...)`` validation (v1 during transform_response, v2
during ``to_model_response``), so under the frozen-uuid fixture the values
are counter positions, not parity surface — the mint FACT (bare uuid, every
id-less entry) is pinned by its own test.
"""

import copy
import json
import re
import time

import httpx
import pytest

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.ollama.chat.transformation import OllamaChatConfig
from litellm.types.utils import ModelResponse

from litellm.translation.inbound.openai_chat import parse_request
from litellm.translation.inbound.openai_chat.response import serialize_response
from litellm.translation.providers.ollama_chat import parse_response
from litellm.translation_seam import (
    UsageStyle,
    build_translation_deps,
    to_model_response,
)

MODEL = "llama3.1"
PREFIXED = f"ollama_chat/{MODEL}"
_AMBIENT_ID = "chatcmpl-ollama-diff"

_REQUEST = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}

_BASE = {
    "model": MODEL,
    "created_at": "2025-05-24T02:12:05Z",
    "message": {"role": "assistant", "content": "hello"},
    "done": True,
    "done_reason": "stop",
    "prompt_eval_count": 7,
    "eval_count": 3,
}

_RESPONSES = {
    "plain": _BASE,
    "done_reason_length": {**_BASE, "done_reason": "length"},
    "done_reason_missing_defaults_stop": {
        k: v for k, v in _BASE.items() if k != "done_reason"
    },
    "done_reason_unknown_maps_stop": {**_BASE, "done_reason": "load"},
    "wire_model_ignored": {**_BASE, "model": "OTHER-MODEL"},
    "thinking_field_to_reasoning_content": {
        **_BASE,
        "message": {"role": "assistant", "content": "yes", "thinking": "hmm"},
    },
    "think_tags_split_remainder": {
        **_BASE,
        "message": {"role": "assistant", "content": "<think>hmm</think>yes"},
    },
    "think_tags_unclosed_verbatim": {
        **_BASE,
        "message": {"role": "assistant", "content": "<think>hmm yes"},
    },
    "tool_calls_dict_args_restringified": {
        **_BASE,
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "f", "arguments": {"a": 1, "b": [1, 2]}}}
            ],
        },
    },
    "tool_calls_str_args_verbatim": {
        **_BASE,
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "f", "arguments": '{"a":1}'}}],
        },
    },
    "tool_calls_wire_id_kept": {
        **_BASE,
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "wire-id",
                    "type": "function",
                    "function": {"name": "f", "arguments": {}},
                }
            ],
        },
    },
    "tool_calls_force_finish_over_done_reason": {
        **_BASE,
        "done_reason": "length",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "f", "arguments": {}}}],
        },
    },
    "extra_message_key_rides": {
        **_BASE,
        "message": {"role": "assistant", "content": "x", "weird_key": 5},
    },
    "extra_top_level_key_dropped": {**_BASE, "total_duration": 12345},
}

# v1 RAISES out of transform_response (TypeError / ValueError / the eager
# token_counter default); v2 must be a loud typed error.
_LOUD = {
    "non_object_body": ([1, 2], "not an object"),
    "missing_message": (
        {k: v for k, v in _BASE.items() if k != "message"},
        "message",
    ),
    "null_message": ({**_BASE, "message": None}, "message"),
    "non_str_content": (
        {**_BASE, "message": {"role": "assistant", "content": 5}},
        "not a string",
    ),
    "null_content_eager_token_counter": (
        {**_BASE, "message": {"role": "assistant", "content": None}},
        "token_counter",
    ),
    "content_key_missing": (
        {**_BASE, "message": {"role": "assistant"}},
        "missing or null",
    ),
    "null_content_with_thinking": (
        {**_BASE, "message": {"role": "assistant", "thinking": "h", "content": None}},
        "missing or null",
    ),
    "tool_call_non_object_entry": (
        {
            **_BASE,
            "message": {"role": "assistant", "content": "", "tool_calls": ["x"]},
        },
        "not an object",
    ),
    "tool_call_missing_function": (
        {
            **_BASE,
            "message": {"role": "assistant", "content": "", "tool_calls": [{"id": "c"}]},
        },
        "function",
    ),
    "unhashable_done_reason": (
        {**_BASE, "done_reason": {"x": 1}},
        "unhashable",
    ),
}

# v1 SERVES these; v2 falls back typed so v1 keeps serving.
_V1_SERVES_FALLBACKS = {
    "missing_prompt_eval_count_token_counter": (
        {k: v for k, v in _BASE.items() if k != "prompt_eval_count"},
        "token_counter",
    ),
    "missing_eval_count_token_counter": (
        {k: v for k, v in _BASE.items() if k != "eval_count"},
        "token_counter",
    ),
    "function_call_done_reason_outside_ir": (
        {**_BASE, "done_reason": "function_call"},
        "function_call",
    ),
}


def _v1_model_response(raw) -> dict:
    logging = Logging(
        model=MODEL,
        messages=copy.deepcopy(_REQUEST["messages"]),
        stream=False,
        call_type="completion",
        start_time=time.time(),
        litellm_call_id="diff-ollama-response",
        function_id="diff-ollama-response",
    )
    response = httpx.Response(
        200,
        json=copy.deepcopy(raw),
        request=httpx.Request("POST", "http://localhost:11434/api/chat"),
    )
    return (
        OllamaChatConfig()
        .transform_response(
            model=MODEL,
            raw_response=response,
            model_response=ModelResponse(id=_AMBIENT_ID),
            logging_obj=logging,
            request_data={},
            messages=copy.deepcopy(_REQUEST["messages"]),
            optional_params={},
            litellm_params={},
            encoding=None,
        )
        .model_dump()
    )


def _v2_parse(raw):
    parsed = parse_request(copy.deepcopy(_REQUEST))
    assert parsed.is_ok(), parsed.error.summary
    return parse_response(copy.deepcopy(raw), parsed.ok)


def _v2_with_style(raw, style: UsageStyle) -> dict:
    response = _v2_parse(raw)
    assert response.is_ok(), response.error.summary
    body = serialize_response(response.ok, build_translation_deps(), "openai")
    return to_model_response(
        body, ModelResponse(id=_AMBIENT_ID), usage_style=style
    ).model_dump()


def _v2_model_response(raw) -> dict:
    return _v2_with_style(raw, "openai")


_UUID_RE = re.compile(
    r"^(call_)?[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


def _norm(payload: dict) -> str:
    """Minted tool ids are envelope nondeterminism (both sides mint inside
    Message validation at different frozen-counter positions): uuid-shaped
    ids normalize to a sentinel, wire-carried ids stay verbatim."""
    normalized = copy.deepcopy(payload)
    for choice in normalized.get("choices", []):
        for call in (choice.get("message") or {}).get("tool_calls") or []:
            if isinstance(call.get("id"), str) and _UUID_RE.match(call["id"]):
                call["id"] = "MINTED"
    return json.dumps(normalized, sort_keys=True, default=str)


@pytest.mark.parametrize("name", sorted(_RESPONSES))
def test_v2_response_matches_v1(name: str, frozen_ambient) -> None:
    raw = _RESPONSES[name]
    assert _norm(_v2_model_response(raw)) == _norm(_v1_model_response(raw))


@pytest.mark.parametrize("name", sorted(_RESPONSES))
def test_prefix_and_envelope_quirks(name: str, frozen_ambient) -> None:
    """The response model is ollama_chat/{REQUEST model} (wire model
    ignored), the ambient chatcmpl id is kept, and created is the frozen
    ambient stamp (v1 re-stamps int(time.time()) — the same value)."""
    v1 = _v1_model_response(_RESPONSES[name])
    v2 = _v2_model_response(_RESPONSES[name])
    for dump in (v1, v2):
        assert dump["model"] == PREFIXED
        assert dump["id"] == _AMBIENT_ID
        assert dump["created"] == 1718064000


@pytest.mark.parametrize("name", sorted(_LOUD))
def test_loud_shapes_error_on_both_sides(name: str, frozen_ambient) -> None:
    raw, fragment = _LOUD[name]
    result = _v2_parse(raw)
    assert result.is_error(), f"{name} unexpectedly parsed"
    assert fragment in result.error.summary, result.error.summary
    with pytest.raises(Exception):
        _v1_model_response(copy.deepcopy(raw))


@pytest.mark.parametrize("name", sorted(_V1_SERVES_FALLBACKS))
def test_v1_serves_fallback_rows(name: str, frozen_ambient) -> None:
    raw, fragment = _V1_SERVES_FALLBACKS[name]
    result = _v2_parse(raw)
    assert result.is_error(), f"{name} unexpectedly parsed"
    assert fragment in result.error.summary, result.error.summary
    v1 = _v1_model_response(copy.deepcopy(raw))
    assert v1["choices"], f"{name}: v1 stopped serving; re-decide the fallback"


def test_minted_tool_ids_are_bare_uuids_on_both_sides(frozen_ambient) -> None:
    """Both sides mint the missing tool id inside Message(**...) validation:
    a bare uuid (no call_ prefix — that prefix exists only in the dormant
    json+function_name synth arm)."""
    raw = _RESPONSES["tool_calls_dict_args_restringified"]
    for dump in (_v1_model_response(raw), _v2_model_response(raw)):
        minted = dump["choices"][0]["message"]["tool_calls"][0]["id"]
        assert _UUID_RE.match(minted) and not minted.startswith("call_")


def test_wire_tool_id_survives_both_sides(frozen_ambient) -> None:
    raw = _RESPONSES["tool_calls_wire_id_kept"]
    for dump in (_v1_model_response(raw), _v2_model_response(raw)):
        assert dump["choices"][0]["message"]["tool_calls"][0]["id"] == "wire-id"


def test_dict_args_restringified_with_default_separators(frozen_ambient) -> None:
    raw = _RESPONSES["tool_calls_dict_args_restringified"]
    for dump in (_v1_model_response(raw), _v2_model_response(raw)):
        arguments = dump["choices"][0]["message"]["tool_calls"][0]["function"][
            "arguments"
        ]
        assert arguments == '{"a": 1, "b": [1, 2]}'


def test_finish_map_mirror_matches_v1_table() -> None:
    """FINISH_REASON_MIRROR is the one in-package mirror of v1's
    core_helpers._FINISH_REASON_MAP; key-for-key drift would silently
    diverge served finish reasons."""
    from litellm.litellm_core_utils.core_helpers import _FINISH_REASON_MAP

    from litellm.translation.providers.ollama_chat.response import (
        FINISH_REASON_MIRROR,
    )

    assert dict(FINISH_REASON_MIRROR) == dict(_FINISH_REASON_MAP)


def test_wrong_construction_arm_diverges_and_the_table_pins_it(
    frozen_ambient,
) -> None:
    """The cohere no-id template: the ollama body deliberately carries NO id,
    so the correct cdr ("openai") arm keeps the ambient chatcmpl id exactly
    like v1's fresh-ModelResponse mutation, while the wrong "openai_like"
    arm (ModelResponse(**json)) ignores the pre-allocated envelope and mints
    a fresh id."""
    from litellm.translation.engine.pipeline import OWN_MODULE_RESPONSE_STYLES

    assert OWN_MODULE_RESPONSE_STYLES["ollama_chat"] == "openai"
    raw = _RESPONSES["plain"]
    v1 = _v1_model_response(raw)
    correct = _v2_with_style(raw, OWN_MODULE_RESPONSE_STYLES["ollama_chat"])
    wrong = _v2_with_style(raw, "openai_like")
    assert _norm(correct) == _norm(v1)
    assert correct["id"] == _AMBIENT_ID
    assert wrong["id"] != _AMBIENT_ID
    assert _norm(wrong) != _norm(v1), (
        "the construction arms stopped diverging on the envelope id — "
        "the F1 gate lost its discriminator; re-decide before relying on it"
    )
