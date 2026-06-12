"""Differential parity for the fireworks_ai response path (httpx; the ONE
wave-2b-alpha prefixing parser so far).

v1's ``transform_response`` REPLACES the base: ``ModelResponse(**json)`` —
the OpenAILike DIRECT construction (NOT cdr; different pydantic dump, no
stop->tool_calls rewrite) — then ``model = "fireworks_ai/" + WIRE model``
and the tool-calls-in-content repair. These rows run that exact v1 chain
against v2's direct-parser policy with the seam's ``openai_like``
construction arm (the compat_httpx RESPONSE_STYLES shape — the future fork
MUST use it, never the "openai" arm or a seam preset). The repair shape and
the non-string wire model are two-sided fallback rows.
"""

import copy
import json

import pydantic
import pytest

from litellm.types.utils import ModelResponse

from litellm.translation.inbound.openai_chat import parse_request
from litellm.translation.inbound.openai_chat.response import serialize_response
from litellm.translation.providers.fireworks_ai import parse_response
from litellm.translation_seam import build_translation_deps, to_model_response

from ._own_module_corpus import run_v1_response_transform

REQUEST_MODEL = "accounts/fireworks/models/deepseek-v3p2"
WIRE_MODEL = "accounts/fireworks/models/deepseek-v3p2"

_REQUEST = {
    "model": REQUEST_MODEL,
    "messages": [{"role": "user", "content": "hi"}],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ],
}

_RESPONSES = {
    "text": {
        "id": "fw-1",
        "object": "chat.completion",
        "created": 1718000000,
        "model": WIRE_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    },
    "tool_calls": {
        "id": "fw-2",
        "object": "chat.completion",
        "created": 1718000000,
        "model": WIRE_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
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
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 5, "total_tokens": 14},
    },
    # tool-bearing request whose content does NOT parse as a requested
    # function: NO repair fires in v1; the verbatim ride serves
    "plain_content_with_tools": {
        "id": "fw-3",
        "object": "chat.completion",
        "created": 1718000000,
        "model": WIRE_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "just text"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    },
}

_REPAIR_BODY = {
    "id": "fw-repair",
    "object": "chat.completion",
    "created": 1718000000,
    "model": WIRE_MODEL,
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": '{"name": "get_weather", "arguments": "{\\"city\\": \\"Paris\\"}"}',
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
}

_NON_STRING_MODEL_BODY = {**_RESPONSES["text"], "model": 7}


def _v1_model_response(raw: dict) -> dict:
    return run_v1_response_transform(
        "fireworks_ai",
        raw,
        REQUEST_MODEL,
        optional_params={"tools": copy.deepcopy(_REQUEST["tools"])},
    ).model_dump()


def _v2_model_response(raw: dict) -> dict:
    parsed = parse_request(copy.deepcopy(_REQUEST))
    assert parsed.is_ok(), parsed.error.summary
    response = parse_response(copy.deepcopy(raw), parsed.ok)
    assert response.is_ok(), response.error.summary
    body = serialize_response(response.ok, build_translation_deps(), "openai")
    # fireworks is DIRECT construction: the seam fork must use the
    # "openai_like" arm (never "openai", never a model preset)
    return to_model_response(
        body, ModelResponse(), usage_style="openai_like"
    ).model_dump()


def _norm(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


@pytest.mark.parametrize("name", sorted(_RESPONSES))
def test_v2_response_matches_v1(name: str, frozen_ambient) -> None:
    raw = _RESPONSES[name]
    assert _norm(_v2_model_response(raw)) == _norm(_v1_model_response(raw))


@pytest.mark.parametrize("name", sorted(_RESPONSES))
def test_fireworks_prefix_on_the_response_model(name: str, frozen_ambient) -> None:
    raw = _RESPONSES[name]
    v1 = _v1_model_response(raw)
    v2 = _v2_model_response(raw)
    assert v1["model"] == f"fireworks_ai/{WIRE_MODEL}"
    assert v2["model"] == f"fireworks_ai/{WIRE_MODEL}"


def test_repair_shape_falls_back_and_v1_serves_its_repair(frozen_ambient) -> None:
    """Two-sided: v1 synthesizes a tool_call (uuid4 id, content -> None);
    v2 fails closed naming the repair (the pure parser cannot mint ids)."""
    parsed = parse_request(copy.deepcopy(_REQUEST))
    assert parsed.is_ok()
    result = parse_response(copy.deepcopy(_REPAIR_BODY), parsed.ok)
    assert result.is_error()
    assert "uuid4" in result.error.summary, result.error.summary
    v1 = _v1_model_response(_REPAIR_BODY)
    message = v1["choices"][0]["message"]
    assert message["content"] is None
    assert message["tool_calls"][0]["function"]["name"] == "get_weather"
    assert message["tool_calls"][0]["id"]  # freshly minted


def test_repair_skips_the_reserved_json_tool_name(frozen_ambient) -> None:
    """content parsing to the reserved json_tool_call name is NOT repaired
    by v1 (served verbatim) — and v2 serves it too (the trigger excludes the
    reserved name)."""
    body = copy.deepcopy(_REPAIR_BODY)
    body["choices"][0]["message"][
        "content"
    ] = '{"name": "json_tool_call", "arguments": "{}"}'
    request = {**copy.deepcopy(_REQUEST)}
    request["tools"] = [
        {
            "type": "function",
            "function": {
                "name": "json_tool_call",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    parsed = parse_request(request)
    assert parsed.is_ok()
    result = parse_response(copy.deepcopy(body), parsed.ok)
    assert result.is_ok(), result.error.summary
    v1 = run_v1_response_transform(
        "fireworks_ai", body, REQUEST_MODEL, optional_params={"tools": request["tools"]}
    ).model_dump()
    assert (
        v1["choices"][0]["message"]["content"]
        == body["choices"][0]["message"]["content"]
    )


def test_non_string_wire_model_falls_back_where_v1_raises(frozen_ambient) -> None:
    """v1's ModelResponse(**json) raises pydantic ValidationError BEFORE the
    prefix rewrite; the direct parser's F2 arm fails closed (the family
    rule: v2 never serves what v1 raises on)."""
    with pytest.raises(pydantic.ValidationError):
        _v1_model_response(_NON_STRING_MODEL_BODY)
    parsed = parse_request(copy.deepcopy(_REQUEST))
    assert parsed.is_ok()
    result = parse_response(copy.deepcopy(_NON_STRING_MODEL_BODY), parsed.ok)
    assert result.is_error()
    assert "non-string wire model" in result.error.summary


def test_missing_wire_model_keeps_none_on_both_sides(frozen_ambient) -> None:
    """v1's `if response.model is not None` arm: a body WITHOUT a model key
    constructs model=None and skips the prefix — v2's rewrite policy returns
    None and leaves the body untouched."""
    body = {k: v for k, v in _RESPONSES["text"].items() if k != "model"}
    v1 = _v1_model_response(body)
    v2 = _v2_model_response(body)
    assert v1["model"] is None
    assert v2["model"] is None
    assert _norm(v2) == _norm(v1)
