"""Differential parity for the snowflake response path (httpx; prefixed).

v1's ``transform_response`` rewrites choices[0]'s ``content_list`` (text
concat + tool_use -> OpenAI tool_calls with ``json.dumps(input)``
arguments), deletes it, builds ``ModelResponse(**json)`` — the OpenAILike
DIRECT construction — and overwrites ``model = "snowflake/" + (wire model
or "")``. The request model rides ``_hidden_params["model"]`` (envelope,
dump-invisible — pinned below as a fork obligation). These rows run that
exact v1 chain against v2's pre-rewrite + direct parser with the seam's
``openai_like`` construction arm.
"""

import copy
import json

import pydantic
import pytest

from litellm.types.utils import ModelResponse

from litellm.translation.inbound.openai_chat import parse_request
from litellm.translation.inbound.openai_chat.response import serialize_response
from litellm.translation.providers.snowflake import parse_response
from litellm.translation_seam import (
    UsageStyle,
    build_translation_deps,
    to_model_response,
)

from ._own_module_corpus import run_v1_response_transform

REQUEST_MODEL = "claude-3-5-sonnet"
WIRE_MODEL = "claude-3-5-sonnet"

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
    "content_list_text": {
        "id": "sf-1",
        "object": "chat.completion",
        "created": 1718000000,
        "model": WIRE_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content_list": [
                        {"type": "text", "text": "I will "},
                        {"type": "text", "text": "check."},
                    ],
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    },
    "content_list_tool_use": {
        "id": "sf-2",
        "object": "chat.completion",
        "created": 1718000000,
        "model": WIRE_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content_list": [
                        {"type": "text", "text": "Checking."},
                        {
                            "type": "tool_use",
                            "tool_use": {
                                "tool_use_id": "tooluse_1",
                                "name": "get_weather",
                                "input": {"city": "Paris", "unit": "C"},
                            },
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 5, "total_tokens": 14},
    },
    # bodies WITHOUT content_list ride verbatim (plain content key)
    "plain_content": {
        "id": "sf-3",
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
}

_NON_STRING_MODEL_BODY = {**_RESPONSES["plain_content"], "model": 7}


def _v1_full(raw: dict) -> ModelResponse:
    return run_v1_response_transform("snowflake", raw, REQUEST_MODEL)


def _v1_model_response(raw: dict) -> dict:
    return _v1_full(raw).model_dump()


def _v2_with_style(raw: dict, style: UsageStyle) -> dict:
    parsed = parse_request(copy.deepcopy(_REQUEST))
    assert parsed.is_ok(), parsed.error.summary
    response = parse_response(copy.deepcopy(raw), parsed.ok)
    assert response.is_ok(), response.error.summary
    body = serialize_response(response.ok, build_translation_deps(), "openai")
    return to_model_response(body, ModelResponse(), usage_style=style).model_dump()


def _v2_model_response(raw: dict) -> dict:
    # snowflake is DIRECT construction: the seam fork must use the
    # "openai_like" arm (never "openai", never a model preset) — the truth is
    # pipeline.OWN_MODULE_RESPONSE_STYLES["snowflake"], divergence-pinned below
    return _v2_with_style(raw, "openai_like")


def _norm(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


@pytest.mark.parametrize("name", sorted(_RESPONSES))
def test_v2_response_matches_v1(name: str, frozen_ambient) -> None:
    raw = _RESPONSES[name]
    assert _norm(_v2_model_response(raw)) == _norm(_v1_model_response(raw))


@pytest.mark.parametrize("name", sorted(_RESPONSES))
def test_snowflake_prefix_on_the_response_model(name: str, frozen_ambient) -> None:
    raw = _RESPONSES[name]
    v1 = _v1_model_response(raw)
    v2 = _v2_model_response(raw)
    assert v1["model"] == f"snowflake/{WIRE_MODEL}"
    assert v2["model"] == f"snowflake/{WIRE_MODEL}"


def test_tool_use_arguments_keep_v1_json_dumps_bytes(frozen_ambient) -> None:
    """The arguments string is json.dumps(input) with DEFAULT separators
    (space after colon and comma) — byte-pinned, because the dump comparison
    normalizes whole payloads but the arguments value is itself a string."""
    v2 = _v2_model_response(_RESPONSES["content_list_tool_use"])
    arguments = v2["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    assert arguments == '{"city": "Paris", "unit": "C"}'
    v1 = _v1_model_response(_RESPONSES["content_list_tool_use"])
    assert (
        v1["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        == arguments
    )


def test_missing_wire_model_prefixes_the_empty_string(frozen_ambient) -> None:
    """v1: "snowflake/" + (None or "") == "snowflake/" — pinned both sides."""
    body = {k: v for k, v in _RESPONSES["plain_content"].items() if k != "model"}
    v1 = _v1_model_response(body)
    v2 = _v2_model_response(body)
    assert v1["model"] == "snowflake/"
    assert v2["model"] == "snowflake/"
    assert _norm(v2) == _norm(v1)


def test_non_string_wire_model_falls_back_where_v1_raises(frozen_ambient) -> None:
    """v1's ModelResponse(**json) raises pydantic ValidationError BEFORE the
    prefix overwrite; the direct parser's F2 arm fails closed."""
    with pytest.raises(pydantic.ValidationError):
        _v1_model_response(_NON_STRING_MODEL_BODY)
    parsed = parse_request(copy.deepcopy(_REQUEST))
    assert parsed.is_ok()
    result = parse_response(copy.deepcopy(_NON_STRING_MODEL_BODY), parsed.ok)
    assert result.is_error()
    assert "non-string wire model" in result.error.summary


def test_wrong_construction_arm_diverges_and_the_table_pins_it(
    frozen_ambient,
) -> None:
    """critic-wave2b-alpha MAJOR-4 (the snowflake half — see the fireworks_ai
    twin for the full rationale): the obligated "openai_like" style is
    machine-readable in pipeline.OWN_MODULE_RESPONSE_STYLES and the wrong
    (openai/cdr) arm provably diverges from v1 on a parser-admissible body —
    cdr's enumerate rebuild rewrites a verbatim wire choice index 5 to 0
    while v1's ModelResponse(**json) keeps it."""
    from litellm.translation.engine.pipeline import OWN_MODULE_RESPONSE_STYLES

    assert OWN_MODULE_RESPONSE_STYLES["snowflake"] == "openai_like"
    raw = copy.deepcopy(_RESPONSES["plain_content"])
    raw["choices"][0]["index"] = 5
    v1 = _v1_model_response(raw)
    correct = _v2_with_style(raw, OWN_MODULE_RESPONSE_STYLES["snowflake"])
    wrong = _v2_with_style(raw, "openai")
    assert _norm(correct) == _norm(v1)
    assert correct["choices"][0]["index"] == 5
    assert wrong["choices"][0]["index"] == 0
    assert _norm(wrong) != _norm(v1), (
        "the construction arms stopped diverging on the index-rewrite body — "
        "the MAJOR-4 gate lost its discriminator; re-decide before relying on it"
    )


def test_v1_hidden_request_model_is_the_fork_obligation(frozen_ambient) -> None:
    """v1 stores the REQUEST model in _hidden_params['model'] (the cost
    calculator's key); dump-invisible, so the future fork must set it —
    recorded in CLAUDE.md."""
    v1 = _v1_full(_RESPONSES["plain_content"])
    assert v1._hidden_params.get("model") == REQUEST_MODEL
