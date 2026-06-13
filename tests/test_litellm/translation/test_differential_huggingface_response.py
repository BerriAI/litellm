"""Differential parity for the huggingface response path (httpx, NO prefix).

HuggingFaceChatConfig has no ``transform_response`` override: the inherited
base GPT transform is LIVE on the dedicated elif and runs
``convert_to_model_response_object`` over a fresh ``ModelResponse``
(model=None) — the bare-wire-model cometapi/xai shape. These rows run that
exact v1 chain against v2's shared openai parser, byte-identical dumps,
and pin that no ``huggingface/`` prefix ever appears.
"""

import copy
import json

import pydantic
import pytest

from litellm.types.utils import ModelResponse

from litellm.translation.inbound.openai_chat import parse_request
from litellm.translation.inbound.openai_chat.response import serialize_response
from litellm.translation.providers.huggingface import parse_response
from litellm.translation_seam import (
    UsageStyle,
    build_translation_deps,
    to_model_response,
)

from ._own_module_corpus import run_v1_response_transform

_REQUEST = {
    "model": "meta-llama/Llama-3.3-70B-Instruct",
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
        "id": "chatcmpl-hf1",
        "object": "chat.completion",
        "created": 1718000000,
        "model": "meta-llama/Llama-3.3-70B-Instruct",
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
        "id": "chatcmpl-hf2",
        "object": "chat.completion",
        "created": 1718000000,
        "model": "meta-llama/Llama-3.3-70B-Instruct",
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
    # reasoning models served via HF endpoints carry reasoning_content on
    # the message — the unknown-key verbatim ride
    "reasoning_content": {
        "id": "chatcmpl-hf3",
        "object": "chat.completion",
        "created": 1718000000,
        "model": "deepseek-ai/DeepSeek-R1",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning_content": "step by step",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 9,
            "total_tokens": 20,
            "completion_tokens_details": {"reasoning_tokens": 6},
        },
    },
}


def _v1_model_response(raw: dict) -> dict:
    return run_v1_response_transform(
        "huggingface", raw, "meta-llama/Llama-3.3-70B-Instruct"
    ).model_dump()


def _v2_with_style(raw: dict, style: UsageStyle) -> dict:
    parsed = parse_request(copy.deepcopy(_REQUEST))
    assert parsed.is_ok(), parsed.error.summary
    response = parse_response(copy.deepcopy(raw), parsed.ok)
    assert response.is_ok(), response.error.summary
    body = serialize_response(response.ok, build_translation_deps(), "openai")
    return to_model_response(body, ModelResponse(), usage_style=style).model_dump()


def _v2_model_response(raw: dict) -> dict:
    # huggingface rides the cdr arm — the truth is
    # pipeline.OWN_MODULE_RESPONSE_STYLES["huggingface"], divergence-pinned below
    return _v2_with_style(raw, "openai")


def _norm(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


@pytest.mark.parametrize("name", sorted(_RESPONSES))
def test_v2_response_matches_v1(name: str, frozen_ambient) -> None:
    raw = _RESPONSES[name]
    assert _norm(_v2_model_response(raw)) == _norm(_v1_model_response(raw))


@pytest.mark.parametrize("name", sorted(_RESPONSES))
def test_no_huggingface_prefix_on_the_response_model(name: str, frozen_ambient) -> None:
    raw = _RESPONSES[name]
    v1 = _v1_model_response(raw)
    v2 = _v2_model_response(raw)
    assert v1["model"] == raw["model"]
    assert v2["model"] == raw["model"]
    assert not v2["model"].startswith("huggingface/")


def test_wrong_construction_arm_diverges_and_the_table_pins_it(
    frozen_ambient,
) -> None:
    """verifier-wave2b-final F1: the "openai"-arm OWN_MODULE_RESPONSE_STYLES
    values were enforced by NOTHING — a wrong-arm value flip survived the
    whole suite. The watsonx/groq wrong-arm template INVERTED for a
    cdr-parser member (the v2 parser already cdr-normalizes choices, so the
    openai_like members' index-5 discriminator dies before the seam): a
    FLOAT wire ``created`` rides the normalized body; the correct cdr
    ("openai") arm coerces it via _safe_convert_created_field and serves
    exactly like v1, while the wrong "openai_like" arm
    (ModelResponse(**json)) raises ValidationError on traffic v1 serves —
    if the arms stop diverging here, the raises assert fails: re-decide
    before relying on the pin."""
    from litellm.translation.engine.pipeline import OWN_MODULE_RESPONSE_STYLES

    assert OWN_MODULE_RESPONSE_STYLES["huggingface"] == "openai"
    raw = copy.deepcopy(_RESPONSES["text"])
    raw["created"] = raw["created"] + 0.5
    v1 = _v1_model_response(raw)
    correct = _v2_with_style(raw, OWN_MODULE_RESPONSE_STYLES["huggingface"])
    assert _norm(correct) == _norm(v1)
    assert correct["created"] == _RESPONSES["text"]["created"]
    with pytest.raises(pydantic.ValidationError):
        _v2_with_style(raw, "openai_like")
