"""Differential parity for the openrouter response path (httpx, NO prefix).

``OpenrouterConfig.transform_response`` is super() — the base GPT transform,
LIVE on the dedicated elif, over a FRESH ``ModelResponse`` (model=None; bare
wire model, the xai R4 / deepseek pattern) — plus the ``usage.cost`` ->
``_hidden_params`` post-step. These rows run that exact v1 chain against
v2's shared openai parser: byte-identical dumps (``usage.cost`` and
``cost_details`` ride the unknown-usage-key mirror through BOTH sides), no
``openrouter/`` prefix, and the hidden-params cost header pinned as the
FORK OBLIGATION (the dump cannot see ``_hidden_params``; the future
completion() fork must rebuild the header from the v2 body's usage.cost —
litellm/translation/CLAUDE.md records the obligation).
"""

import copy
import json

import pydantic
import pytest

from litellm.types.utils import ModelResponse

from litellm.translation.inbound.openai_chat import parse_request
from litellm.translation.inbound.openai_chat.response import serialize_response
from litellm.translation.providers.openrouter import parse_response
from litellm.translation_seam import (
    UsageStyle,
    build_translation_deps,
    to_model_response,
)

from ._own_module_corpus import run_v1_response_transform

_REQUEST = {
    "model": "openai/gpt-4o",
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
        "id": "gen-or-1",
        "object": "chat.completion",
        "created": 1718000000,
        "model": "openai/gpt-4o",
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
        "id": "gen-or-2",
        "object": "chat.completion",
        "created": 1718000000,
        "model": "openai/gpt-4o",
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
    # openrouter reasoning responses carry message.reasoning (NOT
    # reasoning_content) — the unknown-message-key ride, untouched by v1
    "message_reasoning": {
        "id": "gen-or-3",
        "object": "chat.completion",
        "created": 1718000000,
        "model": "anthropic/claude-3.7-sonnet",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning": "step by step",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 9, "total_tokens": 20},
    },
    # the usage.include=true echo: cost + cost_details in usage
    "usage_cost": {
        "id": "gen-or-4",
        "object": "chat.completion",
        "created": 1718000000,
        "model": "openai/gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
            "cost": 0.0015,
            "cost_details": {"upstream_inference_cost": None},
        },
    },
}

_COST_HEADER = "llm_provider-x-litellm-response-cost"


def _v1_model_response(raw: dict) -> ModelResponse:
    return run_v1_response_transform("openrouter", raw, "openai/gpt-4o")


def _v2_with_style(raw: dict, style: UsageStyle) -> dict:
    parsed = parse_request(copy.deepcopy(_REQUEST))
    assert parsed.is_ok(), parsed.error.summary
    response = parse_response(copy.deepcopy(raw), parsed.ok)
    assert response.is_ok(), response.error.summary
    body = serialize_response(response.ok, build_translation_deps(), "openai")
    return to_model_response(body, ModelResponse(), usage_style=style).model_dump()


def _v2_model_response(raw: dict) -> dict:
    # openrouter rides the cdr arm — the truth is
    # pipeline.OWN_MODULE_RESPONSE_STYLES["openrouter"], divergence-pinned below
    return _v2_with_style(raw, "openai")


def _norm(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


@pytest.mark.parametrize("name", sorted(_RESPONSES))
def test_v2_response_matches_v1(name: str, frozen_ambient) -> None:
    raw = _RESPONSES[name]
    assert _norm(_v2_model_response(raw)) == _norm(_v1_model_response(raw).model_dump())


@pytest.mark.parametrize("name", sorted(_RESPONSES))
def test_no_openrouter_prefix_on_the_response_model(name: str, frozen_ambient) -> None:
    raw = _RESPONSES[name]
    v1 = _v1_model_response(raw).model_dump()
    v2 = _v2_model_response(raw)
    assert v1["model"] == raw["model"]
    assert v2["model"] == raw["model"]
    assert not v2["model"].startswith("openrouter/")


def test_v1_cost_hidden_param_is_the_fork_obligation(frozen_ambient) -> None:
    """The one envelope post-step: v1 copies usage.cost into
    _hidden_params['additional_headers'][cost-header] as a FLOAT. The dump
    rows above cannot see it; this pins (a) v1 sets it, (b) v1 does NOT set
    it when the wire carries no cost, and (c) the v2 dump retains usage.cost
    byte-identically so the future fork can rebuild the exact header
    (CLAUDE.md HARD OBLIGATION)."""
    v1 = _v1_model_response(_RESPONSES["usage_cost"])
    headers = v1._hidden_params.get("additional_headers", {})
    assert headers.get(_COST_HEADER) == 0.0015
    assert isinstance(headers.get(_COST_HEADER), float)
    v1_plain = _v1_model_response(_RESPONSES["text"])
    assert _COST_HEADER not in v1_plain._hidden_params.get("additional_headers", {})
    v2 = _v2_model_response(_RESPONSES["usage_cost"])
    assert v2["usage"]["cost"] == 0.0015
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

    assert OWN_MODULE_RESPONSE_STYLES["openrouter"] == "openai"
    raw = copy.deepcopy(_RESPONSES["text"])
    raw["created"] = raw["created"] + 0.5
    v1 = _v1_model_response(raw).model_dump()
    correct = _v2_with_style(raw, OWN_MODULE_RESPONSE_STYLES["openrouter"])
    assert _norm(correct) == _norm(v1)
    assert correct["created"] == _RESPONSES["text"]["created"]
    with pytest.raises(pydantic.ValidationError):
        _v2_with_style(raw, "openai_like")
