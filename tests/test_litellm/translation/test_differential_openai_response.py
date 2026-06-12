"""Differential parity for the openai response path.

The v1 reference is ``convert_to_model_response_object`` over the SDK-dump
response shape: the live normalizer on the SDK path (dossier gotcha #1;
``OpenAIGPTConfig.transform_response`` never runs there). v2 goes
``parse_response`` -> ``serialize_response(dialect="openai")`` ->
``to_model_response(usage_style="openai")``, and the two ``ModelResponse``
dumps must be identical: wire id/created/system_fingerprint survive, usage is
the verbatim ``Usage(**raw)`` passthrough (cached + reasoning details), the
finish_reason stop -> tool_calls rewrite fires, and reasoning content is
extracted from the key or ``<think>`` tags. Shapes the surface cannot trigger
must be typed errors, never silent drops.
"""

import copy
import json

import pytest

from litellm.types.utils import ModelResponse
from litellm.utils import convert_to_model_response_object

from litellm.translation.inbound.openai_chat import parse_request
from litellm.translation.inbound.openai_chat.response import serialize_response
from litellm.translation.providers.openai_compat.response import parse_response
from litellm.translation_seam import build_translation_deps, to_model_response

MODEL = "gpt-4o"

_REQUEST = {
    "model": MODEL,
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

_USAGE = {
    "completion_tokens": 6,
    "prompt_tokens": 12,
    "total_tokens": 18,
    "completion_tokens_details": {
        "accepted_prediction_tokens": 0,
        "audio_tokens": 0,
        "reasoning_tokens": 0,
        "rejected_prediction_tokens": 0,
    },
    "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0},
}

_RESPONSES = {
    "text": {
        "id": "chatcmpl-A1",
        "object": "chat.completion",
        "created": 1718000000,
        "model": "gpt-4o-2024-08-06",
        "system_fingerprint": "fp_abc123",
        "service_tier": "default",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "logprobs": None,
                "message": {
                    "content": "Hello there.",
                    "role": "assistant",
                    "refusal": None,
                    "annotations": [],
                    "audio": None,
                    "function_call": None,
                    "tool_calls": None,
                },
            }
        ],
        "usage": _USAGE,
    },
    "tool_calls_rewrites_stop": {
        # finish_reason "stop" + tool_calls -> "tool_calls" on both sides
        "id": "chatcmpl-T1",
        "object": "chat.completion",
        "created": 1718000001,
        "model": "gpt-4o-2024-08-06",
        "system_fingerprint": "fp_x",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "logprobs": None,
                "message": {
                    "content": None,
                    "role": "assistant",
                    "refusal": None,
                    "annotations": [],
                    "audio": None,
                    "function_call": None,
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
            }
        ],
        "usage": _USAGE,
    },
    "cached_and_reasoning_usage_details": {
        "id": "chatcmpl-U1",
        "object": "chat.completion",
        "created": 1718000002,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "finish_reason": "length",
                "logprobs": None,
                "message": {
                    "content": "partial",
                    "role": "assistant",
                    "refusal": None,
                    "annotations": [],
                },
            }
        ],
        "usage": {
            "completion_tokens": 100,
            "prompt_tokens": 1000,
            "total_tokens": 1100,
            "completion_tokens_details": {
                "accepted_prediction_tokens": 0,
                "audio_tokens": 0,
                "reasoning_tokens": 64,
                "rejected_prediction_tokens": 0,
            },
            "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 512},
        },
    },
    "reasoning_content_key": {
        # compat providers (deepseek-style) return reasoning_content beside
        # content; v1 lifts it onto Message.reasoning_content
        "id": "chatcmpl-R1",
        "object": "chat.completion",
        "created": 1718000003,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "logprobs": None,
                "message": {
                    "content": "answer",
                    "role": "assistant",
                    "reasoning_content": "thought hard",
                },
            }
        ],
        "usage": {"completion_tokens": 5, "prompt_tokens": 7, "total_tokens": 12},
    },
    "think_tag_extraction": {
        "id": "chatcmpl-K1",
        "object": "chat.completion",
        "created": 1718000004,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "logprobs": None,
                "message": {"content": "<think>hmm</think>final", "role": "assistant"},
            }
        ],
        "usage": {"completion_tokens": 5, "prompt_tokens": 7, "total_tokens": 12},
    },
}

_UNSUPPORTED = {
    "multiple_choices": (
        {
            "id": "chatcmpl-N",
            "created": 1,
            "model": MODEL,
            "choices": [
                {"index": 0, "finish_reason": "stop", "message": {"content": "a"}},
                {"index": 1, "finish_reason": "stop", "message": {"content": "b"}},
            ],
            "usage": _USAGE,
        },
        "multiple response choices",
    ),
    "legacy_function_call_output": (
        {
            "id": "chatcmpl-F",
            "created": 1,
            "model": MODEL,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "function_call",
                    "message": {
                        "content": None,
                        "role": "assistant",
                        "function_call": {"name": "f", "arguments": "{}"},
                    },
                }
            ],
            "usage": _USAGE,
        },
        "function_call",
    ),
    "multi_tool_use_parallel_repair": (
        {
            "id": "chatcmpl-M",
            "created": 1,
            "model": MODEL,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": None,
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "multi_tool_use.parallel",
                                    "arguments": '{"tool_uses": []}',
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": _USAGE,
        },
        "multi_tool_use.parallel",
    ),
}


PRESET_PREFIXED_MODEL = "someprovider/pre-call-model"


def _v1_model_response(raw: dict, preset_model: str | None = None) -> dict:
    result = convert_to_model_response_object(
        response_object=copy.deepcopy(raw),
        model_response_object=ModelResponse(model=preset_model),
    )
    return result.model_dump()


def _v2_model_response(raw: dict, preset_model: str | None = None) -> dict:
    parsed = parse_request(copy.deepcopy(_REQUEST))
    assert parsed.is_ok(), parsed.error.summary
    response = parse_response(copy.deepcopy(raw), parsed.ok)
    assert response.is_ok(), response.error.summary
    body = serialize_response(response.ok, build_translation_deps(), "openai")
    return to_model_response(
        body, ModelResponse(model=preset_model), usage_style="openai"
    ).model_dump()


def _norm(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


@pytest.mark.parametrize("name", sorted(_RESPONSES))
def test_v2_response_matches_v1(name: str, frozen_ambient) -> None:
    raw = _RESPONSES[name]
    assert _norm(_v2_model_response(raw)) == _norm(_v1_model_response(raw))


@pytest.mark.parametrize("preset", [PRESET_PREFIXED_MODEL, "no-slash-model"])
def test_preset_model_handling_matches_v1(preset: str, frozen_ambient) -> None:
    """v1's openai handler pre-sets model_response.model to
    "{custom_llm_provider}/{model}" for every non-"openai" compat consumer
    BEFORE conversion, and the completion branch then rewrites it to
    "{provider}/{wire model}". A fresh ModelResponse (model=None) can never
    exercise that arm, so these rows pass a pre-set model into BOTH sides:
    the prefixed preset must be re-prefixed onto the wire model, the
    slash-free preset must be kept verbatim (critic-openai B1)."""
    raw = _RESPONSES["text"]
    v1 = _v1_model_response(raw, preset_model=preset)
    v2 = _v2_model_response(raw, preset_model=preset)
    assert _norm(v2) == _norm(v1)
    expected = (
        "someprovider/gpt-4o-2024-08-06"
        if preset == PRESET_PREFIXED_MODEL
        else "no-slash-model"
    )
    assert v2["model"] == expected


@pytest.mark.parametrize("name", sorted(_UNSUPPORTED))
def test_unreachable_response_shape_is_a_typed_error(name: str) -> None:
    raw, reason_fragment = _UNSUPPORTED[name]
    parsed = parse_request(copy.deepcopy(_REQUEST))
    assert parsed.is_ok(), parsed.error.summary
    result = parse_response(copy.deepcopy(raw), parsed.ok)
    assert result.is_error(), f"{name} unexpectedly parsed"
    assert reason_fragment in result.error.summary, result.error.summary
