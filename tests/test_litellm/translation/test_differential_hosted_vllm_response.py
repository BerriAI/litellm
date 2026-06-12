"""Differential parity for the hosted_vllm response path (httpx, NO prefix).

``HostedVLLMChatConfig`` has no ``transform_response`` override: the
inherited base GPT transform is LIVE on the dedicated elif over a FRESH
``ModelResponse`` (model=None) — the bare-wire-model deepseek/xai shape.
These rows run that exact v1 chain against v2's shared openai parser,
byte-identical dumps, and pin that no ``hosted_vllm/`` prefix ever appears.
"""

import copy
import json

import pytest

from litellm.types.utils import ModelResponse

from litellm.translation.inbound.openai_chat import parse_request
from litellm.translation.inbound.openai_chat.response import serialize_response
from litellm.translation.providers.hosted_vllm import parse_response
from litellm.translation_seam import build_translation_deps, to_model_response

from ._own_module_corpus import run_v1_response_transform

_REQUEST = {
    "model": "qwen2.5-7b-instruct",
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
        "id": "chatcmpl-vllm1",
        "object": "chat.completion",
        "created": 1718000000,
        "model": "qwen2.5-7b-instruct",
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
        "id": "chatcmpl-vllm2",
        "object": "chat.completion",
        "created": 1718000000,
        "model": "qwen2.5-7b-instruct",
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
    # vLLM reasoning-parser responses carry reasoning_content on the message
    "reasoning_content": {
        "id": "chatcmpl-vllm3",
        "object": "chat.completion",
        "created": 1718000000,
        "model": "qwen3-8b",
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
        "usage": {"prompt_tokens": 11, "completion_tokens": 9, "total_tokens": 20},
    },
}


def _v1_model_response(raw: dict) -> dict:
    return run_v1_response_transform(
        "hosted_vllm", raw, "qwen2.5-7b-instruct"
    ).model_dump()


def _v2_model_response(raw: dict) -> dict:
    parsed = parse_request(copy.deepcopy(_REQUEST))
    assert parsed.is_ok(), parsed.error.summary
    response = parse_response(copy.deepcopy(raw), parsed.ok)
    assert response.is_ok(), response.error.summary
    body = serialize_response(response.ok, build_translation_deps(), "openai")
    return to_model_response(body, ModelResponse(), usage_style="openai").model_dump()


def _norm(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


@pytest.mark.parametrize("name", sorted(_RESPONSES))
def test_v2_response_matches_v1(name: str, frozen_ambient) -> None:
    raw = _RESPONSES[name]
    assert _norm(_v2_model_response(raw)) == _norm(_v1_model_response(raw))


@pytest.mark.parametrize("name", sorted(_RESPONSES))
def test_no_hosted_vllm_prefix_on_the_response_model(name: str, frozen_ambient) -> None:
    raw = _RESPONSES[name]
    v1 = _v1_model_response(raw)
    v2 = _v2_model_response(raw)
    assert v1["model"] == raw["model"]
    assert v2["model"] == raw["model"]
    assert not v2["model"].startswith("hosted_vllm/")
