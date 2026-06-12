"""Differential parity for the groq response path (wave-2b-beta).

v1 side: ``GroqChatConfig.transform_response`` — the OpenAILike DIRECT
construction (``custom_llm_provider=None`` -> the prefix arm is DEAD, BARE
wire model) plus the service_tier clamp post-step ({auto, default, flex},
None/unknown -> "auto"). The post-step reads ``getattr(model_response,
"service_tier")`` with NO default, so a response body WITHOUT the key
crashes v1 with AttributeError — v2 fails closed on it (pinned). Unknown
top-level keys (``x_groq``) survive the direct construction.
"""

import copy
import json
import time

import httpx
import pytest

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.groq.chat.transformation import GroqChatConfig
from litellm.types.utils import ModelResponse

from litellm.translation.inbound.openai_chat import parse_request
from litellm.translation.inbound.openai_chat.response import serialize_response
from litellm.translation.providers.groq import parse_response
from litellm.translation_seam import build_translation_deps, to_model_response

MODEL = "llama-3.3-70b-versatile"
_REQUEST = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}

_BASE = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1718000000,
    "model": MODEL,
    "service_tier": "on_demand",
    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
}

_RESPONSES = {
    "text_tier_clamped_to_auto": {
        **_BASE,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
            }
        ],
    },
    "tool_calls": {
        **_BASE,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "f", "arguments": '{"a":1}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    },
    "x_groq_extra_survives": {
        **_BASE,
        "x_groq": {"id": "req_1"},
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "x"},
                "finish_reason": "stop",
            }
        ],
    },
    "flex_tier_kept": {
        **_BASE,
        "service_tier": "flex",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "x"},
                "finish_reason": "stop",
            }
        ],
    },
    "null_tier_to_auto": {
        **_BASE,
        "service_tier": None,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "x"},
                "finish_reason": "stop",
            }
        ],
    },
}


def _v1_model_response(raw: dict) -> dict:
    logging = Logging(
        model=MODEL,
        messages=copy.deepcopy(_REQUEST["messages"]),
        stream=False,
        call_type="completion",
        start_time=time.time(),
        litellm_call_id="diff-groq-response",
        function_id="diff-groq-response",
    )
    response = httpx.Response(
        200,
        json=copy.deepcopy(raw),
        request=httpx.Request(
            "POST", "https://api.groq.com/openai/v1/chat/completions"
        ),
    )
    return (
        GroqChatConfig()
        .transform_response(
            model=MODEL,
            raw_response=response,
            model_response=ModelResponse(),
            logging_obj=logging,
            request_data={},
            messages=copy.deepcopy(_REQUEST["messages"]),
            optional_params={},
            litellm_params={},
            encoding=None,
        )
        .model_dump()
    )


def _v2_parse(raw: dict):
    parsed = parse_request(copy.deepcopy(_REQUEST))
    assert parsed.is_ok(), parsed.error.summary
    return parse_response(copy.deepcopy(raw), parsed.ok)


def _v2_model_response(raw: dict) -> dict:
    response = _v2_parse(raw)
    assert response.is_ok(), response.error.summary
    body = serialize_response(response.ok, build_translation_deps(), "openai")
    return to_model_response(body, None, usage_style="openai_like").model_dump()


def _norm(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


@pytest.mark.parametrize("name", sorted(_RESPONSES))
def test_v2_response_matches_v1(name: str, frozen_ambient) -> None:
    raw = _RESPONSES[name]
    assert _norm(_v2_model_response(raw)) == _norm(_v1_model_response(raw))


@pytest.mark.parametrize("name", sorted(_RESPONSES))
def test_bare_wire_model_no_prefix(name: str, frozen_ambient) -> None:
    v1 = _v1_model_response(_RESPONSES[name])
    v2 = _v2_model_response(_RESPONSES[name])
    assert v1["model"] == MODEL
    assert v2["model"] == MODEL
    assert not str(v2["model"]).startswith("groq/")


def test_missing_service_tier_fails_closed_where_v1_crashes(frozen_ambient) -> None:
    raw = {
        **{k: v for k, v in _BASE.items() if k != "service_tier"},
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "x"},
                "finish_reason": "stop",
            }
        ],
    }
    result = _v2_parse(raw)
    assert result.is_error()
    assert "service_tier" in result.error.summary
    with pytest.raises(AttributeError):
        _v1_model_response(raw)


def test_non_string_wire_model_falls_back_where_v1_raises(frozen_ambient) -> None:
    raw = {**_RESPONSES["text_tier_clamped_to_auto"], "model": 7}
    result = _v2_parse(raw)
    assert result.is_error()
    assert "non-string wire model" in result.error.summary
    with pytest.raises(Exception):
        _v1_model_response(raw)
