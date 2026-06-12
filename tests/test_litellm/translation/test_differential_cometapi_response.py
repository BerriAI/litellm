"""Differential parity for the cometapi response path (httpx, NO prefix).

cometapi is a compat_httpx family row (main.py:2547 dedicated elif, BEFORE
the big SDK elif despite compat-list membership): ``transform_response`` is
LIVE and is the inherited base one — ``convert_to_model_response_object``
over a fresh ``ModelResponse`` with ``model=None``, so the response model is
the BARE wire model (the xai R4 pattern). These rows run v1's
``CometAPIConfig.transform_response`` over a real ``httpx.Response`` against
v2's shared openai parser with NO preset model, byte-identical dumps, and
pin that no ``cometapi/`` prefix ever appears.
"""

import copy
import json
import time

import httpx
import pytest

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.types.utils import ModelResponse
from litellm.utils import ProviderConfigManager
from litellm.types.utils import LlmProviders

from litellm.translation.inbound.openai_chat import parse_request
from litellm.translation.inbound.openai_chat.response import serialize_response
from litellm.translation.providers.openai_compat.response import parse_response
from litellm.translation_seam import build_translation_deps, to_model_response

_REQUEST = {
    "model": "gpt-4o-mini",
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
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1718000000,
        "model": "gpt-4o-mini-2024-07-18",
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
        "id": "chatcmpl-2",
        "object": "chat.completion",
        "created": 1718000000,
        "model": "gpt-4o-mini-2024-07-18",
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
    "reasoning_usage_details": {
        "id": "chatcmpl-3",
        "object": "chat.completion",
        "created": 1718000000,
        "model": "deepseek-r1",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "thought through"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 9,
            "total_tokens": 20,
            "completion_tokens_details": {"reasoning_tokens": 4},
        },
    },
}


def _v1_model_response(raw: dict) -> dict:
    config = ProviderConfigManager.get_provider_chat_config(
        model="gpt-4o-mini", provider=LlmProviders("cometapi")
    )
    logging = Logging(
        model="gpt-4o-mini",
        messages=copy.deepcopy(_REQUEST["messages"]),
        stream=False,
        call_type="completion",
        start_time=time.time(),
        litellm_call_id="diff-cometapi-response",
        function_id="diff-cometapi-response",
    )
    response = httpx.Response(
        200,
        json=copy.deepcopy(raw),
        request=httpx.Request("POST", "https://api.cometapi.com/v1/chat/completions"),
    )
    result = config.transform_response(
        model="gpt-4o-mini",
        raw_response=response,
        model_response=ModelResponse(),
        logging_obj=logging,
        request_data={},
        messages=copy.deepcopy(_REQUEST["messages"]),
        optional_params={},
        litellm_params={},
        encoding=None,
    )
    return result.model_dump()


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
def test_no_cometapi_prefix_on_the_response_model(name: str, frozen_ambient) -> None:
    """The R4 pin: the httpx path starts from a fresh ModelResponse
    (model=None) and adopts the BARE wire model; the seam must never preset
    ``cometapi/{model}`` the way it does for the SDK family members."""
    raw = _RESPONSES[name]
    v1 = _v1_model_response(raw)
    v2 = _v2_model_response(raw)
    assert v1["model"] == raw["model"]
    assert v2["model"] == raw["model"]
    assert not v2["model"].startswith("cometapi/")
