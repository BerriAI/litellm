"""Differential parity for the watsonx response path (wave-2b-beta).

v1 side: ``OpenAILikeChatConfig._transform_response`` invoked the way
``OpenAILikeChatHandler`` calls it — ``custom_llm_provider="watsonx"``, the
ONE wave-2b path where the openai_like PREFIX arm is LIVE:
``ModelResponse(**response_json)`` directly, then
``model = "watsonx/" + (wire model or "")``. v2: the watsonx parser
(openai validation + verbatim wire with the prefixed WIRE model) + the
seam's ``openai_like`` construction arm. v1 ignores the pre-allocated
model_response on this path, so no shared-id plumbing is needed: both
sides derive every byte from the wire body.
"""

import copy
import json
import time

import httpx
import pytest

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.openai_like.chat.transformation import OpenAILikeChatConfig
from litellm.types.utils import ModelResponse

from litellm.translation.inbound.openai_chat import parse_request
from litellm.translation.inbound.openai_chat.response import serialize_response
from litellm.translation.providers.watsonx import parse_response
from litellm.translation_seam import build_translation_deps, to_model_response

MODEL = "ibm/granite-3-8b-instruct"
_REQUEST = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}

_BASE = {
    "id": "chat-1",
    "object": "chat.completion",
    "created": 1718000000,
    "model": MODEL,
    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
}

_RESPONSES = {
    "text": {
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
    "null_usage_tokens_sanitized": {
        # the OpenAILike *_tokens-None sanitize is observationally dead:
        # Usage coerces None -> 0 in the constructor (the family pin)
        **_BASE,
        "usage": {"prompt_tokens": None, "completion_tokens": 2, "total_tokens": None},
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "x"},
                "finish_reason": "stop",
            }
        ],
    },
    "missing_model_prefixes_empty": {
        **{key: value for key, value in _BASE.items() if key != "model"},
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
        litellm_call_id="diff-watsonx-response",
        function_id="diff-watsonx-response",
    )
    response = httpx.Response(
        200,
        json=copy.deepcopy(raw),
        request=httpx.Request("POST", "https://x/ml/v1/text/chat"),
    )
    return OpenAILikeChatConfig._transform_response(
        model=MODEL,
        response=response,
        model_response=ModelResponse(),
        stream=False,
        logging_obj=logging,
        optional_params={},
        api_key=None,
        data={},
        messages=copy.deepcopy(_REQUEST["messages"]),
        print_verbose=None,
        encoding=None,
        json_mode=None,
        custom_llm_provider="watsonx",
        base_model=None,
    ).model_dump()


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


def test_live_prefix_uses_the_wire_model(frozen_ambient) -> None:
    v1 = _v1_model_response(_RESPONSES["text"])
    v2 = _v2_model_response(_RESPONSES["text"])
    assert v1["model"] == f"watsonx/{MODEL}"
    assert v2["model"] == f"watsonx/{MODEL}"


def test_missing_model_serves_bare_prefix(frozen_ambient) -> None:
    v2 = _v2_model_response(_RESPONSES["missing_model_prefixes_empty"])
    assert v2["model"] == "watsonx/"


def test_non_string_wire_model_falls_back_where_v1_raises(frozen_ambient) -> None:
    """The verifier-longtail F2 arm: v1's ModelResponse(**json) raises
    pydantic ValidationError BEFORE the prefix overwrite."""
    raw = {**_RESPONSES["text"], "model": 7}
    result = _v2_parse(raw)
    assert result.is_error()
    assert "non-string wire model" in result.error.summary
    with pytest.raises(Exception):
        _v1_model_response(raw)
