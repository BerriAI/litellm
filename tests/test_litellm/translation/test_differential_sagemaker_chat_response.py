"""Differential parity for the sagemaker_chat response path (wave-2b-beta).

v1 side: the BASE ``transform_response`` (LIVE on the httpx path) ->
``convert_to_model_response_object`` over a fresh ModelResponse — the
cometapi shape: bare wire model, NO seam preset, construction arm
"openai". v2: the shared openai parser verbatim (the sagemaker_chat module
re-exports it).
"""

import copy
import json
import time

import httpx
import pydantic
import pytest

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.sagemaker.chat.transformation import SagemakerChatConfig
from litellm.types.utils import ModelResponse

from litellm.translation.inbound.openai_chat import parse_request
from litellm.translation.inbound.openai_chat.response import serialize_response
from litellm.translation.providers.sagemaker_chat import parse_response
from litellm.translation_seam import (
    UsageStyle,
    build_translation_deps,
    to_model_response,
)

MODEL = "my-endpoint"
WIRE_MODEL = "tgi"
_REQUEST = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}

_BASE = {
    "id": "chat-1",
    "object": "chat.completion",
    "created": 1718000000,
    "model": WIRE_MODEL,
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
    "stop_to_tool_calls_rewrite": {
        # cdr rewrites finish stop -> tool_calls when tool calls are present
        **_BASE,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c2",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ],
                },
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
        litellm_call_id="diff-sagemaker-response",
        function_id="diff-sagemaker-response",
    )
    response = httpx.Response(
        200,
        json=copy.deepcopy(raw),
        request=httpx.Request(
            "POST",
            "https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/my-endpoint/invocations",
        ),
    )
    return (
        SagemakerChatConfig()
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


def _v2_with_style(raw: dict, style: UsageStyle) -> dict:
    parsed = parse_request(copy.deepcopy(_REQUEST))
    assert parsed.is_ok(), parsed.error.summary
    response = parse_response(copy.deepcopy(raw), parsed.ok)
    assert response.is_ok(), response.error.summary
    body = serialize_response(response.ok, build_translation_deps(), "openai")
    return to_model_response(body, ModelResponse(), usage_style=style).model_dump()


def _v2_model_response(raw: dict) -> dict:
    # sagemaker_chat rides the cdr arm — the truth is
    # pipeline.OWN_MODULE_RESPONSE_STYLES["sagemaker_chat"], divergence-pinned below
    return _v2_with_style(raw, "openai")


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
    assert v1["model"] == WIRE_MODEL
    assert v2["model"] == WIRE_MODEL
    assert not str(v2["model"]).startswith("sagemaker")


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

    assert OWN_MODULE_RESPONSE_STYLES["sagemaker_chat"] == "openai"
    raw = copy.deepcopy(_RESPONSES["text"])
    raw["created"] = raw["created"] + 0.5
    v1 = _v1_model_response(raw)
    correct = _v2_with_style(raw, OWN_MODULE_RESPONSE_STYLES["sagemaker_chat"])
    assert _norm(correct) == _norm(v1)
    assert correct["created"] == _RESPONSES["text"]["created"]
    with pytest.raises(pydantic.ValidationError):
        _v2_with_style(raw, "openai_like")
