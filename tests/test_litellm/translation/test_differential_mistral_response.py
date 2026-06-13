"""Differential parity for the mistral response path (wave-2b-beta).

v1 side: ``MistralConfig.transform_response`` (LIVE on the httpx path) —
the two raw-body pre-steps (empty content -> None FIRST, then the magistral
content-list collapse where the LAST text block wins and thinking texts
join with newlines into ``reasoning_content``) and then
``convert_to_model_response_object`` over a fresh ModelResponse (model None
-> BARE wire model). v2: the mistral pre-steps + the shared openai parser
+ the seam's openai construction arm, byte-identical dumps.
"""

import copy
import json
import time

import httpx
import pydantic
import pytest

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.mistral.chat.transformation import MistralConfig
from litellm.types.utils import ModelResponse

from litellm.translation.inbound.openai_chat import parse_request
from litellm.translation.inbound.openai_chat.response import serialize_response
from litellm.translation.providers.mistral import parse_response
from litellm.translation_seam import (
    UsageStyle,
    build_translation_deps,
    to_model_response,
)

MODEL = "mistral-large-latest"
WIRE_MODEL = "mistral-large-2411"
_AMBIENT_ID = "chatcmpl-mistral-diff"

_REQUEST = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}

_BASE = {
    "id": "cmpl-1",
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
    "empty_content_to_none": {
        **_BASE,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "stop",
            }
        ],
    },
    "magistral_thinking_blocks": {
        **_BASE,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": [
                                {"type": "text", "text": "step1"},
                                {"type": "text", "text": "step2"},
                            ],
                        },
                        {"type": "text", "text": "answer"},
                    ],
                },
                "finish_reason": "stop",
            }
        ],
    },
    "thinking_only_keeps_empty_string": {
        # the pre-step ORDER pin: empty->None runs FIRST, so the collapsed
        # "" content from a text-less list STAYS "" (never None)
        **_BASE,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": [{"type": "text", "text": "only think"}],
                        }
                    ],
                },
                "finish_reason": "stop",
            }
        ],
    },
    "last_text_block_wins": {
        **_BASE,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "first"},
                        {"type": "text", "text": "second"},
                    ],
                },
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
                    "content": "",
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
}

_LOUD = {
    "non_dict_content_block": (
        {
            **_BASE,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": ["bare"]},
                    "finish_reason": "stop",
                }
            ],
        },
        "not an object",
    ),
    "non_list_thinking": (
        {
            **_BASE,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "thinking", "thinking": "bare"}],
                    },
                    "finish_reason": "stop",
                }
            ],
        },
        "not a list",
    ),
    # verifier-wave2b-beta F3: the old arms coerced-and-served both shapes —
    # a non-str thinking text (v1's "\n".join TypeError) and an empty
    # content list (v1's truthy gate skips the collapse; cdr raises
    # "Invalid response object" on Message(content=[])).
    "non_string_thinking_text": (
        {
            **_BASE,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "thinking",
                                "thinking": [{"type": "text", "text": 5}],
                            },
                            {"type": "text", "text": "answer"},
                        ],
                    },
                    "finish_reason": "stop",
                }
            ],
        },
        "thinking text is not a string",
    ),
    "empty_content_list": (
        {
            **_BASE,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": []},
                    "finish_reason": "stop",
                }
            ],
        },
        "content list is empty",
    ),
}


def _v1_model_response(raw: dict) -> dict:
    logging = Logging(
        model=MODEL,
        messages=copy.deepcopy(_REQUEST["messages"]),
        stream=False,
        call_type="completion",
        start_time=time.time(),
        litellm_call_id="diff-mistral-response",
        function_id="diff-mistral-response",
    )
    response = httpx.Response(
        200,
        json=copy.deepcopy(raw),
        request=httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions"),
    )
    return (
        MistralConfig()
        .transform_response(
            model=MODEL,
            raw_response=response,
            model_response=ModelResponse(id=_AMBIENT_ID),
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


def _v2_with_style(raw: dict, style: UsageStyle) -> dict:
    response = _v2_parse(raw)
    assert response.is_ok(), response.error.summary
    body = serialize_response(response.ok, build_translation_deps(), "openai")
    return to_model_response(
        body, ModelResponse(id=_AMBIENT_ID), usage_style=style
    ).model_dump()


def _v2_model_response(raw: dict) -> dict:
    # mistral rides the cdr arm — the truth is
    # pipeline.OWN_MODULE_RESPONSE_STYLES["mistral"], divergence-pinned below
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
    assert not str(v2["model"]).startswith("mistral/")


def test_reasoning_content_extracted(frozen_ambient) -> None:
    v2 = _v2_model_response(_RESPONSES["magistral_thinking_blocks"])
    message = v2["choices"][0]["message"]
    assert message["content"] == "answer"
    assert message["reasoning_content"] == "step1\nstep2"


def test_thinking_only_content_stays_empty_string(frozen_ambient) -> None:
    v1 = _v1_model_response(_RESPONSES["thinking_only_keeps_empty_string"])
    v2 = _v2_model_response(_RESPONSES["thinking_only_keeps_empty_string"])
    assert v1["choices"][0]["message"]["content"] == ""
    assert v2["choices"][0]["message"]["content"] == ""


@pytest.mark.parametrize("name", sorted(_LOUD))
def test_loud_shapes_error_on_both_sides(name: str, frozen_ambient) -> None:
    raw, fragment = _LOUD[name]
    result = _v2_parse(raw)
    assert result.is_error(), f"{name} unexpectedly parsed"
    assert fragment in result.error.summary, result.error.summary
    with pytest.raises(Exception):
        _v1_model_response(copy.deepcopy(raw))


def test_wrong_construction_arm_diverges_and_the_table_pins_it(
    frozen_ambient,
) -> None:
    """verifier-wave2b-final F1: the "openai"-arm OWN_MODULE_RESPONSE_STYLES
    values were enforced by NOTHING — a wrong-arm value flip on THIS row was
    the verifier's proof (it survived the whole 3038 suite). The watsonx/
    groq wrong-arm template INVERTED for a cdr-parser member (the v2 parser
    already cdr-normalizes choices, so the openai_like members' index-5
    discriminator dies before the seam): a FLOAT wire ``created`` rides the
    normalized body; the correct cdr ("openai") arm coerces it via
    _safe_convert_created_field and serves exactly like v1, while the wrong
    "openai_like" arm (ModelResponse(**json)) raises ValidationError on
    traffic v1 serves — if the arms stop diverging here, the raises assert
    fails: re-decide before relying on the pin."""
    from litellm.translation.engine.pipeline import OWN_MODULE_RESPONSE_STYLES

    assert OWN_MODULE_RESPONSE_STYLES["mistral"] == "openai"
    raw = copy.deepcopy(_RESPONSES["text"])
    raw["created"] = raw["created"] + 0.5
    v1 = _v1_model_response(raw)
    correct = _v2_with_style(raw, OWN_MODULE_RESPONSE_STYLES["mistral"])
    assert _norm(correct) == _norm(v1)
    assert correct["created"] == _RESPONSES["text"]["created"]
    with pytest.raises(pydantic.ValidationError):
        _v2_with_style(raw, "openai_like")
