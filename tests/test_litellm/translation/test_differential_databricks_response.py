"""Differential parity for the databricks response path (wave 3).

v1 side: ``DatabricksConfig.transform_response`` mutating a pre-allocated
``ModelResponse`` (probed in-process at HEAD). v1 reads a ``DatabricksResponse``
TypedDict whose construction does NOT validate, so a body missing a required
key (id/created/usage/choices, or a choice's index/message/finish_reason)
raises a RAW ``KeyError`` (a drift from researcher-5's "DatabricksException"
claim, pinned). The model is ``databricks/{wire model}``, id/created copied
verbatim, usage from prompt/completion tokens, and ``_transform_dbrx_choices``
flattens a content block-list to a joined text string, turns reasoning/summary
blocks into ``reasoning_content`` + ``thinking_blocks`` (missing signature ->
""), and lifts a first-item ``citations`` list to
``provider_specific_fields.citations`` with a ``supported_text`` mirror.

v2 side: the parser builds the normalized body on ``ChatResponse.wire`` and the
seam constructs it. Because the parser PRE-NORMALIZES the whole body, the
"openai" (cdr) and "openai_like" (ModelResponse(**json)) construction arms
COINCIDE on the corpus (the cohere/ollama N3 rule: the wire body is the parity
surface), so unlike ollama there is no envelope-id discriminator. The
construction arm is pinned by the registration value
``OWN_MODULE_RESPONSE_STYLES["databricks"] == "openai"`` and a documented
arms-coincide note (a flip would not change bytes on this corpus — the
registration gate is the guard, not output divergence).
"""

import copy
import json
import time

import httpx
import pytest

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.databricks.chat.transformation import DatabricksConfig
from litellm.types.utils import ModelResponse

from litellm.translation.engine.pipeline import OWN_MODULE_RESPONSE_STYLES
from litellm.translation.inbound.openai_chat import parse_request
from litellm.translation.inbound.openai_chat.response import serialize_response
from litellm.translation.providers.databricks import parse_response
from litellm.translation_seam import (
    UsageStyle,
    build_translation_deps,
    to_model_response,
)

MODEL = "databricks-dbrx-instruct"
PREFIXED = "databricks/dbrx"
_AMBIENT_ID = "chatcmpl-databricks-amb"

_REQUEST = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}

_BASE = {
    "id": "chatcmpl-db",
    "created": 1718000000,
    "model": "dbrx",
    "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "hello"},
        }
    ],
}

_RESPONSES = {
    "plain": _BASE,
    "finish_length": {
        **_BASE,
        "choices": [{**_BASE["choices"][0], "finish_reason": "length"}],
    },
    "finish_unknown_maps_stop": {
        **_BASE,
        "choices": [{**_BASE["choices"][0], "finish_reason": "weird"}],
    },
    "wire_model_prefixed": {**_BASE, "model": "my-endpoint"},
    "content_list_flattens": {
        **_BASE,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "a"},
                        {"type": "text", "text": "b"},
                    ],
                },
            }
        ],
    },
    "reasoning_summary_to_reasoning_content": {
        **_BASE,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "ans"},
                        {
                            "type": "reasoning",
                            "summary": [
                                {
                                    "type": "summary_text",
                                    "text": "thinking",
                                    "signature": "sig",
                                }
                            ],
                        },
                    ],
                },
            }
        ],
    },
    "reasoning_summary_missing_signature": {
        **_BASE,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "reasoning",
                            "summary": [{"type": "summary_text", "text": "t"}],
                        }
                    ],
                },
            }
        ],
    },
    "citations_lift_with_supported_text": {
        **_BASE,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "ans",
                            "citations": [{"source": "doc1"}],
                        }
                    ],
                },
            }
        ],
    },
    "tool_calls_verbatim": {
        **_BASE,
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "f", "arguments": '{"a": 1}'},
                        }
                    ],
                },
            }
        ],
    },
    "unknown_top_level_key_dropped": {**_BASE, "object": "WRONG", "extra": 9},
}

# v1 raises a RAW KeyError (the unvalidated TypedDict access); v2 falls back
# typed so v1 serves its own crash.
_LOUD = {
    "missing_id": ({k: v for k, v in _BASE.items() if k != "id"}, "id"),
    "missing_created": (
        {k: v for k, v in _BASE.items() if k != "created"},
        "created",
    ),
    "missing_usage": ({k: v for k, v in _BASE.items() if k != "usage"}, "usage"),
    "missing_choices": (
        {k: v for k, v in _BASE.items() if k != "choices"},
        "choices",
    ),
    "choice_missing_message": (
        {**_BASE, "choices": [{"index": 0, "finish_reason": "stop"}]},
        "message",
    ),
    "choice_missing_finish_reason": (
        {
            **_BASE,
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "x"}}
            ],
        },
        "finish_reason",
    ),
}


def _v1_model_response(raw) -> dict:
    logging = Logging(
        model=MODEL,
        messages=copy.deepcopy(_REQUEST["messages"]),
        stream=False,
        call_type="completion",
        start_time=time.time(),
        litellm_call_id="diff-databricks-response",
        function_id="diff-databricks-response",
    )
    response = httpx.Response(
        200,
        json=copy.deepcopy(raw),
        request=httpx.Request("POST", "https://x/serving-endpoints"),
    )
    return (
        DatabricksConfig()
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


def _v2_parse(raw):
    parsed = parse_request(copy.deepcopy(_REQUEST))
    assert parsed.is_ok(), parsed.error.summary
    return parse_response(copy.deepcopy(raw), parsed.ok)


def _v2_with_style(raw, style: UsageStyle) -> dict:
    response = _v2_parse(raw)
    assert response.is_ok(), response.error.summary
    body = serialize_response(response.ok, build_translation_deps(), "openai")
    return to_model_response(
        body, ModelResponse(id=_AMBIENT_ID), usage_style=style
    ).model_dump()


def _v2_model_response(raw) -> dict:
    return _v2_with_style(raw, OWN_MODULE_RESPONSE_STYLES["databricks"])


def _norm(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


@pytest.mark.parametrize("name", sorted(_RESPONSES))
def test_v2_response_matches_v1(name: str) -> None:
    raw = _RESPONSES[name]
    assert _norm(_v2_model_response(raw)) == _norm(_v1_model_response(raw))


@pytest.mark.parametrize("name", sorted(_RESPONSES))
def test_model_is_prefixed_and_ids_copied_verbatim(name: str) -> None:
    """model = databricks/{wire model}; id/created ride the WIRE chunk
    verbatim (no ambient overwrite — the wire carries them, unlike ollama)."""
    raw = _RESPONSES[name]
    v1 = _v1_model_response(raw)
    v2 = _v2_model_response(raw)
    for dump in (v1, v2):
        assert dump["model"] == f"databricks/{raw['model']}"
        assert dump["id"] == raw["id"]
        assert dump["created"] == raw["created"]


@pytest.mark.parametrize("name", sorted(_LOUD))
def test_loud_shapes_fall_back_where_v1_raw_keyerrors(name: str) -> None:
    raw, fragment = _LOUD[name]
    result = _v2_parse(raw)
    assert result.is_error(), f"{name} unexpectedly parsed"
    assert fragment in result.error.summary, result.error.summary
    with pytest.raises(KeyError):
        _v1_model_response(copy.deepcopy(raw))


def test_content_list_flattens_to_joined_text() -> None:
    raw = _RESPONSES["content_list_flattens"]
    for dump in (_v1_model_response(raw), _v2_model_response(raw)):
        assert dump["choices"][0]["message"]["content"] == "ab"


def test_reasoning_summary_missing_signature_defaults_empty() -> None:
    raw = _RESPONSES["reasoning_summary_missing_signature"]
    for dump in (_v1_model_response(raw), _v2_model_response(raw)):
        blocks = dump["choices"][0]["message"]["thinking_blocks"]
        assert blocks[0]["signature"] == ""


def test_citations_lift_to_provider_specific_fields_with_supported_text() -> None:
    raw = _RESPONSES["citations_lift_with_supported_text"]
    for dump in (_v1_model_response(raw), _v2_model_response(raw)):
        psf = dump["choices"][0]["message"]["provider_specific_fields"]
        assert psf["citations"] == [[{"source": "doc1", "supported_text": "ans"}]]


def test_construction_arms_coincide_and_the_table_pins_the_value() -> None:
    """Unlike ollama (whose body carries no id, so the cdr vs ModelResponse
    arms diverge on the envelope id), the databricks parser PRE-NORMALIZES the
    whole body onto ChatResponse.wire, so the "openai" and "openai_like" arms
    produce identical bytes on the corpus. The construction arm is therefore
    pinned by the registration VALUE, not by output divergence — a flip would
    be silent here, so the registration gate is the guard. This test pins both
    the value AND the coincidence fact (if a future change makes the arms
    diverge, re-introduce an output-divergence pin)."""
    assert OWN_MODULE_RESPONSE_STYLES["databricks"] == "openai"
    for name in ("plain", "tool_calls_verbatim", "finish_length"):
        raw = _RESPONSES[name]
        cdr = _v2_with_style(raw, "openai")
        direct = _v2_with_style(raw, "openai_like")
        assert _norm(cdr) == _norm(direct), (
            f"{name}: the arms diverged — databricks gained an envelope-shaped "
            "discriminator; pin it with an output-divergence test like ollama's"
        )


def test_finish_reason_mirror_matches_v1_table() -> None:
    """The IR finish set is the four chat-completion reasons; an unknown wire
    finish maps to stop on both sides (probed: 'weird' -> 'stop')."""
    raw = _RESPONSES["finish_unknown_maps_stop"]
    v1 = _v1_model_response(raw)
    v2 = _v2_model_response(raw)
    assert v1["choices"][0]["finish_reason"] == "stop"
    assert v2["choices"][0]["finish_reason"] == "stop"
