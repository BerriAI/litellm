"""Differential parity for the cohere v2 response path (wave-2b-beta).

v1 side: ``CohereV2ChatConfig.transform_response`` (LIVE on the httpx path)
mutating a pre-allocated ``ModelResponse`` — content joined from
``message.content[].text``, citations -> url_citation annotations,
tool-call responses REPLACE the message (text content lost, ``content=None``,
explicit ``annotations`` kwarg), usage from ``usage.tokens`` (NOT
billed_units), ``model`` = the request model, the wire id IGNORED (ambient
chatcmpl id kept) and ``finish_reason`` NEVER read (the fresh-Choices
default "stop" survives even for tool calls — quirk-pinned below).

v2 side: the cohere parser builds the normalized body on
``ChatResponse.wire``; the seam's ``openai`` construction arm reproduces
v1's assembly byte-for-byte. Both sides get the SAME pre-allocated
``ModelResponse(id=...)`` because v1 keeps the ambient envelope id (v1 mints
it fresh per call in production — envelope nondeterminism, not parser
scope).
"""

import copy
import json
import time

import httpx
import pytest

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.cohere.chat.v2_transformation import CohereV2ChatConfig
from litellm.types.utils import ModelResponse

from litellm.translation.inbound.openai_chat import parse_request
from litellm.translation.inbound.openai_chat.response import serialize_response
from litellm.translation.providers.cohere import parse_response
from litellm.translation_seam import build_translation_deps, to_model_response

MODEL = "command-r"
_AMBIENT_ID = "chatcmpl-cohere-diff"

_REQUEST = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}

_RESPONSES = {
    "text": {
        "id": "wire-id-ignored",
        "finish_reason": "COMPLETE",
        "message": {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "hel"},
                {"type": "text", "text": "lo"},
            ],
        },
        "usage": {"tokens": {"input_tokens": 3, "output_tokens": 2}},
        "logprobs": {"token_ids": []},
    },
    "tool_calls": {
        "id": "wire-id-ignored",
        "finish_reason": "TOOL_CALL",
        "message": {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "f", "arguments": '{"a":1}'},
                }
            ],
        },
        "usage": {"tokens": {"input_tokens": 5, "output_tokens": 4}},
    },
    "citations_to_annotations": {
        "id": "x",
        "finish_reason": "COMPLETE",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "hi"}],
            "citations": [
                {
                    "start": 0,
                    "end": 2,
                    "text": "hi",
                    "sources": [
                        {"type": "document", "id": "d1", "document": {"title": "T"}}
                    ],
                }
            ],
        },
        "usage": {"tokens": {"input_tokens": 1, "output_tokens": 1}},
    },
    "tool_calls_discard_text_keep_annotations": {
        "id": "x",
        "finish_reason": "TOOL_CALL",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "discarded by v1"}],
            "tool_calls": [
                {
                    "id": "c",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }
            ],
            "citations": [
                {
                    "start": 0,
                    "end": 1,
                    "text": "i",
                    "sources": [
                        {"type": "document", "id": "d", "document": {"title": "T"}}
                    ],
                }
            ],
        },
        "usage": {"tokens": {"input_tokens": 1, "output_tokens": 1}},
    },
    "missing_tokens_defaults_zero": {
        "id": "x",
        "finish_reason": "COMPLETE",
        "message": {"role": "assistant", "content": []},
        "usage": {},
    },
    "null_content": {
        "id": "x",
        "finish_reason": "COMPLETE",
        "message": {"role": "assistant", "content": None},
        "usage": {"tokens": {"input_tokens": 1, "output_tokens": 1}},
    },
    "billed_units_ignored": {
        "id": "x",
        "finish_reason": "COMPLETE",
        "message": {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        "usage": {"billed_units": {"input_tokens": 9, "output_tokens": 9}},
    },
}


def _with_citations(citations) -> dict:
    return {
        "id": "x",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "citations": citations,
        },
        "usage": {"tokens": {"input_tokens": 1, "output_tokens": 2}},
    }


# v1 RAISES out of transform_response on these shapes (KeyError /
# AttributeError / TypeError / the TypedDict-splat CohereError 422); v2 must
# be a loud typed error, never a served response.
_LOUD = {
    "non_object_body": ([1, 2], "not an object"),
    "missing_message": (
        {"id": "x", "usage": {"tokens": {}}},
        "message",
    ),
    "missing_usage": (
        {"id": "x", "message": {"role": "assistant", "content": []}},
        "usage",
    ),
    "non_string_text": (
        {
            "id": "x",
            "message": {"role": "assistant", "content": [{"text": 5}]},
            "usage": {"tokens": {}},
        },
        "not a string",
    ),
    "string_content": (
        {
            "id": "x",
            "message": {"role": "assistant", "content": "plain"},
            "usage": {"tokens": {}},
        },
        "not a list",
    ),
    "null_tokens_object": (
        {
            "id": "x",
            "message": {"role": "assistant", "content": []},
            "usage": {"tokens": None},
        },
        "tokens",
    ),
    "string_token_count": (
        {
            "id": "x",
            "message": {"role": "assistant", "content": []},
            "usage": {"tokens": {"input_tokens": "3"}},
        },
        "not an integer",
    ),
    # The verifier-wave2b-beta F1 class: hostile citation shapes where v1's
    # _translate_citations_to_openai_annotations AttributeErrors out of
    # transform_response while the old skip-arms silently served an
    # annotation-less body.
    "citation_entry_int": (_with_citations([5]), "citation entry is not an object"),
    "citation_entry_string": (
        _with_citations(["x"]),
        "citation entry is not an object",
    ),
    "citations_dict": (_with_citations({"a": 1}), "not a list"),
    "citations_string": (_with_citations("abc"), "not a list"),
    "sources_dict": (
        _with_citations([{"start": 0, "end": 1, "sources": {"k": "v"}}]),
        "sources is truthy but not a list",
    ),
    "sources_string": (
        _with_citations([{"start": 0, "end": 1, "sources": "xy"}]),
        "sources is truthy but not a list",
    ),
    "source_entry_int": (
        _with_citations([{"start": 0, "end": 1, "sources": [7]}]),
        "source is not an object",
    ),
    "document_non_dict": (
        _with_citations(
            [
                {
                    "start": 0,
                    "end": 1,
                    "sources": [{"type": "document", "document": 9, "id": "s"}],
                }
            ]
        ),
        "document is not an object",
    ),
    # F8's untyped escape: a tool_call without a function key crashes BOTH
    # sides with the identical Message TypeError — typed here so the raise
    # never escapes v2's parse as a seam crash.
    "tool_call_missing_function": (
        {
            "id": "x",
            "message": {
                "role": "assistant",
                "tool_calls": [{"id": "t1", "type": "function"}],
            },
            "usage": {"tokens": {"input_tokens": 1, "output_tokens": 1}},
        },
        "no 'function' key",
    ),
}

# verifier-wave2b-beta F8: citation VALUE types v1 SERVES (the annotations
# ride a TypedDict attached by plain setattr — no validation on the non-tool
# path) but v2's typed Message construction would reject with a raw
# ValidationError. v2 falls back at the typed boundary instead; v1 serves.
_V1_SERVES_FALLBACKS = {
    "citation_start_string": (
        _with_citations(
            [
                {
                    "start": "a",
                    "end": 1,
                    "sources": [
                        {"type": "document", "document": {"title": "t"}, "id": "s"}
                    ],
                }
            ]
        ),
        "start/end/title value type",
    ),
    "citation_end_bool": (
        _with_citations(
            [
                {
                    "start": 0,
                    "end": True,
                    "sources": [
                        {"type": "document", "document": {"title": "t"}, "id": "s"}
                    ],
                }
            ]
        ),
        "start/end/title value type",
    ),
    "citation_title_int": (
        _with_citations(
            [
                {
                    "start": 0,
                    "end": 1,
                    "sources": [
                        {"type": "document", "document": {"title": 7}, "id": "s"}
                    ],
                }
            ]
        ),
        "start/end/title value type",
    ),
    "citation_url_int": (
        _with_citations(
            [
                {
                    "start": 0,
                    "end": 1,
                    "sources": [
                        {
                            "type": "document",
                            "document": {"title": "t"},
                            "id": "s",
                            "url": 5,
                        }
                    ],
                }
            ]
        ),
        "url is truthy but not a string",
    ),
}


def _v1_model_response(raw: dict) -> dict:
    logging = Logging(
        model=MODEL,
        messages=copy.deepcopy(_REQUEST["messages"]),
        stream=False,
        call_type="completion",
        start_time=time.time(),
        litellm_call_id="diff-cohere-response",
        function_id="diff-cohere-response",
    )
    response = httpx.Response(
        200,
        json=copy.deepcopy(raw),
        request=httpx.Request("POST", "https://api.cohere.com/v2/chat"),
    )
    return (
        CohereV2ChatConfig()
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


def _v2_model_response(raw: dict) -> dict:
    response = _v2_parse(raw)
    assert response.is_ok(), response.error.summary
    body = serialize_response(response.ok, build_translation_deps(), "openai")
    return to_model_response(
        body, ModelResponse(id=_AMBIENT_ID), usage_style="openai"
    ).model_dump()


def _norm(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


@pytest.mark.parametrize("name", sorted(_RESPONSES))
def test_v2_response_matches_v1(name: str, frozen_ambient) -> None:
    raw = _RESPONSES[name]
    assert _norm(_v2_model_response(raw)) == _norm(_v1_model_response(raw))


@pytest.mark.parametrize("name", sorted(_RESPONSES))
def test_request_model_and_stop_finish_quirks(name: str, frozen_ambient) -> None:
    """v1 never reads the wire finish_reason (fresh-Choices "stop" survives
    even for tool calls), keeps the REQUEST model verbatim (no prefix, no
    wire model), and ignores the wire response id."""
    v1 = _v1_model_response(_RESPONSES[name])
    v2 = _v2_model_response(_RESPONSES[name])
    for dump in (v1, v2):
        assert dump["model"] == MODEL
        assert dump["choices"][0]["finish_reason"] == "stop"
        assert dump["id"] == _AMBIENT_ID


@pytest.mark.parametrize("name", sorted(_LOUD))
def test_loud_shapes_error_on_both_sides(name: str, frozen_ambient) -> None:
    raw, fragment = _LOUD[name]
    result = _v2_parse(raw)
    assert result.is_error(), f"{name} unexpectedly parsed"
    assert fragment in result.error.summary, result.error.summary
    with pytest.raises(Exception):
        _v1_model_response(copy.deepcopy(raw))


@pytest.mark.parametrize("name", sorted(_V1_SERVES_FALLBACKS))
def test_unvalidated_citation_values_fall_back_where_v1_serves(
    name: str, frozen_ambient
) -> None:
    """F8: v1 serves the unvalidated annotation (plain setattr); v2's typed
    Message construction would raise a RAW ValidationError out of the seam —
    so the parse falls back typed and v1 keeps serving. If v1 ever starts
    validating, the serve assertion fails and the row must be re-decided."""
    raw, fragment = _V1_SERVES_FALLBACKS[name]
    result = _v2_parse(raw)
    assert result.is_error(), f"{name} unexpectedly parsed"
    assert fragment in result.error.summary, result.error.summary
    v1 = _v1_model_response(copy.deepcopy(raw))
    annotations = v1["choices"][0]["message"].get("annotations")
    assert annotations, f"{name}: v1 stopped serving the annotation; re-decide"
