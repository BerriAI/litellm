"""Differential parity for the xai response path.

The v1 reference is ``XAIChatConfig.transform_response`` — LIVE on the
httpx path (main.py:2289), the inverse of the openai SDK route — over a
real ``httpx.Response``. Two-sided: v1 at HEAD must equal the committed
snapshot (drift guard) AND v2 (``parse_response`` ->
``serialize_response("openai")`` -> ``to_model_response("openai")``) must
equal it byte-for-byte.

The corpus pins the five xai response behaviors: the R1 finish_reason ""
chain (v1's own ``_fix_choice_finish_reason_for_tool_calls`` is dead;
v1-as-executed emits "stop" WITH tool_calls — both sides run the same live
``map_finish_reason`` via ``Choices``), the reasoning-token fold (+ its
idempotency guard), num_sources_used -> web_search_requests (live-search
billing), total_tokens normalization, and the citations top-level
passthrough. The R4 row proves the bare wire model: NO ``xai/`` prefix ever
appears (fresh ModelResponse, the cdr ``model is None`` arm).
"""

import copy
import json

import pytest

from litellm.types.utils import ModelResponse

from litellm.translation.inbound.openai_chat import parse_request
from litellm.translation.inbound.openai_chat.response import serialize_response
from litellm.translation.providers.xai.response import parse_response
from litellm.translation_seam import build_translation_deps, to_model_response

from ._xai_corpus import (
    SNAPSHOTS_DIR,
    canonical_json,
    corpus,
    jsonable,
    run_v1_response_transform,
)

RESPONSES = corpus("responses")

_REQUEST = {
    "model": "grok-4-0709",
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


def _v2_model_response(raw: dict, request_model: str) -> dict:
    request = {**copy.deepcopy(_REQUEST), "model": request_model}
    parsed = parse_request(request)
    assert parsed.is_ok(), parsed.error.summary
    response = parse_response(copy.deepcopy(raw), parsed.ok)
    assert response.is_ok(), response.error.summary
    body = serialize_response(response.ok, build_translation_deps(), "openai")
    return to_model_response(body, ModelResponse(), usage_style="openai").model_dump()


def _norm(payload: object) -> str:
    return json.dumps(jsonable(payload), sort_keys=True)


@pytest.mark.parametrize("name", sorted(RESPONSES))
def test_v1_at_head_still_matches_the_snapshot(name: str) -> None:
    row = RESPONSES[name]
    snapshot = (SNAPSHOTS_DIR / "responses" / f"{name}.json").read_text()
    assert canonical_json(run_v1_response_transform(row["body"], row["model"])) == (
        snapshot
    )


@pytest.mark.parametrize("name", sorted(RESPONSES))
def test_v2_response_matches_the_snapshot(name: str) -> None:
    row = RESPONSES[name]
    snapshot = (SNAPSHOTS_DIR / "responses" / f"{name}.json").read_text()
    assert canonical_json(_v2_model_response(row["body"], row["model"])) == snapshot


def test_finish_empty_string_yields_stop_with_tool_calls() -> None:
    """R1 pinned semantically, not just byte-wise: the served finish is
    "stop" AND the tool calls survive (the violated stop->tool_calls
    invariant v1-as-executed exhibits)."""
    row = RESPONSES["finish_empty_string_with_tool_calls"]
    dumped = _v2_model_response(row["body"], row["model"])
    choice = dumped["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["tool_calls"], "tool_calls must survive the '' finish"


def test_no_xai_prefix_on_the_response_model() -> None:
    """R4: the xai httpx path starts from a FRESH ModelResponse (model=None,
    main.py:1401) and adopts the bare wire model; no seam may prefix it."""
    row = RESPONSES["text_basic"]
    v1 = run_v1_response_transform(row["body"], row["model"]).model_dump()
    v2 = _v2_model_response(row["body"], row["model"])
    assert v1["model"] == v2["model"] == "grok-4-0709"
    assert "/" not in v2["model"]


def test_websearch_billing_fields_reach_usage() -> None:
    """The live-search billing hook: cost_per_web_search_request reads
    usage.prompt_tokens_details.web_search_requests."""
    row = RESPONSES["websearch_sources_and_citations"]
    dumped = _v2_model_response(row["body"], row["model"])
    assert dumped["usage"]["prompt_tokens_details"]["web_search_requests"] == 3
    assert dumped["citations"] == ["https://a.test", "https://b.test"]


_UNSUPPORTED = {
    "multiple_choices": (
        {
            "id": "r",
            "created": 1,
            "model": "grok-4-0709",
            "choices": [
                {"index": 0, "finish_reason": "stop", "message": {"content": "a"}},
                {"index": 1, "finish_reason": "stop", "message": {"content": "b"}},
            ],
        },
        "multiple response choices",
    ),
    "legacy_function_call_output": (
        {
            "id": "r",
            "created": 1,
            "model": "grok-4-0709",
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
        },
        "function_call",
    ),
}


@pytest.mark.parametrize("name", sorted(_UNSUPPORTED))
def test_unreachable_response_shape_is_a_typed_error(name: str) -> None:
    raw, reason_fragment = _UNSUPPORTED[name]
    parsed = parse_request(copy.deepcopy(_REQUEST))
    assert parsed.is_ok()
    result = parse_response(copy.deepcopy(raw), parsed.ok)
    assert result.is_error(), f"{name} unexpectedly parsed"
    assert reason_fragment in result.error.summary, result.error.summary


def test_uncoercible_usage_is_loud_on_both_sides() -> None:
    """v1 raises out of cdr's Usage validation for an uncoercible token
    value; v2 must return a boundary error, never silently treat it as 0
    (critic-grok M2). The numeric-string direction is the two-sided corpus
    row numeric_string_usage_coerced."""
    raw = {
        "id": "resp-bad",
        "object": "chat.completion",
        "created": 1718000011,
        "model": "grok-3-mini",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "logprobs": None,
                "message": {"role": "assistant", "content": "a"},
            }
        ],
        "usage": {
            "prompt_tokens": "abc",
            "completion_tokens": 2,
            "total_tokens": 9,
            "completion_tokens_details": {"reasoning_tokens": 7},
        },
    }
    from ._xai_corpus import run_v1_response_transform

    with pytest.raises(Exception):
        run_v1_response_transform(raw, "grok-3-mini")
    parsed = parse_request(copy.deepcopy(_REQUEST))
    assert parsed.is_ok()
    result = parse_response(copy.deepcopy(raw), parsed.ok)
    assert result.is_error(), "uncoercible usage must be loud, not a silent 0"
    assert "not int-coercible" in result.error.summary, result.error.summary
