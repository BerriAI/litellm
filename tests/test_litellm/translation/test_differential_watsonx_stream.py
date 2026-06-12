"""Differential parity for watsonx streaming (wave-2b-beta), pinned at the
SSE line seam.

v1 side: lines through the databricks ``ModelResponseIterator`` (researcher-4
DRIFT: watsonx streams yield ``GenericStreamingChunk`` dicts, NOT the plain
openai dialect) into ``CustomStreamWrapper("watsonx")``'s GENERIC arm —
constructed by the openai_like handler with the BARE wire model. v2 side:
``fold_lines`` with the watsonx parser and the shared ``generic`` chunk
dialect, adapted by the same gate-local envelope the cohere gate documents
(per-chunk fresh ids — normalized here — and no citations preset).

Streams that end without a wire finish get the wrapper's SYNTHESIZED
trailing finish chunk (seam scope, the cohere precedent); streams with an
explicit finish chunk match byte-for-byte. PINNED DIVERGENCES: v1's
iterator SILENTLY SWALLOWS non-JSON lines and chunks failing
ModelResponseStream validation (pydantic ValidationError is a ValueError,
eaten by the iterator's except arm) — v2 errors loudly on both.
"""

import copy
import json
import time

import pytest

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.databricks.streaming_utils import ModelResponseIterator
from litellm.types.utils import ModelResponseStream, Usage

from litellm.translation.engine.stream import fold_events, fold_lines
from litellm.translation.inbound.openai_chat.stream import initial_state
from litellm.translation.providers.watsonx import parse_event, parse_line

MODEL = "ibm/granite-3-8b-instruct"

_USAGE = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}


def _chunk(
    delta: dict | None = None,
    finish: str | None = None,
    usage: dict | None = None,
    choices: list | None = None,
) -> dict:
    payload: dict = {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 1718000000,
        "model": MODEL,
    }
    payload["choices"] = (
        choices
        if choices is not None
        else [
            {
                "index": 0,
                "delta": delta if delta is not None else {},
                "finish_reason": finish,
            }
        ]
    )
    if usage is not None:
        payload["usage"] = usage
    return payload


STREAMS = {
    "text": [
        _chunk({"role": "assistant", "content": "He"}),
        _chunk({"content": "llo"}),
        _chunk({}, finish="stop"),
    ],
    "content_and_finish_in_one_wire_chunk": [
        # v1 emits the content chunk, then the finish flush as its own chunk
        _chunk({"role": "assistant", "content": "Hi"}, finish="stop"),
    ],
    "tools": [
        _chunk(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "f", "arguments": ""},
                    }
                ],
            }
        ),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}),
        _chunk({}, finish="tool_calls"),
    ],
    "name_only_tool_start_rides_with_empty_arguments": [
        # validation defaults missing/null arguments to "", so the
        # iterator's `arguments is not None` check never fails
        _chunk(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "c",
                        "type": "function",
                        "function": {"name": "f"},
                    }
                ],
            }
        ),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}),
        _chunk({}, finish="tool_calls"),
    ],
    "mid_stream_usage_stripped": [
        _chunk({"role": "assistant", "content": "Hi"}, usage=_USAGE),
        _chunk({}, finish="stop"),
    ],
    "ibm_time_limit_maps_to_stop": [
        _chunk({"role": "assistant", "content": "Hi"}),
        _chunk({}, finish="time_limit"),
    ],
}

USAGE_STREAM = [
    _chunk({"role": "assistant", "content": "Hi"}),
    _chunk({}, finish="stop"),
    _chunk(choices=[], usage=_USAGE),
]


def _v1_chunks(
    events: list | None,
    stream_options: dict | None = None,
    raw_lines: list | None = None,
) -> list:
    lines = (
        raw_lines
        if raw_lines is not None
        else [f"data: {json.dumps(event)}" for event in copy.deepcopy(events or [])]
        + ["data: [DONE]"]
    )
    handler = ModelResponseIterator(streaming_response=iter(lines), sync_stream=True)
    logging = Logging(
        model=MODEL,
        messages=[{"role": "user", "content": "stream"}],
        stream=True,
        call_type="completion",
        start_time=time.time(),
        litellm_call_id="diff-watsonx-stream",
        function_id="diff-watsonx-stream",
    )
    wrapper = CustomStreamWrapper(
        completion_stream=handler,
        model=MODEL,
        custom_llm_provider="watsonx",
        logging_obj=logging,
        stream_options=stream_options,
    )
    return [chunk.model_dump() for chunk in wrapper]


def _to_stream_chunk(body: dict) -> dict:
    """The generic-arm chunk envelope (the cohere gate precedent): fresh
    ModelResponseStream per chunk, no citations/system_fingerprint preset,
    usage as the verbatim ``Usage(**dict)``."""
    payload = dict(body)
    usage_payload = payload.pop("usage", None)
    chunk = ModelResponseStream(**payload)
    if isinstance(usage_payload, dict):
        setattr(chunk, "usage", Usage(**usage_payload))
    return chunk.model_dump()


def _v2_chunks(events: list | None, raw_lines: list | None = None) -> list:
    lines = (
        raw_lines
        if raw_lines is not None
        else [f"data: {json.dumps(event)}" for event in copy.deepcopy(events or [])]
        + ["data: [DONE]"]
    )
    folded = fold_lines(lines, parse_line, initial_state(MODEL, dialect="generic"))
    assert folded.is_ok(), folded.error.summary
    return [_to_stream_chunk(chunk) for chunk in folded.ok]


def _norm(chunks: list) -> str:
    normalized = []
    for chunk in chunks:
        assert str(chunk["id"]).startswith("chatcmpl-")
        normalized.append({**chunk, "id": "ID"})
    return json.dumps(normalized, sort_keys=True, default=str)


@pytest.mark.parametrize("name", sorted(STREAMS))
def test_v2_stream_matches_v1(name: str, frozen_ambient) -> None:
    events = STREAMS[name]
    assert _norm(_v2_chunks(events)) == _norm(_v1_chunks(events))


def test_no_wire_finish_gets_the_synthesized_tail(frozen_ambient) -> None:
    """A stream ending without a finish chunk: v1 synthesizes the trailing
    "stop" at StopIteration (seam scope, the cohere contract)."""
    events = [_chunk({"role": "assistant", "content": "Hi"})]
    v1 = _v1_chunks(events)
    v2 = _v2_chunks(events)
    assert len(v1) == len(v2) + 1
    assert _norm(v2) == _norm(v1[:-1])
    assert v1[-1]["choices"][0]["finish_reason"] == "stop"


def test_raw_json_lines_parse_without_data_prefix(frozen_ambient) -> None:
    """v1's _strip_sse_data_from_chunk leaves non-prefixed lines alone and
    the iterator json.loads them — the line seam is data:-OPTIONAL."""
    raw = [
        json.dumps(_chunk({"role": "assistant", "content": "raw"})),
        json.dumps(_chunk({}, finish="stop")),
        "data: [DONE]",
    ]
    assert _norm(_v2_chunks(None, raw_lines=raw)) == _norm(
        _v1_chunks(None, raw_lines=raw)
    )


def test_usage_tail_pins_the_seam_contract(frozen_ambient) -> None:
    v1 = _v1_chunks(USAGE_STREAM, stream_options={"include_usage": True})
    v2 = _v2_chunks(USAGE_STREAM)
    assert len(v1) == len(v2)
    assert _norm(v2[:-1]) == _norm(v1[: len(v2) - 1])
    assert v2[-1]["choices"] == []
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        assert v1[-1]["usage"][key] == v2[-1]["usage"][key], key


def test_usage_tail_swallowed_without_stream_options(frozen_ambient) -> None:
    v1 = _v1_chunks(USAGE_STREAM, None)
    v2 = _v2_chunks(USAGE_STREAM)
    assert len(v1) == len(v2) - 1
    assert _norm(v1) == _norm(v2[:-1])


def test_v2_line_and_event_folds_agree(frozen_ambient) -> None:
    events = STREAMS["tools"]
    folded = fold_events(
        copy.deepcopy(events), parse_event, initial_state(MODEL, dialect="generic")
    )
    assert folded.is_ok(), folded.error.summary
    via_events = [_to_stream_chunk(chunk) for chunk in folded.ok]
    assert _norm(via_events) == _norm(_v2_chunks(events))


_PINNED_DIVERGENCES = {
    "non_json_line": (None, ["data: {notjson", "data: [DONE]"], "non-JSON"),
    "non_string_content": (
        [_chunk({"role": "assistant", "content": 5})],
        None,
        "not a string",
    ),
    "missing_choices": (
        [{"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": MODEL}],
        None,
        "choices",
    ),
    "tool_call_without_function": (
        [_chunk({"role": "assistant", "tool_calls": [{"index": 0, "id": "c"}]})],
        None,
        "function",
    ),
}


@pytest.mark.parametrize("name", sorted(_PINNED_DIVERGENCES))
def test_pinned_divergences_v1_swallows_v2_loud(name: str, frozen_ambient) -> None:
    """PINNED DIVERGENCE rows (fail-closed on failure paths): v1's iterator
    silently swallows each shape (the except-ValueError arm eats pydantic
    ValidationErrors and JSON decode errors alike) and the stream serves on
    with the chunk LOST; v2 is a loud typed error. If v1 ever starts
    serving or raising, the swallow assertion fails and the rows must be
    re-decided."""
    events, raw_lines, fragment = _PINNED_DIVERGENCES[name]
    v1 = _v1_chunks(events, raw_lines=raw_lines)
    assert all(
        choice["delta"]["content"] in (None, "") and not choice["delta"]["tool_calls"]
        for chunk in v1
        for choice in chunk["choices"]
    ), f"{name}: v1 stopped swallowing; re-decide the pinned divergence"
    if raw_lines is not None:
        result = parse_line(raw_lines[0])
    else:
        result = parse_event(copy.deepcopy(events[0]))
    assert result.is_error(), f"{name} unexpectedly parsed"
    assert fragment in result.error.summary, result.error.summary
