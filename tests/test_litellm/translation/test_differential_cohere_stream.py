"""Differential parity for cohere v2 streaming (wave-2b-beta), pinned at the
bare-JSON line seam.

v1 side: raw JSON lines (NO ``data:`` framing, NO ``[DONE]`` — v1's
``CohereV2ModelResponseIterator`` json.loads each line directly) through the
iterator into ``CustomStreamWrapper("cohere_chat")``'s GENERIC arm. v2 side:
``fold_lines`` with the cohere parser and the shared ``generic`` chunk
dialect.

The v1 wire-vs-parser quirk drives TWO regimes, both pinned:

- REAL-WIRE (``type``-keyed events, what Cohere v2 actually sends): the
  parser's message-end arm never fires (it reads ``event``), so the wrapper
  SYNTHESIZES the trailing finish chunk at StopIteration — "tool_calls"
  when tool deltas were seen, else "stop". v2 == v1 minus that synthesized
  tail; the tail is the streaming seam's obligation (asserted per row).
- EVENT-keyed message-end (the shape the parser was written for): finish is
  mapped (COMPLETE -> stop, MAX_TOKENS -> length, ERROR_TOXIC ->
  content_filter, anything else -> stop) and emitted VERBATIM — no
  seen-tool-calls rewrite — and wire usage rides; v1 swallows it without
  ``stream_options.include_usage`` and emits a final usage chunk with the
  WIRE values under it, while v2 passes a ``choices: []`` usage tail through
  (the family seam contract).

Envelope contracts the adapter below documents (generic-arm seam
obligations, distinct from the openai/xai dialect): v1 mints a FRESH
chatcmpl id per chunk (ids are normalized here and the freshness pinned),
sets NO ``system_fingerprint``/``citations`` keys, and re-reads no wire id.
"""

import copy
import json
import time

import pytest

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.cohere.common_utils import CohereV2ModelResponseIterator
from litellm.types.utils import ModelResponseStream, Usage

from litellm.translation.engine.stream import fold_events, fold_lines
from litellm.translation.inbound.openai_chat.stream import initial_state
from litellm.translation.providers.cohere import parse_event, parse_line

MODEL = "command-r"

_USAGE_EVENT = {
    "event": "message-end",
    "data": {
        "delta": {
            "finish_reason": "COMPLETE",
            "usage": {"tokens": {"input_tokens": 3, "output_tokens": 2}},
        }
    },
}

# regime 1: type-keyed (real wire) — v1 synthesizes the trailing finish.
REAL_WIRE_STREAMS = {
    "text": (
        [
            {"type": "message-start", "delta": {"message": {"role": "assistant"}}},
            {
                "type": "content-delta",
                "delta": {"message": {"content": {"text": "Hel"}}},
            },
            {
                "type": "content-delta",
                "delta": {"message": {"content": {"text": "lo"}}},
            },
            {
                "type": "message-end",
                "data": {
                    "delta": {
                        "finish_reason": "COMPLETE",
                        "usage": {"tokens": {"input_tokens": 3, "output_tokens": 2}},
                    }
                },
            },
        ],
        "stop",
    ),
    "tools": (
        [
            {
                "type": "tool-call-start",
                "delta": {"message": {"tool_calls": {"id": "c1", "type": "function"}}},
            },
            {
                "type": "tool-call-delta",
                "delta": {
                    "tool_calls": [
                        {"id": "c1", "name": "get_weather", "arguments": '{"city"'}
                    ]
                },
            },
            {
                "type": "tool-call-delta",
                "delta": {"tool_calls": [{"arguments": ':"Paris"}'}]},
            },
            {"type": "message-end", "data": {"delta": {"finish_reason": "TOOL_CALL"}}},
        ],
        "tool_calls",
    ),
}

# regime 2: event-keyed shapes — exact parity (and the usage-tail contract).
EVENT_KEYED_STREAMS = {
    "plan_citation_content": [
        {
            "event": "tool-plan-delta",
            "data": {"delta": {"message": {"tool_plan": "plan"}}},
        },
        {"type": "content-delta", "delta": {"message": {"content": {"text": "Hi"}}}},
        {
            "event": "citation-start",
            "data": {
                "delta": {
                    "message": {
                        "citations": {
                            "start": 0,
                            "end": 2,
                            "text": "Hi",
                            "sources": [
                                {
                                    "type": "document",
                                    "id": "d1",
                                    "document": {"title": "T"},
                                }
                            ],
                        }
                    }
                }
            },
        },
        {"event": "message-end", "data": {"delta": {"finish_reason": "COMPLETE"}}},
    ],
    "max_tokens_maps_to_length": [
        {"type": "content-delta", "delta": {"message": {"content": {"text": "Hi"}}}},
        {"event": "message-end", "data": {"delta": {"finish_reason": "MAX_TOKENS"}}},
    ],
    "error_toxic_maps_to_content_filter": [
        {"type": "content-delta", "delta": {"message": {"content": {"text": "Hi"}}}},
        {"event": "message-end", "data": {"delta": {"finish_reason": "ERROR_TOXIC"}}},
    ],
    "tool_then_event_end_no_rewrite": [
        {
            "type": "tool-call-delta",
            "delta": {"tool_calls": [{"id": "c", "name": "f", "arguments": "{}"}]},
        },
        {"event": "message-end", "data": {"delta": {"finish_reason": "TOOL_CALL"}}},
    ],
}

USAGE_STREAM = [
    {"type": "content-delta", "delta": {"message": {"content": {"text": "Hi"}}}},
    _USAGE_EVENT,
]


def _make_logging() -> Logging:
    return Logging(
        model=MODEL,
        messages=[{"role": "user", "content": "stream"}],
        stream=True,
        call_type="completion",
        start_time=time.time(),
        litellm_call_id="diff-cohere-stream",
        function_id="diff-cohere-stream",
    )


def _v1_chunks(events: list, stream_options: dict | None = None) -> list:
    lines = [json.dumps(event) for event in copy.deepcopy(events)]
    handler = CohereV2ModelResponseIterator(
        streaming_response=iter(lines), sync_stream=True
    )
    wrapper = CustomStreamWrapper(
        completion_stream=handler,
        model=MODEL,
        custom_llm_provider="cohere_chat",
        logging_obj=_make_logging(),
        stream_options=stream_options,
    )
    return [chunk.model_dump() for chunk in wrapper]


def _to_stream_chunk(body: dict) -> dict:
    """The generic-arm chunk envelope a future cohere streaming seam must
    build (and the reason ``to_model_response_stream`` is NOT used here):
    v1's wrapper constructs a FRESH ModelResponseStream per chunk — ambient
    per-chunk id and created, NO citations/system_fingerprint presets — and
    a body-carried usage becomes the verbatim ``Usage(**dict)``."""
    payload = dict(body)
    usage_payload = payload.pop("usage", None)
    chunk = ModelResponseStream(**payload)
    if isinstance(usage_payload, dict):
        setattr(chunk, "usage", Usage(**usage_payload))
    return chunk.model_dump()


def _v2_chunks(events: list) -> list:
    lines = [json.dumps(event) for event in copy.deepcopy(events)]
    folded = fold_lines(lines, parse_line, initial_state(MODEL, dialect="generic"))
    assert folded.is_ok(), folded.error.summary
    return [_to_stream_chunk(chunk) for chunk in folded.ok]


def _norm(chunks: list) -> str:
    """Byte-compare with ids normalized: v1 mints a FRESH chatcmpl id per
    chunk (envelope nondeterminism, pinned by
    test_v1_mints_fresh_ids_per_chunk); everything else must be identical."""
    normalized = []
    for chunk in chunks:
        assert str(chunk["id"]).startswith("chatcmpl-")
        normalized.append({**chunk, "id": "ID"})
    return json.dumps(normalized, sort_keys=True, default=str)


@pytest.mark.parametrize("name", sorted(REAL_WIRE_STREAMS))
def test_real_wire_v2_matches_v1_minus_synthesized_tail(
    name: str, frozen_ambient
) -> None:
    events, synth_finish = REAL_WIRE_STREAMS[name]
    v1 = _v1_chunks(events)
    v2 = _v2_chunks(events)
    assert len(v1) == len(v2) + 1
    assert _norm(v2) == _norm(v1[:-1])
    tail = v1[-1]["choices"][0]
    assert tail["finish_reason"] == synth_finish
    assert tail["delta"]["content"] is None and tail["delta"]["tool_calls"] is None


@pytest.mark.parametrize("name", sorted(EVENT_KEYED_STREAMS))
def test_event_keyed_v2_matches_v1(name: str, frozen_ambient) -> None:
    events = EVENT_KEYED_STREAMS[name]
    assert _norm(_v2_chunks(events)) == _norm(_v1_chunks(events))


def test_usage_tail_pins_the_seam_contract(frozen_ambient) -> None:
    """v2 passes the wire usage through on a ``choices: []`` tail; v1
    swallows it without include_usage and emits its final usage chunk (the
    WIRE values for this event-keyed shape) under it."""
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


def test_real_wire_include_usage_synthesis_is_estimated(frozen_ambient) -> None:
    """On the real (type-keyed) wire the message-end usage NEVER reaches the
    wrapper, so v1's include_usage final chunk carries token-counter
    ESTIMATES, not the wire 3/2 — the seam obligation the docstring names."""
    events, _ = REAL_WIRE_STREAMS["text"]
    v1 = _v1_chunks(events, stream_options={"include_usage": True})
    usage = v1[-1]["usage"]
    assert usage is not None
    assert (usage["prompt_tokens"], usage["completion_tokens"]) != (3, 2)


def test_v1_mints_fresh_ids_per_chunk(frozen_ambient) -> None:
    v1 = _v1_chunks(EVENT_KEYED_STREAMS["plan_citation_content"])
    ids = [chunk["id"] for chunk in v1]
    assert len(set(ids)) == len(ids)


def test_v2_line_and_event_folds_agree(frozen_ambient) -> None:
    events = EVENT_KEYED_STREAMS["plan_citation_content"]
    folded = fold_events(
        copy.deepcopy(events), parse_event, initial_state(MODEL, dialect="generic")
    )
    assert folded.is_ok(), folded.error.summary
    via_events = [_to_stream_chunk(chunk) for chunk in folded.ok]
    assert _norm(via_events) == _norm(_v2_chunks(events))


_LOUD_CHUNKS = {
    "non_dict_tool_call_entry": (
        {"type": "tool-call-delta", "delta": {"tool_calls": ["bare"]}},
        "tool_calls[0]",
    ),
    "non_dict_delta": (
        {"type": "tool-call-delta", "delta": "x"},
        "not an object",
    ),
    "null_usage_tokens": (
        {
            "event": "message-end",
            "data": {"delta": {"finish_reason": "COMPLETE", "usage": {"tokens": None}}},
        },
        "tokens",
    ),
    "non_numeric_index": (
        {"index": "x", "type": "content-delta"},
        "index",
    ),
}


@pytest.mark.parametrize("name", sorted(_LOUD_CHUNKS))
def test_loud_chunk_shapes_error_on_both_sides(name: str, frozen_ambient) -> None:
    """v1's chunk_parser raises ValueError -> RuntimeError out of the
    iterator on each of these; v2 is a loud typed error, never a served
    chunk."""
    event, fragment = _LOUD_CHUNKS[name]
    result = parse_event(copy.deepcopy(event))
    assert result.is_error(), f"{name} unexpectedly parsed"
    assert fragment in result.error.summary, result.error.summary
    with pytest.raises(Exception):
        _v1_chunks([event])


def test_non_string_text_pinned_divergence(frozen_ambient) -> None:
    """PINNED DIVERGENCE (fail-closed on a failure path, the compat_httpx
    error-chunk precedent): a non-str ``content.text`` is SILENTLY SWALLOWED
    by v1 (the wrapper drops the chunk and only the synthesized finish
    emerges); v2 errors loudly naming the shape. If v1 ever starts serving
    or raising on it, the first assertion fails and the row must be
    re-decided."""
    event = {"type": "content-delta", "delta": {"message": {"content": {"text": 5}}}}
    v1 = _v1_chunks([event])
    assert all(
        choice["delta"]["content"] in (None, "")
        for chunk in v1
        for choice in chunk["choices"]
    ), "v1 started serving the non-str text; re-decide the pinned divergence"
    result = parse_event(copy.deepcopy(event))
    assert result.is_error()
    assert "not a string" in result.error.summary


def test_non_string_finish_defaults_to_stop_like_v1(frozen_ambient) -> None:
    """v1's map_finish_reason defaults every unmapped value — non-strings
    included — to "stop" (probed); the v2 parser mirrors the default."""
    events = [{"event": "message-end", "data": {"delta": {"finish_reason": 5}}}]
    v1 = _v1_chunks(events)
    v2 = _v2_chunks(events)
    assert _norm(v2) == _norm(v1)
    assert v2[-1]["choices"][0]["finish_reason"] == "stop"


def test_non_json_line_is_loud_on_both_sides(frozen_ambient) -> None:
    """No data:/[DONE] framing on this wire: v1 json.loads every line and
    raises RuntimeError on a non-JSON one."""
    result = parse_line("data: {}")
    assert result.is_error()
    assert "non-JSON" in result.error.summary
    handler = CohereV2ModelResponseIterator(
        streaming_response=iter(["data: {}"]), sync_stream=True
    )
    with pytest.raises(Exception):
        next(handler)
