"""Differential parity for the databricks stream path (wave 3).

v1 side: ``DatabricksChatResponseIterator.chunk_parser`` (a
``BaseModelResponseIterator``), replayed in-process at HEAD. v1 yields ONE
``ModelResponseStream`` per wire chunk (no split) and DROPS the wire ``usage``
ENTIRELY (DB-R5: the iterator never passes usage through), so the v2 fold never
attaches a usage tail. A content block-list flattens to a string,
reasoning/summary blocks become reasoning_content + thinking_blocks, a first-
item ``citations`` list lifts to ``provider_specific_fields.citation``, and a
``{}`` tool argument string is rewritten to ``""``. Under json_mode (the
STATEFUL ``_last_function_name`` machine) a ``json_tool_call`` tool delta folds
to content with the json.loads+dumps byte REFORMAT (DB-R8). json_mode is a
REQUEST-side fallback (v2 never sends a json_mode request), so that arm is
dormant from v2's own flow but pinned here against the REAL iterator.

The line seam (``parse_line``): an SSE ``data:`` prefix is stripped, ``[DONE]``
ends the stream, and a line that fails ``json.loads`` (or an empty line) is
SILENTLY SWALLOWED by v1's base iterator; v2 errors loudly — the watsonx/ollama
line-seam PINNED DIVERGENCE (fail-closed on a failure path, named report row).
"""

import copy
import json

import pytest

from litellm.llms.databricks.chat.transformation import DatabricksChatResponseIterator

from litellm.translation.engine.stream import fold_events, fold_lines
from litellm.translation.inbound.openai_chat.stream import StreamState, initial_state
from litellm.translation.providers.databricks import parse_event, parse_line

MODEL = "databricks-dbrx-instruct"


def _chunk(delta: dict, finish=None, index: int = 0) -> dict:
    return {
        "id": "chatcmpl-stream",
        "created": 1718000000,
        "model": "dbrx",
        "choices": [{"index": index, "delta": delta, "finish_reason": finish}],
    }


STREAMS = {
    "content": [
        _chunk({"role": "assistant", "content": "Hel"}),
        _chunk({"content": "lo"}),
        _chunk({}, finish="stop"),
    ],
    "content_list_flattens": [
        _chunk(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "a"},
                    {"type": "text", "text": "b"},
                ],
            }
        ),
        _chunk({}, finish="stop"),
    ],
    "reasoning_summary": [
        _chunk(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "reasoning",
                        "summary": [
                            {"type": "summary_text", "text": "think", "signature": "s"}
                        ],
                    }
                ],
            }
        ),
        _chunk({}, finish="stop"),
    ],
    "reasoning_summary_missing_signature": [
        _chunk(
            {
                "role": "assistant",
                "content": [
                    {"type": "reasoning", "summary": [{"type": "summary_text", "text": "t"}]}
                ],
            }
        ),
        _chunk({}, finish="stop"),
    ],
    "citations_lift": [
        _chunk(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "ans", "citations": [{"source": "d1"}]}
                ],
            }
        ),
        _chunk({}, finish="stop"),
    ],
    "tool_call": [
        _chunk(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "f", "arguments": '{"a": 1}'},
                    }
                ],
            }
        ),
        _chunk({}, finish="tool_calls"),
    ],
    "tool_call_empty_args_blanked": [
        _chunk(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
            }
        ),
        _chunk({}, finish="tool_calls"),
    ],
    "usage_chunk_dropped": [
        _chunk({"role": "assistant", "content": "x"}),
        {
            **_chunk({}, finish="stop"),
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        },
    ],
}

# json_mode: the STATEFUL json_tool_call -> content byte-reformat (DB-R8). v2
# never SENDS a json_mode request (it falls back), so the live state flag is
# always False; the gate sets it True to pin v1's reformat against the REAL
# iterator.
_JSON_MODE_STREAM = [
    _chunk(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "index": 0,
                    "id": "t",
                    "type": "function",
                    "function": {"name": "json_tool_call", "arguments": '{"a":1}'},
                }
            ],
        }
    ),
    _chunk({}, finish="stop"),
]


def _v1_chunks(events: list, json_mode: bool = False) -> list:
    iterator = DatabricksChatResponseIterator(
        streaming_response=iter([]), sync_stream=True, json_mode=json_mode
    )
    return [
        iterator.chunk_parser(copy.deepcopy(event)).model_dump() for event in events
    ]


def _v2_state(json_mode: bool) -> StreamState:
    base = initial_state(MODEL, dialect="databricks")
    return base if not json_mode else _with_json_mode(base)


def _with_json_mode(state: StreamState) -> StreamState:
    from dataclasses import replace

    return replace(state, databricks_json_mode=True)


def _v2_bodies(events: list, json_mode: bool = False) -> list:
    folded = fold_events(
        copy.deepcopy(events), parse_event, _v2_state(json_mode)
    )
    assert folded.is_ok(), folded.error.summary
    return list(folded.ok)


def _v1_delta(chunk: dict) -> dict:
    return chunk["choices"][0]["delta"]


def _v2_delta(body: dict) -> dict:
    return body["choices"][0]["delta"]


def _norm_delta(delta: dict) -> str:
    """v1's ModelResponseStream Delta materializes a fixed key set
    (function_call/audio/etc. defaulted to None); v2 emits the normalized
    fold body. Compare the keys the fold OWNS — content, reasoning_content,
    thinking_blocks, tool_calls, role, provider_specific_fields — which is the
    DB-R5/R8 parity surface (the Delta envelope defaults are seam scope)."""
    keys = (
        "content",
        "reasoning_content",
        "thinking_blocks",
        "tool_calls",
        "role",
        "provider_specific_fields",
    )
    return json.dumps(
        {key: delta.get(key) for key in keys}, sort_keys=True, default=str
    )


@pytest.mark.parametrize("name", sorted(STREAMS))
def test_v2_stream_delta_matches_v1(name: str) -> None:
    events = STREAMS[name]
    v1 = _v1_chunks(events)
    v2 = _v2_bodies(events)
    assert len(v1) == len(v2)
    for v1_chunk, v2_body in zip(v1, v2):
        assert _norm_delta(_v1_delta(v1_chunk)) == _norm_delta(_v2_delta(v2_body))


@pytest.mark.parametrize("name", sorted(STREAMS))
def test_one_body_per_wire_chunk_no_usage_tail(name: str) -> None:
    """DB-R5: v1 yields exactly one out-chunk per wire chunk and NEVER attaches
    usage (even the usage-bearing wire chunk produces a usage-less body)."""
    events = STREAMS[name]
    v2 = _v2_bodies(events)
    assert len(v2) == len(events)
    for body in v2:
        assert body.get("usage") is None


def test_usage_is_dropped_entirely_both_sides() -> None:
    events = STREAMS["usage_chunk_dropped"]
    v1 = _v1_chunks(events)
    v2 = _v2_bodies(events)
    for v1_chunk in v1:
        assert getattr(v1_chunk, "usage", None) is None or v1_chunk.get("usage") is None
    for body in v2:
        assert body.get("usage") is None


def test_empty_args_rewritten_to_blank_string() -> None:
    events = STREAMS["tool_call_empty_args_blanked"]
    v1 = _v1_chunks(events)
    v1_args = _v1_delta(v1[0])["tool_calls"][0]["function"]["arguments"]
    assert v1_args == ""
    v2_args = _v2_delta(_v2_bodies(events)[0])["tool_calls"][0]["function"]["arguments"]
    assert v2_args == ""


def test_json_mode_byte_reformat_db_r8() -> None:
    """DB-R8: under json_mode the json_tool_call delta becomes content with the
    json.loads+dumps reformat ({"a":1} -> {"a": 1}) and tool_calls nulled.
    Replayed against the REAL iterator with json_mode=True (v2 never sends a
    json_mode request, so this pins the dormant arm)."""
    v1 = _v1_chunks(_JSON_MODE_STREAM, json_mode=True)
    v2 = _v2_bodies(_JSON_MODE_STREAM, json_mode=True)
    v1_delta = _v1_delta(v1[0])
    v2_delta = _v2_delta(v2[0])
    assert v1_delta["content"] == '{"a": 1}'
    assert v1_delta.get("tool_calls") is None
    assert _norm_delta(v1_delta) == _norm_delta(v2_delta)


def test_json_mode_off_keeps_the_tool_call() -> None:
    """The discriminator for DB-R8: WITHOUT json_mode the same json_tool_call
    delta rides as a tool_call (no content reformat) — so the byte-reformat is
    genuinely gated on the stateful flag."""
    v1 = _v1_chunks(_JSON_MODE_STREAM, json_mode=False)
    v2 = _v2_bodies(_JSON_MODE_STREAM, json_mode=False)
    assert _v1_delta(v1[0]).get("tool_calls") is not None
    assert _norm_delta(_v1_delta(v1[0])) == _norm_delta(_v2_delta(v2[0]))


_PINNED_DIVERGENCES = {
    "non_json_line": (["data: {notjson", "data: [DONE]"], "non-JSON"),
    "empty_line": (["data: ", "data: [DONE]"], "empty"),
}


@pytest.mark.parametrize("name", sorted(_PINNED_DIVERGENCES))
def test_line_seam_pinned_divergence_loud_where_v1_swallows(name: str) -> None:
    """v1's base iterator silently swallows a non-JSON or empty line; v2 errors
    loudly (the fail-closed line-seam divergence, named report row)."""
    raw_lines, fragment = _PINNED_DIVERGENCES[name]
    folded = fold_lines(
        list(raw_lines), parse_line, initial_state(MODEL, dialect="databricks")
    )
    assert folded.is_error(), f"{name} unexpectedly folded"
    assert fragment in folded.error.summary, folded.error.summary


def test_data_prefix_stripped_and_done_ends_stream() -> None:
    raw = [
        f"data: {json.dumps(_chunk({'role': 'assistant', 'content': 'hi'}))}",
        f"data: {json.dumps(_chunk({}, finish='stop'))}",
        "data: [DONE]",
    ]
    folded = fold_lines(raw, parse_line, initial_state(MODEL, dialect="databricks"))
    assert folded.is_ok(), folded.error.summary
    assert len(folded.ok) == 2


def test_response_format_tool_name_mirror_matches_constant() -> None:
    from litellm.constants import RESPONSE_FORMAT_TOOL_NAME

    from litellm.translation.providers.databricks.stream import (
        RESPONSE_FORMAT_TOOL_NAME as mirror,
    )

    assert mirror == RESPONSE_FORMAT_TOOL_NAME
