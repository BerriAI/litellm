"""Differential parity for xai streaming, pinned at the SSE data-line seam.

v1 side: raw ``data:`` lines through ``XAIChatCompletionStreamingHandler``
(the chunk_parser owns the xai BEHAVIOR: dummy-choice injection into
``choices: []`` usage tails, per-chunk usage fold/normalize, the base
handler's reasoning rename) into ``CustomStreamWrapper("xai")`` — NOT the
parsed-chunk seam the openai gate uses, because those rewrites sit below
the line seam (researcher-3 R3). v2 side: ``fold_lines`` with the xai
parser and the ``xai`` chunk dialect. Two-sided over the generated corpus:
v1 at HEAD must equal the committed snapshot AND v2 must equal it
byte-for-byte for content/reasoning/tool/finish chunks.

The usage tail is the one pinned envelope difference (the openai-port seam
contract, inherited unchanged): v1's wrapper swallows the dummy-choice tail
mid-stream and SYNTHESIZES a final usage chunk at StopIteration (only under
``include_usage``); the v2 fold passes the wire ``choices: []`` chunk
through with the FOLDED usage, and the future streaming seam owns the
synthesis. The contract row pins byte-identical prefixes and equal usage
numbers (reasoning folded into completion on both sides).
"""

import copy
import json

import pytest

from litellm.translation.engine.stream import fold_events, fold_lines
from litellm.translation.inbound.openai_chat.stream import initial_state
from litellm.translation.providers.xai.stream import parse_event, parse_line
from litellm.translation_seam import to_model_response_stream

from ._xai_corpus import (
    SNAPSHOTS_DIR,
    STREAM_MODEL,
    canonical_json,
    corpus,
    load_json,
    replay_xai_sse_lines,
)

STREAMS = corpus("streams")
_TAIL_ROW = "usage_tail_include_usage"
_PREFIX_ROWS = sorted(name for name in STREAMS if name != _TAIL_ROW)


def _v2_chunks(events: list) -> list:
    lines = [f"data: {json.dumps(event)}" for event in copy.deepcopy(events)]
    lines.append("data: [DONE]")
    folded = fold_lines(lines, parse_line, initial_state(STREAM_MODEL, dialect="xai"))
    assert folded.is_ok(), folded.error.summary
    return [
        to_model_response_stream(chunk, "chatcmpl-AMBIENT").model_dump()
        for chunk in folded.ok
    ]


def _norm(chunks: list) -> str:
    return json.dumps(chunks, sort_keys=True, default=str)


@pytest.mark.parametrize("name", sorted(STREAMS))
def test_v1_at_head_still_matches_the_snapshot(name: str, frozen_ambient) -> None:
    row = STREAMS[name]
    snapshot = (SNAPSHOTS_DIR / "streams" / f"{name}.json").read_text()
    assert (
        canonical_json(replay_xai_sse_lines(row["events"], row["stream_options"]))
        == snapshot
    )


@pytest.mark.parametrize("name", _PREFIX_ROWS)
def test_v2_stream_matches_the_snapshot(name: str, frozen_ambient) -> None:
    snapshot = load_json(SNAPSHOTS_DIR / "streams" / f"{name}.json")
    assert _norm(_v2_chunks(STREAMS[name]["events"])) == _norm(snapshot)


def test_usage_tail_pins_the_seam_contract(frozen_ambient) -> None:
    """v1's tail is the wrapper-synthesized usage chunk (envelope); v2's
    tail is the wire ``choices: []`` chunk with the folded usage. Prefix
    byte-identical, usage numbers equal — including the reasoning fold
    (2 + 7 -> 9) and the normalized total."""
    snapshot = load_json(SNAPSHOTS_DIR / "streams" / f"{_TAIL_ROW}.json")
    v2 = _v2_chunks(STREAMS[_TAIL_ROW]["events"])
    assert len(v2) == len(snapshot)
    assert _norm(v2[:-1]) == _norm(snapshot[: len(v2) - 1])
    v1_tail, v2_tail = snapshot[-1], v2[-1]
    assert v2_tail["choices"] == []
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        assert v1_tail["usage"][key] == v2_tail["usage"][key], key
    assert v2_tail["usage"]["completion_tokens"] == 9  # 2 + 7 folded
    assert (
        v1_tail["usage"]["completion_tokens_details"]["reasoning_tokens"]
        == v2_tail["usage"]["completion_tokens_details"]["reasoning_tokens"]
        == 7
    )


def test_usage_tail_swallowed_without_stream_options(frozen_ambient) -> None:
    """Without include_usage v1 swallows the tail entirely (usage goes to
    hidden params); the v2 passthrough keeps the prefix identical and the
    seam owns withholding the tail — pinned so the streaming seam knows."""
    events = STREAMS[_TAIL_ROW]["events"]
    v1 = replay_xai_sse_lines(events, None)
    v2 = _v2_chunks(events)
    assert len(v1) == len(v2) - 1
    assert _norm([c for c in map(dict, v1)]) == _norm(v2[:-1])


def test_v2_line_and_event_folds_agree(frozen_ambient) -> None:
    events = STREAMS["text"]["events"]
    folded = fold_events(
        copy.deepcopy(events), parse_event, initial_state(STREAM_MODEL, dialect="xai")
    )
    assert folded.is_ok(), folded.error.summary
    via_events = [
        to_model_response_stream(chunk, "chatcmpl-AMBIENT").model_dump()
        for chunk in folded.ok
    ]
    assert _norm(via_events) == _norm(_v2_chunks(events))


def _chunk(delta=None, finish=None, usage=None, choices=None):
    payload = {
        "id": "cmpl-u1",
        "object": "chat.completion.chunk",
        "created": 1718000000,
        "model": STREAM_MODEL,
        "choices": [
            {
                "index": 0,
                "delta": delta or {},
                "logprobs": None,
                "finish_reason": finish,
            }
        ],
        "usage": usage,
    }
    if choices is not None:
        payload["choices"] = choices
    return payload


_UNSUPPORTED_CHUNKS = {
    "function_call_delta": (
        _chunk({"function_call": {"name": "f", "arguments": ""}}),
        "function_call",
    ),
    "multiple_choices": (
        _chunk(
            choices=[
                {"index": 0, "delta": {"content": "a"}, "finish_reason": None},
                {"index": 1, "delta": {"content": "b"}, "finish_reason": None},
            ]
        ),
        "multiple stream choices",
    ),
    "unknown_delta_key": (
        _chunk({"content": "x", "thinking_blocks": []}),
        "stream delta keys",
    ),
    "error_payload": (
        {"error": {"message": "boom"}, "choices": []},
        "provider stream error",
    ),
}


@pytest.mark.parametrize("name", sorted(_UNSUPPORTED_CHUNKS))
def test_unreachable_chunk_shape_is_a_typed_error(name: str) -> None:
    event, reason_fragment = _UNSUPPORTED_CHUNKS[name]
    result = parse_event(event)
    assert result.is_error(), f"{name} unexpectedly parsed"
    assert reason_fragment in result.error.summary, result.error.summary


def test_numeric_string_usage_folds_like_v1s_own_arithmetic() -> None:
    """v1's dict-path fold is int(x or 0): numeric strings fold
    (completion 2+7 -> 9, untouched keys keep their original values).
    Pinned against v1's OWN static methods (critic-grok M2)."""
    from litellm.llms.xai.chat.transformation import XAIChatConfig

    from litellm.translation.providers.xai.response import (
        fold_reasoning_tokens,
        normalize_usage_totals,
    )

    usage = {
        "prompt_tokens": "171",
        "completion_tokens": "2",
        "total_tokens": "180",
        "completion_tokens_details": {"reasoning_tokens": "7"},
    }
    v1_usage = copy.deepcopy(usage)
    XAIChatConfig._fold_reasoning_tokens_into_completion(v1_usage)
    XAIChatConfig._normalize_openai_compatible_usage_totals(v1_usage)
    folded = fold_reasoning_tokens(copy.deepcopy(usage))
    assert not isinstance(folded, Exception)
    v2_usage = normalize_usage_totals(folded)
    assert v2_usage == v1_usage
    assert v2_usage["completion_tokens"] == 9


def test_uncoercible_usage_chunk_is_loud_on_both_sides() -> None:
    """v1's chunk_parser raises ValueError out of the stream iterator on an
    uncoercible token value; v2's parse_event must error, never serve the
    chunk with the field read as 0 (critic-grok M2)."""
    from litellm.llms.xai.chat.transformation import XAIChatConfig

    usage = {
        "prompt_tokens": "abc",
        "completion_tokens": 2,
        "total_tokens": 180,
        "completion_tokens_details": {"reasoning_tokens": 7},
    }
    with pytest.raises(ValueError):
        XAIChatConfig._fold_reasoning_tokens_into_completion(copy.deepcopy(usage))
    result = parse_event(_chunk(choices=[], usage=copy.deepcopy(usage)))
    assert result.is_error(), "uncoercible usage must be loud, not a silent 0"
    assert "not int-coercible" in result.error.summary, result.error.summary
