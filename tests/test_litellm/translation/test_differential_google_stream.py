"""Differential parity for google streaming.

Gemini routes pin at the parsed-event seam: the recorded ``alt=sse`` lines
replay through the REAL ``ModelResponseIterator`` inside
``CustomStreamWrapper`` on the v1 side, while v2 folds the decoded
``GenerateContentResponse`` events (SSE framing is transport plumbing). The
fold reproduces v1's stateful bits: cumulative tool index across chunks, the
``has_seen_tool_calls`` stop->tool_calls rewrite, the wrapper-synthesized
trailing finish chunk, withheld usage, and thought signatures riding inside
tool-call ids. Vertex claude streams are anthropic SSE through the anthropic
parser (the bedrock_invoke precedent), id-normalized like the other
anthropic-family gates; gemini chunk ids are the wire ``responseId`` and
compare verbatim.
"""

import copy
import json

import pytest

from litellm.translation.engine.stream import fold_events, fold_lines
from litellm.translation.inbound.openai_chat import parse_request
from litellm.translation.inbound.openai_chat.stream import initial_state
from litellm.translation.providers.anthropic.stream import (
    parse_sse_line,
    reverse_names,
)
from litellm.translation.providers.google_genai.stream import parse_event
from litellm.translation_seam import to_model_response_stream
from litellm.translation_seam_google import to_model_response_stream_google

from . import _google_corpus as corpus

_GEMINI_PROVIDERS = ("gemini", "vertex_gemini")


def _fixture_ids(provider_key: str) -> list:
    return sorted(
        path.stem
        for path in (corpus.FIXTURES_DIR / "streams" / provider_key).glob("*.txt")
    )


def _read_lines(provider_key: str, fixture_id: str) -> list:
    path = corpus.FIXTURES_DIR / "streams" / provider_key / f"{fixture_id}.txt"
    return path.read_text().splitlines()


def _norm(chunks: list, normalize_id: bool) -> str:
    if normalize_id:
        chunks = [{**chunk, "id": "chatcmpl-X"} for chunk in chunks]
    return json.dumps(chunks, sort_keys=True, default=str)


def _v2_gemini_chunks(provider_key: str, lines: list) -> list:
    model, _, _ = corpus.resolve(provider_key)
    events = corpus.sse_events(lines)
    folded = fold_events(
        events, parse_event, initial_state(model=model, dialect="gemini")
    )
    assert folded.is_ok(), folded.error.summary
    return [to_model_response_stream_google(body).model_dump() for body in folded.ok]


def _v2_vertex_anthropic_chunks(lines: list) -> list:
    model, _, _ = corpus.resolve("vertex_anthropic")
    parsed = parse_request(
        {
            "model": model,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "stream"}],
        }
    )
    assert parsed.is_ok(), parsed.error.summary
    reverse = reverse_names(parsed.ok)
    folded = fold_lines(
        lines,
        lambda line: parse_sse_line(line, reverse),
        initial_state(model=model, dialect="anthropic"),
    )
    assert folded.is_ok(), folded.error.summary
    return [
        to_model_response_stream(chunk, "chatcmpl-X").model_dump()
        for chunk in folded.ok
    ]


@pytest.mark.parametrize(
    "provider_key,fixture_id",
    [(p, f) for p in _GEMINI_PROVIDERS for f in _fixture_ids(p)],
)
def test_v2_gemini_stream_matches_v1_and_snapshot(
    provider_key: str, fixture_id: str, frozen_ambient
) -> None:
    lines = _read_lines(provider_key, fixture_id)
    v1 = corpus.replay_v1_gemini_sse(provider_key, copy.deepcopy(lines))
    v2 = _v2_gemini_chunks(provider_key, lines)
    assert _norm(v2, False) == _norm(v1, False)
    snapshot = corpus.load_json(
        corpus.SNAPSHOTS_DIR / "streams" / provider_key / f"{fixture_id}.json"
    )
    assert _norm(v2, False) == _norm(snapshot, False), (
        f"v2/v1 drifted from the characterization snapshot for {fixture_id}; "
        "regenerate the corpus and ship the diff as its own PR"
    )


@pytest.mark.parametrize("fixture_id", _fixture_ids("vertex_anthropic"))
def test_v2_vertex_anthropic_stream_matches_v1_and_snapshot(
    fixture_id: str, frozen_ambient
) -> None:
    lines = _read_lines("vertex_anthropic", fixture_id)
    v1 = corpus.replay_v1_vertex_anthropic_sse(copy.deepcopy(lines))
    v2 = _v2_vertex_anthropic_chunks(lines)
    assert _norm(v2, True) == _norm(v1, True)
    snapshot = corpus.load_json(
        corpus.SNAPSHOTS_DIR / "streams" / "vertex_anthropic" / f"{fixture_id}.json"
    )
    assert _norm(v2, True) == _norm(snapshot, True), (
        f"v2/v1 drifted from the characterization snapshot for {fixture_id}; "
        "regenerate the corpus and ship the diff as its own PR"
    )


def test_mid_stream_error_object_is_loud() -> None:
    folded = fold_events(
        [{"error": {"code": 429, "message": "RESOURCE_EXHAUSTED"}}],
        parse_event,
        initial_state(model="gemini-2.5-pro", dialect="gemini"),
    )
    assert folded.is_error()


def test_finish_only_chunk_rewrites_stop_to_tool_calls() -> None:
    events = [
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {"functionCall": {"name": "get_weather", "args": {}}}
                        ],
                    }
                }
            ],
            "responseId": "r1",
        },
        {"candidates": [{"finishReason": "STOP"}], "responseId": "r1"},
    ]
    folded = fold_events(
        events, parse_event, initial_state(model="gemini-2.5-pro", dialect="gemini")
    )
    assert folded.is_ok(), folded.error.summary
    chunks = list(folded.ok)
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
