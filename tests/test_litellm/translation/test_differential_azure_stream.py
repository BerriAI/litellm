"""Differential parity for azure streaming, pinned at the SDK-chunk seam.

v1 side: recorded chunk dicts validated into the REAL SDK
``ChatCompletionChunk`` models and replayed through ``CustomStreamWrapper``
(custom_llm_provider="azure"). v2 side: ``fold_events`` with the azure chunk
parser and the ``azure`` chunk dialect. The azure deltas over the openai
stream gate: the wrapper re-reads ``model`` off every chunk
(streaming_handler.py:1448-1454), choice-level ``content_filter_results``
survives on content chunks (the StreamingChoices rebuild) but not on the
finish flush, and the leading empty-choices ``prompt_filter_results`` chunk
is swallowed whole on both sides.

Second gate: the vendored characterization stream fixtures replayed against
their committed snapshots, v1-at-HEAD (drift guard) and v2 both byte-for-byte.
"""

import copy
import json

import pytest

from litellm.translation.engine.stream import fold_events, fold_lines
from litellm.translation.inbound.openai_chat.stream import initial_state
from litellm.translation.providers.azure.stream import parse_event, parse_line
from litellm.translation_seam import to_model_response_stream

from . import _azure_corpus as corpus

MODEL = corpus.MODEL  # the wrapper's initial model: the deployment name

_FILTERS = {
    "hate": {"filtered": False, "severity": "safe"},
    "self_harm": {"filtered": False, "severity": "safe"},
    "sexual": {"filtered": False, "severity": "safe"},
    "violence": {"filtered": False, "severity": "safe"},
}


def _chunk(delta=None, finish=None, choices=None, filters=None, model=None):
    payload = {
        "id": "chatcmpl-AZS1",
        "object": "chat.completion.chunk",
        "created": 1718000000,
        "model": model if model is not None else "gpt-4.1-2025-04-14",
        "system_fingerprint": "fp_azs",
        "choices": [
            {
                "index": 0,
                "delta": delta or {},
                "logprobs": None,
                "finish_reason": finish,
                "content_filter_results": filters if filters is not None else _FILTERS,
            }
        ],
    }
    if choices is not None:
        payload["choices"] = choices
    return payload


def _delta(**overrides):
    return {
        "content": None,
        "function_call": None,
        "refusal": None,
        "role": None,
        "tool_calls": None,
        **overrides,
    }


_PROMPT_FILTER_CHUNK = {
    "id": "",
    "object": "chat.completion.chunk",
    "created": 0,
    "model": "",
    "choices": [],
    "prompt_filter_results": [{"prompt_index": 0, "content_filter_results": _FILTERS}],
}

STREAMS = {
    "text_with_filter_results": [
        _PROMPT_FILTER_CHUNK,
        _chunk(_delta(role="assistant", content=""), filters={}),
        _chunk(_delta(content="Paris is")),
        _chunk(_delta(content=" the capital.")),
        _chunk(_delta(), finish="stop", filters={}),
    ],
    "model_reread_from_chunks": [
        # the wrapper stamps each emitted chunk with the latest wire model
        _chunk(_delta(role="assistant", content="a"), model="gpt-4.1-2025-04-14"),
        _chunk(_delta(content="b"), model="gpt-4.1-mini-2025-04-14"),
        _chunk(_delta(), finish="stop"),
    ],
    "tools_with_filter_results": [
        _chunk(
            _delta(
                role="assistant",
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": ""},
                    }
                ],
            ),
            filters={},
        ),
        _chunk(
            _delta(
                tool_calls=[
                    {
                        "index": 0,
                        "id": None,
                        "type": None,
                        "function": {"name": None, "arguments": '{"city": "Paris"}'},
                    }
                ]
            ),
            filters={},
        ),
        _chunk(_delta(), finish="tool_calls", filters={}),
    ],
}


def _v2_chunks(events: list) -> list:
    folded = fold_events(
        copy.deepcopy(events), parse_event, initial_state(model=MODEL, dialect="azure")
    )
    assert folded.is_ok(), folded.error.summary
    return [
        to_model_response_stream(chunk, "chatcmpl-AMBIENT").model_dump()
        for chunk in folded.ok
    ]


def _norm(chunks: list) -> str:
    return json.dumps(chunks, sort_keys=True, default=str)


@pytest.mark.parametrize("name", sorted(STREAMS))
def test_v2_stream_matches_v1(name: str, frozen_ambient) -> None:
    events = STREAMS[name]
    assert _norm(_v2_chunks(events)) == _norm(corpus.replay_azure_sdk_chunks(events))


def test_azure_ai_wrapper_branch_matches_too(frozen_ambient) -> None:
    """CustomStreamWrapper's model re-read branch covers azure_ai as well;
    the azure_ai provider re-exports the azure parser and dialect."""
    events = STREAMS["model_reread_from_chunks"]
    assert _norm(_v2_chunks(events)) == _norm(
        corpus.replay_azure_sdk_chunks(events, custom_llm_provider="azure_ai")
    )


def test_v2_stream_decodes_sse_lines_identically(frozen_ambient) -> None:
    events = STREAMS["text_with_filter_results"]
    lines = [f"data: {json.dumps(event)}" for event in events] + ["", "data: [DONE]"]
    folded = fold_lines(lines, parse_line, initial_state(model=MODEL, dialect="azure"))
    assert folded.is_ok(), folded.error.summary
    via_lines = [
        to_model_response_stream(chunk, "chatcmpl-AMBIENT").model_dump()
        for chunk in folded.ok
    ]
    assert _norm(via_lines) == _norm(_v2_chunks(events))


# ---------------------------------------------------------------------------
# Second gate: vendored characterization stream fixtures.
# ---------------------------------------------------------------------------

_FIXTURES = sorted(
    path.stem for path in (corpus.FIXTURES_DIR / "streams" / "azure").glob("*.json")
)


@pytest.mark.parametrize("fixture_id", _FIXTURES)
def test_v1_still_matches_stream_snapshot(fixture_id: str, frozen_ambient) -> None:
    events = corpus.load_json(
        corpus.FIXTURES_DIR / "streams" / "azure" / f"{fixture_id}.json"
    )
    expected = corpus.stream_snapshot_chunks(
        corpus.SNAPSHOTS_DIR / "streams" / "azure" / f"{fixture_id}.json"
    )
    assert corpus.canonical_json(corpus.replay_azure_sdk_chunks(events)) == (
        corpus.canonical_json(expected)
    ), (
        f"v1 drifted from the characterization snapshot for {fixture_id}; "
        "regenerate the corpus and ship the diff as its own PR"
    )


@pytest.mark.parametrize("fixture_id", _FIXTURES)
def test_v2_matches_stream_snapshot(fixture_id: str, frozen_ambient) -> None:
    events = corpus.load_json(
        corpus.FIXTURES_DIR / "streams" / "azure" / f"{fixture_id}.json"
    )
    expected = corpus.stream_snapshot_chunks(
        corpus.SNAPSHOTS_DIR / "streams" / "azure" / f"{fixture_id}.json"
    )
    assert corpus.canonical_json(_v2_chunks(events)) == corpus.canonical_json(expected)
