"""Differential parity for mistral streaming (wave-2b-beta), pinned at the
SSE line seam.

v1 side: ``data:``/[DONE] lines through ``MistralChatResponseIterator``
(the magistral content-list pre-step, then the BASE
``OpenAIChatCompletionStreamingHandler`` rebuild) into
``CustomStreamWrapper("mistral")``. v2 side: ``fold_lines`` with the
mistral parser — the same pre-step composed over the shared httpx_chunk
factory (``reasoning="rename"`` + the NEW ``passthrough_delta_keys`` axis
admitting the pre-step's ``thinking_blocks``; mistral is the axis's
consumer, per the no-consumer-no-arm rule) — and the shared ``xai`` chunk
dialect.
"""

import copy
import json
import time

import pytest

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.mistral.chat.transformation import MistralChatResponseIterator

from litellm.translation.engine.stream import fold_events, fold_lines
from litellm.translation.inbound.openai_chat.stream import initial_state
from litellm.translation.providers.mistral import parse_event, parse_line
from litellm.translation_seam import to_model_response_stream

MODEL = "magistral-medium-latest"
WIRE_MODEL = "magistral-medium-2506"


def _chunk(
    delta: dict | None = None,
    finish: str | None = None,
    usage: dict | None = None,
    **extra: object,
) -> dict:
    payload = {
        "id": "chunk-1",
        "object": "chat.completion.chunk",
        "created": 1718000000,
        "model": WIRE_MODEL,
        "choices": [
            {
                "index": 0,
                "delta": delta if delta is not None else {},
                "finish_reason": finish,
            }
        ],
        **extra,
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


_USAGE = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}

STREAMS = {
    "text": [
        _chunk({"role": "assistant", "content": "He"}),
        _chunk({"content": "llo"}),
        _chunk({}, finish="stop"),
    ],
    "magistral_thinking_blocks": [
        _chunk(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": [{"type": "text", "text": "think1"}],
                    }
                ],
            }
        ),
        _chunk(
            {
                "content": [
                    {
                        "type": "thinking",
                        "thinking": [{"type": "text", "text": "think2"}],
                    },
                    {"type": "text", "text": "ans"},
                ]
            }
        ),
        _chunk({"content": "tail"}),
        _chunk({}, finish="stop"),
    ],
    "reasoning_rename": [
        _chunk({"role": "assistant", "reasoning": "r1"}),
        _chunk({"content": "x"}),
        _chunk({}, finish="stop"),
    ],
    "empty_thinking_role_only_chunk": [
        # the normalize pops empty thinking_blocks: v1 still emits the
        # role-only chunk (content None) — pinned byte-for-byte
        _chunk(
            {"role": "assistant", "content": [{"type": "thinking", "thinking": []}]}
        ),
        _chunk({"content": "x"}),
        _chunk({}, finish="stop"),
    ],
    "tools": [
        _chunk(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "f", "arguments": ""},
                    }
                ],
            }
        ),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}),
        _chunk({}, finish="tool_calls"),
    ],
}

USAGE_STREAM = [
    _chunk({"role": "assistant", "content": "Hi"}),
    _chunk({}, finish="stop"),
    {
        "id": "chunk-1",
        "object": "chat.completion.chunk",
        "created": 1718000000,
        "model": WIRE_MODEL,
        "choices": [],
        "usage": _USAGE,
    },
]


def _v1_chunks(events: list, stream_options: dict | None = None) -> list:
    lines = [f"data: {json.dumps(event)}" for event in copy.deepcopy(events)]
    lines.append("data: [DONE]")
    handler = MistralChatResponseIterator(
        streaming_response=iter(lines), sync_stream=True
    )
    logging = Logging(
        model=MODEL,
        messages=[{"role": "user", "content": "stream"}],
        stream=True,
        call_type="completion",
        start_time=time.time(),
        litellm_call_id="diff-mistral-stream",
        function_id="diff-mistral-stream",
    )
    wrapper = CustomStreamWrapper(
        completion_stream=handler,
        model=MODEL,
        custom_llm_provider="mistral",
        logging_obj=logging,
        stream_options=stream_options,
    )
    return [chunk.model_dump() for chunk in wrapper]


def _v2_chunks(events: list) -> list:
    lines = [f"data: {json.dumps(event)}" for event in copy.deepcopy(events)]
    lines.append("data: [DONE]")
    folded = fold_lines(lines, parse_line, initial_state(MODEL, dialect="xai"))
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
    assert _norm(_v2_chunks(events)) == _norm(_v1_chunks(events))


def test_thinking_blocks_ride_the_delta(frozen_ambient) -> None:
    """Semantic pin on top of the byte rows: the emitted delta carries the
    normalized thinking_blocks (signature "mistral") AND reasoning_content,
    and the request model (not the wire model) names every chunk."""
    v2 = _v2_chunks(STREAMS["magistral_thinking_blocks"])
    delta = v2[0]["choices"][0]["delta"]
    assert delta["reasoning_content"] == "think1"
    assert delta["thinking_blocks"] == [
        {"type": "thinking", "thinking": "think1", "signature": "mistral"}
    ]
    assert all(chunk["model"] == MODEL for chunk in v2)


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
    events = STREAMS["magistral_thinking_blocks"]
    folded = fold_events(
        copy.deepcopy(events), parse_event, initial_state(MODEL, dialect="xai")
    )
    assert folded.is_ok(), folded.error.summary
    via_events = [
        to_model_response_stream(chunk, "chatcmpl-AMBIENT").model_dump()
        for chunk in folded.ok
    ]
    assert _norm(via_events) == _norm(_v2_chunks(events))


_LOUD_CHUNKS = {
    "non_dict_content_block": (
        _chunk({"role": "assistant", "content": ["bare"]}),
        "not an object",
    ),
    "non_list_thinking": (
        _chunk(
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "x"}]}
        ),
        "not a list",
    ),
    "unknown_delta_key": (
        _chunk({"content": "x", "made_up_key": 1}),
        "stream delta keys",
    ),
    # verifier-wave2b-beta F3: the old arm coerced the non-str thinking text
    # to "" and SERVED a stripped chunk (with the mixed shape it served
    # reasoning_content "ok"); v1's pre-step except-arm replays the
    # still-list content and the wrapper raises MidStreamFallbackError.
    "non_string_thinking_text": (
        _chunk(
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": [{"type": "text", "text": 5}]}
                ],
            }
        ),
        "thinking text is not a string",
    ),
    "non_string_thinking_text_beside_ok_text": (
        _chunk(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": [
                            {"type": "text", "text": 5},
                            {"type": "text", "text": "ok"},
                        ],
                    }
                ],
            }
        ),
        "thinking text is not a string",
    ),
}


@pytest.mark.parametrize("name", sorted(_LOUD_CHUNKS))
def test_loud_chunk_shapes(name: str, frozen_ambient) -> None:
    """The malformed content-list shapes raise in v1 too (its pre-step's
    except-arm replays the ORIGINAL chunk through the base rebuild, whose
    Delta validation rejects list content); unknown delta keys are the
    family's standard unreachable-shape typed error."""
    event, fragment = _LOUD_CHUNKS[name]
    result = parse_event(copy.deepcopy(event))
    assert result.is_error(), f"{name} unexpectedly parsed"
    assert fragment in result.error.summary, result.error.summary
    if name != "unknown_delta_key":
        with pytest.raises(Exception):
            _v1_chunks([event])


_ERROR_CHUNK_STREAM = [
    _chunk({"role": "assistant", "content": "He"}),
    {
        "id": "chunk-1",
        "object": "chat.completion.chunk",
        "created": 1718000000,
        "model": MODEL,
        "error": {"message": "upstream exploded", "code": 502},
    },
    _chunk({"content": "llo"}),
    _chunk({}, finish="stop"),
]


def test_error_chunk_divergence_two_sided(frozen_ambient) -> None:
    """Sibling-merge consistency sweep: MistralChatResponseIterator SUBCLASSES
    the base OpenAIChatCompletionStreamingHandler, so mistral INHERITS the
    family's silent error-chunk swallow (probed two-sided at the merge: v1
    serves 3 chunks with no error surface; v2's parse_line is a LOUD typed
    boundary error naming the chunk). Mistral therefore joins the report's
    ONE failure-counted PINNED DIVERGENCE row via
    _wave2b_beta_error_chunk_pins — the wave-2b-alpha merge-notes obligation
    ("any beta provider riding the BASE handler with the silent swallow
    joins the same row; never a second PINNED row"). Flag-on cannot lose
    data (v2 is louder, not quieter). If the v1 half fails, the iterator
    learned to raise: re-decide the divergence and update the report row in
    the same commit."""
    v1 = _v1_chunks(_ERROR_CHUNK_STREAM)
    assert "error" not in json.dumps(v1, default=str), v1
    assert [c["choices"][0]["delta"].get("content") for c in v1[:2]] == ["He", "llo"]
    lines = [f"data: {json.dumps(e)}" for e in copy.deepcopy(_ERROR_CHUNK_STREAM)]
    folded = fold_lines(lines, parse_line, initial_state(MODEL, dialect="xai"))
    assert folded.is_error()
    assert "provider stream error" in folded.error.summary, folded.error.summary
    assert "upstream exploded" in folded.error.summary
