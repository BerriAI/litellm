"""Differential parity for openrouter streaming, pinned at the SSE line seam.

v1 side: raw ``data:`` lines through ``OpenRouterChatCompletionStreamingHandler``
(its OWN handler: key-presence error-chunk RAISE, strict id/created/model/
choices envelope via KeyError -> OpenRouterException 400, the UNCONDITIONAL
``delta["reasoning_content"] = delta.get("reasoning")`` assignment — the
original ``reasoning`` key kept, a native ``reasoning_content`` CLOBBERED to
None, and the ``choice["delta"]`` subscript) into
``CustomStreamWrapper("openrouter")``. v2 side: ``fold_lines`` with the
openrouter parser (the "unconditional" ReasoningMode arm added with this
consumer + the missing-delta pre-step) and the shared ``xai`` chunk dialect.

openrouter is NOT part of the base-handler PINNED DIVERGENCE: its v1 handler
RAISES on error chunks, and the policy row mirrors that raise (the cometapi
shape).
"""

import copy
import json

import pytest

from litellm.exceptions import BadRequestError, MidStreamFallbackError
from litellm.llms.openrouter.chat.transformation import (
    OpenRouterChatCompletionStreamingHandler,
)

from litellm.translation.engine.stream import fold_events, fold_lines
from litellm.translation.inbound.openai_chat.stream import initial_state
from litellm.translation.providers.openrouter import parse_event, parse_line
from litellm.translation_seam import to_model_response_stream

from ._own_module_corpus import provider_config, replay_v1_sse_lines

MODEL = "openai/gpt-4o"
WIRE_MODEL = "openai/gpt-4o"


def _chunk(delta, finish=None, usage=None, choices=None):
    payload = {
        "id": "or-chunk-1",
        "object": "chat.completion.chunk",
        "created": 1718000000,
        "model": WIRE_MODEL,
        "choices": [
            {"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}
        ],
        "usage": usage,
    }
    if choices is not None:
        payload["choices"] = choices
    return payload


STREAMS = {
    "text": [
        _chunk({"role": "assistant", "content": ""}),
        _chunk({"content": "Hello"}),
        _chunk({"content": " world"}),
        _chunk({}, finish="stop"),
    ],
    "tools": [
        _chunk({"role": "assistant", "content": None}),
        _chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": ""},
                    }
                ]
            }
        ),
        _chunk(
            {
                "tool_calls": [
                    {"index": 0, "function": {"arguments": '{"city":"Paris"}'}}
                ]
            }
        ),
        _chunk({}, finish="tool_calls"),
    ],
    # the unconditional assignment: delta.reasoning -> reasoning_content with
    # the original key KEPT (v1 assigns without popping)
    "reasoning_both_keys": [
        _chunk({"role": "assistant", "content": ""}),
        _chunk({"reasoning": "thinking..."}),
        _chunk({"content": "answer"}),
        _chunk({}, finish="stop"),
    ],
    # a NATIVE reasoning_content delta is CLOBBERED to None by v1's
    # assignment -> the emptied delta chunk is swallowed by the wrapper on
    # both sides
    "native_reasoning_content_clobbered": [
        _chunk({"role": "assistant", "content": ""}),
        _chunk({"reasoning_content": "native"}),
        _chunk({"content": "answer"}),
        _chunk({}, finish="stop"),
    ],
    "empty_keepalive_swallowed": [
        _chunk({"role": "assistant", "content": ""}),
        _chunk({}),
        _chunk({"content": "hi"}),
        _chunk({}, finish="stop"),
    ],
}

USAGE_STREAM = [
    _chunk({"role": "assistant", "content": ""}),
    _chunk({"content": "Hi"}),
    _chunk({}, finish="stop"),
    _chunk(
        {},
        choices=[],
        usage={
            "prompt_tokens": 7,
            "completion_tokens": 2,
            "total_tokens": 9,
            "cost": 0.0002,
        },
    ),
]


def _v1_chunks(events, stream_options=None):
    return replay_v1_sse_lines("openrouter", events, MODEL, stream_options)


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


def test_reasoning_delta_keeps_both_keys(frozen_ambient) -> None:
    v2 = _v2_chunks(STREAMS["reasoning_both_keys"])
    delta = v2[1]["choices"][0]["delta"]
    assert delta["reasoning"] == "thinking..."
    assert delta["reasoning_content"] == "thinking..."


def test_native_reasoning_content_chunk_is_swallowed(frozen_ambient) -> None:
    """The clobber pin: v1 turns a native reasoning_content delta into an
    empty delta (reasoning_content = .get('reasoning') = None) and the
    wrapper swallows it — both sides emit one fewer chunk and NO 'native'
    text anywhere."""
    events = STREAMS["native_reasoning_content_clobbered"]
    v1 = _v1_chunks(events)
    v2 = _v2_chunks(events)
    assert len(v1) == len(v2) == len(events) - 1
    assert "native" not in _norm(v1)
    assert "native" not in _norm(v2)


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
    events = STREAMS["reasoning_both_keys"]
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
    "error_chunk": (
        {"error": {"message": "boom", "code": 500}},
        "provider stream error",
    ),
    "missing_id": (
        {
            "object": "chat.completion.chunk",
            "created": 1718000000,
            "model": WIRE_MODEL,
            "choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": None}],
        },
        "KeyError",
    ),
    "missing_created": (
        {
            "id": "or-chunk-1",
            "object": "chat.completion.chunk",
            "model": WIRE_MODEL,
            "choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": None}],
        },
        "KeyError",
    ),
    "missing_model": (
        {
            "id": "or-chunk-1",
            "object": "chat.completion.chunk",
            "created": 1718000000,
            "choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": None}],
        },
        "KeyError",
    ),
    "missing_choices": (
        {
            "id": "or-chunk-1",
            "object": "chat.completion.chunk",
            "created": 1718000000,
            "model": WIRE_MODEL,
        },
        "KeyError",
    ),
    "missing_delta": (
        _chunk({}, choices=[{"index": 0, "finish_reason": None}]),
        "missing 'delta'",
    ),
    "multiple_choices": (
        _chunk(
            {},
            choices=[
                {"index": 0, "delta": {"content": "a"}, "finish_reason": None},
                {"index": 1, "delta": {"content": "b"}, "finish_reason": None},
            ],
        ),
        "multiple stream choices",
    ),
}


@pytest.mark.parametrize("name", sorted(_LOUD_CHUNKS))
def test_loud_chunk_shapes_error_on_both_sides(name: str, frozen_ambient) -> None:
    """v1 RAISES out of the iterator on error chunks (MidStreamFallbackError
    at the wrapper) and on missing envelope keys INCLUDING choices[].delta
    (KeyError -> OpenRouterException 400 -> BadRequestError); v2 is a loud
    typed error value, never a served chunk. multiple_choices is the one
    v1-SERVES shape here (its loop passes both through) — v2's typed
    'unreachable for v2-sent requests' error is the family convention, and
    n>1 cannot leave the inbound boundary."""
    event, reason_fragment = _LOUD_CHUNKS[name]
    result = parse_event(copy.deepcopy(event))
    assert result.is_error(), f"{name} unexpectedly parsed"
    assert reason_fragment in result.error.summary, result.error.summary
    if name == "error_chunk":
        with pytest.raises(MidStreamFallbackError):
            _v1_chunks([event])
    elif name.startswith("missing_"):
        with pytest.raises(BadRequestError):
            _v1_chunks([event])
    else:
        assert name == "multiple_choices"
        served = _v1_chunks([event, _chunk({}, finish="stop")])
        assert len(served) >= 1  # v1 serves the multi-choice chunk


def test_iterator_is_the_openrouter_handler() -> None:
    """openrouter's stream truth is its OWN handler (NOT the base) — if this
    canary fails, re-derive the policy row against the new handler."""
    handler = provider_config("openrouter", MODEL).get_model_response_iterator(
        streaming_response=iter(()), sync_stream=True
    )
    assert type(handler) is OpenRouterChatCompletionStreamingHandler
