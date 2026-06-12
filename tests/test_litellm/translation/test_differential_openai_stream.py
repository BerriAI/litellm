"""Differential parity for openai streaming, pinned at the SDK-chunk seam.

v1 side: recorded chunk dicts validated into the REAL SDK
``ChatCompletionChunk`` models and replayed through ``CustomStreamWrapper``
(custom_llm_provider="openai") — the decode path production runs; SSE
framing is the OpenAI SDK's plumbing, exactly like AWS framing was
botocore's. v2 side: ``engine.stream.fold_events`` with the openai_compat
chunk parser and the ``openai`` chunk dialect. Chunk lists must be
byte-identical for content/tool/finish chunks.

The trailing usage chunk is the one pinned envelope difference: v1's wrapper
consumes the wire ``choices: []`` usage chunk into a SYNTHESIZED final chunk
(``stream_chunk_builder`` over its accumulated state, wrapper-cached model
string); the v2 fold passes the wire chunk through verbatim and the future
streaming seam owns that synthesis. The usage test pins both sides of that
contract: byte-identical prefix, equal usage numbers on the tail.
"""

import copy
import json
import time

import pytest
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

from litellm.translation.engine.stream import fold_events, fold_lines
from litellm.translation.inbound.openai_chat.stream import initial_state
from litellm.translation.providers.openai_compat.stream import parse_event, parse_line
from litellm.translation_seam import to_model_response_stream

MODEL = "gpt-4o"


def _chunk(delta=None, finish=None, usage=None, choices=None):
    payload = {
        "id": "chatcmpl-S1",
        "object": "chat.completion.chunk",
        "created": 1718000000,
        "model": "gpt-4o-2024-08-06",
        "system_fingerprint": "fp_stream",
        "service_tier": None,
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


def _delta(**overrides):
    return {
        "content": None,
        "function_call": None,
        "refusal": None,
        "role": None,
        "tool_calls": None,
        **overrides,
    }


STREAMS = {
    "text": [
        _chunk(_delta(role="assistant", content="")),
        _chunk(_delta(content="Paris is")),
        _chunk(_delta(content=" the capital.")),
        _chunk(_delta(), finish="stop"),
    ],
    "text_no_leading_role": [
        # first content-bearing chunk still gains role: assistant
        _chunk(_delta(content="Hi")),
        _chunk(_delta(content=" there")),
        _chunk(_delta(), finish="stop"),
    ],
    "tools": [
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
            )
        ),
        _chunk(
            _delta(
                tool_calls=[
                    {
                        "index": 0,
                        "id": None,
                        "type": None,
                        "function": {"name": None, "arguments": '{"ci'},
                    }
                ]
            )
        ),
        _chunk(
            _delta(
                tool_calls=[
                    {
                        "index": 0,
                        "id": None,
                        "type": None,
                        "function": {"name": None, "arguments": 'ty": "Paris"}'},
                    }
                ]
            )
        ),
        _chunk(_delta(), finish="tool_calls"),
    ],
    "empty_keepalive_swallowed": [
        _chunk(_delta(role="assistant", content="")),
        _chunk(_delta()),  # empty delta mid-stream: v1 drops it
        _chunk(_delta(content="ok")),
        _chunk(_delta(), finish="stop"),
    ],
}

_USAGE = {
    "completion_tokens": 7,
    "prompt_tokens": 11,
    "total_tokens": 18,
    "completion_tokens_details": {
        "accepted_prediction_tokens": 0,
        "audio_tokens": 0,
        "reasoning_tokens": 0,
        "rejected_prediction_tokens": 0,
    },
    "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0},
}

USAGE_STREAM = STREAMS["text"] + [_chunk(choices=[], usage=_USAGE)]


def _v1_chunks(events: list, stream_options=None) -> list:
    logging = Logging(
        model=MODEL,
        messages=[{"role": "user", "content": "stream"}],
        stream=True,
        call_type="completion",
        start_time=time.time(),
        litellm_call_id="diff-openai-stream",
        function_id="diff-openai-stream",
    )
    sdk_chunks = (
        ChatCompletionChunk.model_validate(event) for event in copy.deepcopy(events)
    )
    wrapper = CustomStreamWrapper(
        completion_stream=sdk_chunks,
        model=MODEL,
        custom_llm_provider="openai",
        logging_obj=logging,
        stream_options=stream_options,
    )
    return [chunk.model_dump() for chunk in wrapper]


def _v2_chunks(events: list) -> list:
    folded = fold_events(
        copy.deepcopy(events), parse_event, initial_state(model=MODEL, dialect="openai")
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
    assert _norm(_v2_chunks(events)) == _norm(_v1_chunks(events))


def test_v2_stream_decodes_sse_lines_identically(frozen_ambient) -> None:
    """fold_lines over the raw SSE framing (data: ... / [DONE]) produces the
    same chunks as fold_events over the parsed payloads."""
    events = STREAMS["text"]
    lines = [f"data: {json.dumps(event)}" for event in events] + ["", "data: [DONE]"]
    folded = fold_lines(lines, parse_line, initial_state(model=MODEL, dialect="openai"))
    assert folded.is_ok(), folded.error.summary
    via_lines = [
        to_model_response_stream(chunk, "chatcmpl-AMBIENT").model_dump()
        for chunk in folded.ok
    ]
    assert _norm(via_lines) == _norm(_v2_chunks(events))


def test_usage_chunk_passthrough_pins_the_seam_contract(frozen_ambient) -> None:
    v1 = _v1_chunks(USAGE_STREAM, stream_options={"include_usage": True})
    v2 = _v2_chunks(USAGE_STREAM)
    # content + finish chunks are byte-identical
    assert _norm(v2[:-1]) == _norm(v1[: len(v2) - 1])
    # v1's tail is the wrapper-synthesized usage chunk (envelope); v2's tail
    # is the wire usage chunk verbatim. The usage numbers must agree.
    assert len(v1) == len(v2)
    v1_tail, v2_tail = v1[-1], v2[-1]
    assert v2_tail["choices"] == []
    assert v1_tail["usage"] is not None and v2_tail["usage"] is not None
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        assert v1_tail["usage"][key] == v2_tail["usage"][key] == _USAGE[key]
    assert (
        v2_tail["usage"]["prompt_tokens_details"]["cached_tokens"]
        == _USAGE["prompt_tokens_details"]["cached_tokens"]
    )
    assert (
        v2_tail["usage"]["completion_tokens_details"]["reasoning_tokens"]
        == _USAGE["completion_tokens_details"]["reasoning_tokens"]
    )


_UNSUPPORTED_CHUNKS = {
    "function_call_delta": (
        _chunk(_delta(function_call={"name": "f", "arguments": ""})),
        "function_call",
    ),
    "unknown_finish_reason": (
        _chunk(_delta(), finish="function_call"),
        "finish_reason",
    ),
    "multiple_choices": (
        _chunk(
            choices=[
                {"index": 0, "delta": _delta(content="a"), "finish_reason": None},
                {"index": 1, "delta": _delta(content="b"), "finish_reason": None},
            ]
        ),
        "multiple stream choices",
    ),
    "unknown_delta_key": (
        _chunk({**_delta(content="x"), "reasoning_content": "hmm"}),
        "stream delta keys",
    ),
}


@pytest.mark.parametrize("name", sorted(_UNSUPPORTED_CHUNKS))
def test_unreachable_chunk_shape_is_a_typed_error(name: str) -> None:
    event, reason_fragment = _UNSUPPORTED_CHUNKS[name]
    result = parse_event(event)
    assert result.is_error(), f"{name} unexpectedly parsed"
    assert reason_fragment in result.error.summary, result.error.summary
