"""Differential parity for the /v1/messages reverse (stream) path.

A recorded Anthropic SSE event stream is parsed into IR ``StreamEvent``s by the
anthropic provider parser. v2 folds those events into Anthropic SSE event dicts
directly; the oracle folds the SAME IR events into chat chunks (the openai_chat
anthropic-dialect fold), adapts them onto ``ModelResponseStream``, and runs
v1's ``AnthropicStreamWrapper`` over them. The emitted event dicts must match.

Two faithfulness adjustments to the oracle bridge, neither masking a v2 bug:
- the ``message_start`` id is ambient (v1 mints a fresh ``msg_{uuid4}``,
  ignoring the wire id), so it is normalized on both sides;
- the openai_chat anthropic-dialect fold does not propagate stream usage onto
  the finish chunk (streaming usage is a seam-scope follow-up in v1), so the
  final usage v1's wrapper merges into ``message_delta`` is reconstructed onto
  the finish ``ModelResponseStream`` exactly as v1's real anthropic flow
  attaches it (``prompt_tokens = input + cache_creation + cache_read``,
  ``prompt_tokens_details.cached_tokens = cache_read``,
  ``_cache_creation/_cache_read_input_tokens``). The values come from the wire
  ``message_start``/``message_delta`` usage, the same source v2 folds from.

The text+tool_use split (issue-18238) is pinned: the content_block_stop ->
content_block_start ordering between the text block and the tool block must be
byte-right. A mutation in the fold (wrong block index, a missing
content_block_stop, a mis-mapped stop_reason, wrong usage) diverges.
"""

import json

import pytest

from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
)
from litellm.types.utils import PromptTokensDetailsWrapper, Usage

from litellm.translation.inbound.anthropic_messages import stream as ams
from litellm.translation.inbound.openai_chat import stream as oai
from litellm.translation.providers.anthropic.stream import parse_event
from litellm.translation_seam import to_model_response_stream

MODEL = "claude-sonnet-4-5"


def _message_start(usage: dict) -> dict:
    return {
        "type": "message_start",
        "message": {
            "id": "msg_stream",
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": usage,
        },
    }


_TEXT = {
    "events": [
        _message_start({"input_tokens": 12, "output_tokens": 1}),
        {"type": "ping"},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Paris"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": " it is"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 9},
        },
        {"type": "message_stop"},
    ],
    "input": (12, 0, 0),
    "output": 9,
}

_TEXT_THEN_TOOL = {
    "events": [
        _message_start({"input_tokens": 50, "output_tokens": 1}),
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "calling"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "get",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"q":'},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": "1}"},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 15},
        },
        {"type": "message_stop"},
    ],
    "input": (50, 0, 0),
    "output": 15,
}

_THINKING_THEN_TEXT = {
    "events": [
        _message_start({"input_tokens": 30, "output_tokens": 1}),
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "step"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "sg"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": "done"},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 8},
        },
        {"type": "message_stop"},
    ],
    "input": (30, 0, 0),
    "output": 8,
}

_CACHED = {
    "events": [
        _message_start(
            {
                "input_tokens": 100,
                "output_tokens": 1,
                "cache_creation_input_tokens": 5,
                "cache_read_input_tokens": 10,
            }
        ),
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hi"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 7},
        },
        {"type": "message_stop"},
    ],
    "input": (100, 5, 10),
    "output": 7,
}

_CASES = {
    "text": _TEXT,
    "text_then_tool_use": _TEXT_THEN_TOOL,
    "cached": _CACHED,
}


def _ir_events(events: list) -> list:
    parsed = []
    for event in events:
        result = parse_event(event, {})
        assert result.is_ok(), result.error.summary
        if result.ok is not None:
            parsed.append(result.ok)
    return parsed


def _v2_events(ir_events: list) -> list:
    state = ams.initial_state()
    out: list = []
    for event in ir_events:
        stepped = ams.step(state, event)
        assert not hasattr(stepped, "summary"), getattr(stepped, "summary", "")
        state, emitted = stepped
        out.extend(emitted)
    return out


def _final_usage(input_split: tuple, output: int) -> Usage:
    uncached_plus_creation, creation, read = input_split
    prompt_tokens = uncached_plus_creation + creation + read
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=output,
        total_tokens=prompt_tokens + output,
    )
    usage.prompt_tokens_details = PromptTokensDetailsWrapper(cached_tokens=read)
    setattr(usage, "_cache_creation_input_tokens", creation)
    setattr(usage, "_cache_read_input_tokens", read)
    return usage


def _v1_events(ir_events: list, input_split: tuple, output: int) -> list:
    state = oai.initial_state(model=MODEL, dialect="anthropic")
    chat_chunks: list = []
    for event in ir_events:
        state, emitted = oai.step(state, event)
        chat_chunks.extend(emitted)
    final_usage = _final_usage(input_split, output)
    streams = []
    for body in chat_chunks:
        chunk = to_model_response_stream(body, "chatcmpl-stream")
        choices = body.get("choices")
        first = choices[0] if isinstance(choices, list) and choices else None
        if isinstance(first, dict) and first.get("finish_reason") is not None:
            setattr(chunk, "usage", final_usage)
        streams.append(chunk)
    wrapper = AnthropicStreamWrapper(completion_stream=iter(streams), model=MODEL)
    return [event for event in wrapper if isinstance(event, dict)]


def _normalize(event: dict) -> dict:
    event = json.loads(json.dumps(event, default=str))
    if event.get("type") == "message_start":
        event["message"]["id"] = "<id>"
    return event


@pytest.mark.parametrize("name", sorted(_CASES))
def test_stream_fold_matches_v1_anthropic_stream_wrapper(name: str) -> None:
    case = _CASES[name]
    ir_events = _ir_events(case["events"])
    v2 = [_normalize(event) for event in _v2_events(ir_events)]
    v1 = [
        _normalize(event)
        for event in _v1_events(ir_events, case["input"], case["output"])
    ]
    assert v2 == v1, name


def test_thinking_first_preserves_wire_block_index_unlike_v1_bridge() -> None:
    """DELIBERATE divergence (researcher-6 §1.3: the IR is Anthropic-shaped, so
    the fold must NOT reproduce v1's index-shifting gymnastics). v1's
    experimental chat-bridge wrapper re-derives blocks from chat-chunk content
    and ALWAYS opens an empty text block at index 0, shifting a thinking-first
    stream to thinking@1. The v2 fold preserves the wire's thinking@0 (the
    correct Anthropic ordering a real backend sent), so it emits no spurious
    leading text block and the thinking block keeps index 0."""
    ir_events = _ir_events(_THINKING_THEN_TEXT["events"])
    v2 = _v2_events(ir_events)
    starts = [
        (event["index"], event["content_block"]["type"])
        for event in v2
        if event["type"] == "content_block_start"
    ]
    assert starts == [(0, "thinking"), (1, "text")]

    v1 = _v1_events(
        ir_events, _THINKING_THEN_TEXT["input"], _THINKING_THEN_TEXT["output"]
    )
    v1_starts = [
        (event["index"], event["content_block"]["type"])
        for event in v1
        if event["type"] == "content_block_start"
    ]
    # v1's bridge forces text@0 then shifts thinking to index 1: the divergence
    # this test pins as deliberate (the fold is wire-faithful, v1 is not).
    assert v1_starts == [(0, "text"), (1, "thinking"), (2, "text")]
