"""IR ``StreamEvent`` -> Anthropic Messages SSE events (a pure fold).

The reverse of the request stream: a provider's stream parser emits IR
``StreamEvent``s (already in Anthropic order, because the stream IR mirrors
Anthropic's SSE shape), and this fold re-encodes them into the Anthropic
Messages event dicts a ``/v1/messages`` client expects:

    message_start
      content_block_start / content_block_delta* / content_block_stop  (xN)
    message_delta (stop_reason + usage)
    message_stop

This does NOT port v1's hold-and-merge state machine
(``AnthropicStreamWrapper``); the provider parsers already order the events,
so the fold only owns the block-index bookkeeping: lazy-open the content block
on the first delta of each block (closing the previous block with
``content_block_stop`` on a block switch), and close the open block before
``message_delta``. The IR delta ``index`` is the Anthropic block index (the
anthropic provider parser carries it through verbatim), so a switch is
detected by an index change; ``tool_use_start`` is the explicit open signal
for a tool block (text/thinking blocks open implicitly on their first delta).

``message_start`` carries the zeroed initial usage v1 emits to signal cache
support; the real token counts ride ``message_delta`` at the end. The fold
emits event dicts; the ``event: <type>\\ndata: <json>\\n\\n`` SSE framing is
identical to v1's and applied above the fold.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Literal

from expression.collections import Block
from typing_extensions import assert_never

from ...errors import BoundaryError, TranslationError
from ...ir import FinishReason, PlainJson, StreamEvent

SseEvent = dict[str, PlainJson]

_BlockType = Literal["text", "thinking", "tool_use"]


@dataclass(frozen=True)
class StreamState:
    message_started: bool = False
    open_index: int | None = None
    open_type: _BlockType | None = None


def initial_state() -> StreamState:
    return StreamState()


_StepResult = tuple[StreamState, tuple[SseEvent, ...]] | TranslationError


def step(state: StreamState, event: StreamEvent) -> _StepResult:
    match event.tag:
        case "start":
            return _start(state, event)
        case "text_delta":
            return _content_delta(
                state,
                event.text_delta.index,
                "text",
                {"type": "text_delta", "text": event.text_delta.text},
            )
        case "tool_use_start":
            return _tool_use_start(state, event)
        case "tool_args_delta":
            return _tool_args_delta(state, event)
        case "thinking_delta":
            return _content_delta(
                state,
                event.thinking_delta.index,
                "thinking",
                {"type": "thinking_delta", "thinking": event.thinking_delta.thinking},
            )
        case "signature_delta":
            return _content_delta(
                state,
                event.signature_delta.index,
                "thinking",
                {
                    "type": "signature_delta",
                    "signature": event.signature_delta.signature,
                },
            )
        case "finish":
            return _finish(state, event)
        case "stop":
            return state, ({"type": "message_stop"},)
        case "wire_chunk" | "chunk":
            return _wrong_event(event.tag)
    assert_never(event.tag)


def _wrong_event(event_tag: str) -> TranslationError:
    return TranslationError.of_boundary(
        BoundaryError.of(
            Block.of_seq(
                [
                    f"anthropic_messages stream fold received a {event_tag} event its"
                    " provider parser can never emit; dialect/parser mismatch"
                ]
            )
        )
    )


def _start(state: StreamState, event: StreamEvent) -> _StepResult:
    start = event.start
    message_start: SseEvent = {
        "type": "message_start",
        "message": {
            "id": start.id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": start.model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        },
    }
    return replace(state, message_started=True), (message_start,)


def _tool_use_start(state: StreamState, event: StreamEvent) -> _StepResult:
    tool = event.tool_use_start
    next_state, prefix = _close_open_block(state)
    opened = replace(next_state, open_index=tool.index, open_type="tool_use")
    start_event: SseEvent = {
        "type": "content_block_start",
        "index": tool.index,
        "content_block": {
            "type": "tool_use",
            "id": tool.id,
            "name": tool.name,
            "input": {},
        },
    }
    return opened, (*prefix, start_event)


def _tool_args_delta(state: StreamState, event: StreamEvent) -> _StepResult:
    delta = event.tool_args_delta
    if state.open_index != delta.index or state.open_type != "tool_use":
        # The anthropic IR always emits tool_use_start before its args; an
        # args delta for an unopened tool block is a parser/ordering bug and
        # must be loud (we cannot synthesize the missing id/name).
        return _wrong_event("tool_args_delta without a matching tool_use_start")
    delta_event: SseEvent = {
        "type": "content_block_delta",
        "index": delta.index,
        "delta": {"type": "input_json_delta", "partial_json": delta.partial_json},
    }
    return state, (delta_event,)


def _content_delta(
    state: StreamState, index: int, block_type: _BlockType, delta: SseEvent
) -> _StepResult:
    next_state, prefix = _ensure_open_block(state, index, block_type)
    delta_event: SseEvent = {
        "type": "content_block_delta",
        "index": index,
        "delta": delta,
    }
    return next_state, (*prefix, delta_event)


def _ensure_open_block(
    state: StreamState, index: int, block_type: _BlockType
) -> tuple[StreamState, tuple[SseEvent, ...]]:
    if state.open_index == index and state.open_type == block_type:
        return state, ()
    closed_state, close_events = _close_open_block(state)
    opened = replace(closed_state, open_index=index, open_type=block_type)
    return opened, (*close_events, _content_block_start(index, block_type))


def _content_block_start(index: int, block_type: _BlockType) -> SseEvent:
    block: PlainJson = (
        {"type": "thinking", "thinking": "", "signature": ""}
        if block_type == "thinking"
        else {"type": "text", "text": ""}
    )
    return {"type": "content_block_start", "index": index, "content_block": block}


def _close_open_block(
    state: StreamState,
) -> tuple[StreamState, tuple[SseEvent, ...]]:
    if state.open_index is None:
        return state, ()
    stop_event: SseEvent = {"type": "content_block_stop", "index": state.open_index}
    return replace(state, open_index=None, open_type=None), (stop_event,)


def _finish(state: StreamState, event: StreamEvent) -> _StepResult:
    closed_state, close_events = _close_open_block(state)
    finish = event.finish
    message_delta: SseEvent = {
        "type": "message_delta",
        "delta": {"stop_reason": _STOP_REASON[finish.finish]},
        "usage": {"input_tokens": 0, "output_tokens": finish.output_tokens},
    }
    return closed_state, (*close_events, message_delta)


_STOP_REASON: Mapping[FinishReason, str] = MappingProxyType(
    {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
    }
)
