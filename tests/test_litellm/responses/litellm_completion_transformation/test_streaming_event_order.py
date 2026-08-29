"""
Regression tests for the chat-completions -> Responses API streaming bridge
event ordering in
litellm/responses/litellm_completion_transformation/streaming_iterator.py.

DeepSeek-style reasoning streams (role-only first chunk, then
reasoning_content deltas, then content deltas) previously produced event
sequences that violate the OpenAI Responses streaming contract:

1. ``response.output_item.added`` for the message item was emitted *before*
   ``response.output_item.done`` for the reasoning item (overlapping item
   lifecycles). Clients that track a single active item (e.g. codex) drop
   all subsequent text deltas as a result.
2. Tool-call-only streams (reasoning + function_call, no assistant text)
   emitted a trailing message ``output_text.done`` / ``content_part.done`` /
   ``output_item.done`` trio for a message item that was never announced via
   ``output_item.added`` ("ghost" message item). The content part also
   carried the reasoning text under a non-standard ``reasoning_text`` type.
3. ``ReasoningSummaryTextDeltaEvent`` omitted ``summary_index`` when
   serialized with ``exclude_unset`` semantics (field equals its default 0),
   which strict consumers reject.
"""

from typing import Final
from unittest.mock import MagicMock

import pytest

from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)
from litellm.types.llms.openai import ResponsesAPIStreamEvents
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

CHAT_COMPLETION_ID = "chatcmpl-77d33d09-effa-4cd2-9c0d-c742d4358256"


def _chunk(
    content: str | None = None,
    reasoning_content: str | None = None,
    tool_calls: list | None = None,
    annotations: list | None = None,
    finish_reason: str | None = None,
) -> ModelResponseStream:
    return ModelResponseStream(
        id=CHAT_COMPLETION_ID,
        created=1748575031,
        model="deepseek-v4-flash",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(
                    role="assistant",
                    content=content,
                    reasoning_content=reasoning_content,
                    tool_calls=tool_calls,
                    annotations=annotations,
                ),
                finish_reason=finish_reason,
            )
        ],
    )


class _FakeStreamWrapper:
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.logging_obj = MagicMock()

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


async def _collect_events(chunks) -> list:
    iterator = LiteLLMCompletionStreamingIterator(
        model="deepseek-v4-flash",
        litellm_custom_stream_wrapper=_FakeStreamWrapper(chunks),
        request_input="hello",
        responses_api_request={},
        custom_llm_provider="openai",
        litellm_metadata={},
    )
    events = []
    async for event in iterator:
        events.append(event)
    return events


def _event_type(event) -> str:
    return str(getattr(event, "type", ""))


def _item_type(event) -> str:
    item = getattr(event, "item", None)
    return str(getattr(item, "type", "")) if item is not None else ""


def _output_item_events(events) -> list:
    return [
        event
        for event in events
        if _event_type(event)
        in (
            str(ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED),
            str(ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE),
        )
    ]


def _reasoning_then_text_chunks() -> list:
    return [
        _chunk(),  # role-only first chunk (DeepSeek style)
        _chunk(reasoning_content="Let me think."),
        _chunk(reasoning_content=" Step two."),
        _chunk(content="Hello"),
        _chunk(content=" world"),
        _chunk(finish_reason="stop"),
    ]


@pytest.mark.asyncio
async def test_reasoning_then_text_emits_native_event_order():
    """
    Reasoning item must be fully closed (output_item.done) before the message
    item is announced (output_item.added), matching the OpenAI Responses
    native sequence: no overlapping item lifecycles.
    """
    events = await _collect_events(_reasoning_then_text_chunks())

    item_events = _output_item_events(events)
    seq = [f"{_event_type(event)}[{_item_type(event)}]" for event in item_events]

    assert seq == [
        f"{ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED}[reasoning]",
        f"{ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE}[reasoning]",
        f"{ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED}[message]",
        f"{ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE}[message]",
    ], f"unexpected item lifecycle sequence: {seq}"

    # text deltas must come after the message item announcement
    types = [_event_type(event) for event in events]
    message_added_idx = types.index(str(ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED))
    assert types.index(str(ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA)) > message_added_idx


@pytest.mark.asyncio
async def test_reasoning_delta_carries_summary_index():
    """
    Reasoning summary text deltas must carry summary_index even under
    exclude_unset serialization semantics (the field equals its default 0).
    """
    events = await _collect_events(_reasoning_then_text_chunks())

    deltas = [
        event for event in events if _event_type(event) == str(ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DELTA)
    ]
    assert len(deltas) == 2
    for event in deltas:
        assert event.summary_index == 0
        assert "summary_index" in event.model_dump(exclude_unset=True)


@pytest.mark.asyncio
async def test_content_part_done_is_output_text_when_reasoning_present():
    """
    The message content part must always be output_text. Previously the
    bridge stuffed the full reasoning text into the message's content part
    under a non-standard reasoning_text type whenever the final response
    carried reasoning_content.
    """
    events = await _collect_events(_reasoning_then_text_chunks())

    part_done = [event for event in events if _event_type(event) == str(ResponsesAPIStreamEvents.CONTENT_PART_DONE)]
    assert len(part_done) == 1
    assert part_done[0].part.type == "output_text"
    assert part_done[0].part.text == "Hello world"


@pytest.mark.asyncio
async def test_tool_call_only_stream_skips_ghost_message_done():
    """
    A reasoning + tool-call stream with no assistant text must not emit any
    message item events at stream end: the message item was never announced
    via output_item.added, so emitting done events for it produces a "ghost"
    item that spec-compliant consumers reject.
    """
    chunks = [
        _chunk(),  # role-only first chunk
        _chunk(reasoning_content="Thinking about the tool call."),
        _chunk(
            tool_calls=[
                {
                    "id": "call_deferred_message",
                    "type": "function",
                    "function": {"name": "do_thing", "arguments": '{"x":1}'},
                    "index": 0,
                }
            ]
        ),
        _chunk(finish_reason="tool_calls"),
    ]
    events = await _collect_events(chunks)

    # reasoning lifecycle must be complete
    item_seq = [f"{_event_type(event)}[{_item_type(event)}]" for event in _output_item_events(events)]
    assert f"{ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED}[reasoning]" in item_seq, (
        f"missing reasoning added: {item_seq}"
    )
    assert f"{ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE}[reasoning]" in item_seq, f"missing reasoning done: {item_seq}"

    # no ghost message item: never added, so it must never be done
    assert f"{ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED}[message]" not in item_seq, (
        f"unexpected message added: {item_seq}"
    )
    assert f"{ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE}[message]" not in item_seq, (
        f"ghost message done emitted: {item_seq}"
    )

    # no trailing message text/part done events either
    types = [_event_type(event) for event in events]
    assert str(ResponsesAPIStreamEvents.OUTPUT_TEXT_DONE) not in types
    assert str(ResponsesAPIStreamEvents.CONTENT_PART_DONE) not in types

    # stream still terminates with response.completed
    assert str(ResponsesAPIStreamEvents.RESPONSE_COMPLETED) in types


@pytest.mark.asyncio
async def test_reasoning_after_tool_call_still_gets_item_declaration():
    """
    When the one-shot added flag is consumed by a tool-call chunk before any
    reasoning_content arrives, the reasoning item declaration must still be
    emitted (reasoning deltas cannot reference an undeclared item).
    """
    chunks = [
        _chunk(
            tool_calls=[
                {
                    "id": "call_late_reasoning",
                    "type": "function",
                    "function": {"name": "do_thing", "arguments": "{}"},
                    "index": 0,
                }
            ]
        ),
        _chunk(reasoning_content="Thinking after the tool call."),
        _chunk(content="Result"),
        _chunk(finish_reason="stop"),
    ]
    events = await _collect_events(chunks)

    item_seq = [f"{_event_type(event)}[{_item_type(event)}]" for event in _output_item_events(events)]
    assert f"{ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED}[reasoning]" in item_seq, (
        f"reasoning item was never declared: {item_seq}"
    )

    # reasoning lifecycle must close before the message item is announced
    assert item_seq.index(f"{ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE}[reasoning]") < item_seq.index(
        f"{ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED}[message]"
    ), f"reasoning done overlaps message added: {item_seq}"

    # reasoning deltas must reference the declared reasoning item
    reasoning_added = [
        event
        for event in _output_item_events(events)
        if f"{_event_type(event)}[{_item_type(event)}]" == f"{ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED}[reasoning]"
    ][0]
    deltas = [
        event for event in events if _event_type(event) == str(ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DELTA)
    ]
    assert len(deltas) == 1
    assert deltas[0].item_id == reasoning_added.item.id


@pytest.mark.asyncio
async def test_annotation_events_wait_for_message_item_announcement():
    """
    Annotation events reference the message item; they must not be emitted
    before the message item's output_item.added, even when the annotation
    chunk arrives before the first content chunk.
    """
    annotation = {
        "type": "url_citation",
        "url_citation": {
            "start_index": 0,
            "end_index": 6,
            "url": "https://example.com",
            "title": "Example",
        },
    }
    chunks = [
        _chunk(annotations=[annotation]),  # annotation before any content
        _chunk(content="Answer with a citation"),
        _chunk(finish_reason="stop"),
    ]
    events = await _collect_events(chunks)

    types = [_event_type(event) for event in events]
    message_added_idx = next(i for i, t in enumerate(types) if t == str(ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED))
    annotation_idx = next(
        i for i, t in enumerate(types) if t == str(ResponsesAPIStreamEvents.OUTPUT_TEXT_ANNOTATION_ADDED)
    )
    assert annotation_idx > message_added_idx, (
        f"annotation.added must not precede the message output_item.added: types={types}"
    )

    # the annotation must reference the announced message item
    annotation_events = [
        event for event in events if _event_type(event) == str(ResponsesAPIStreamEvents.OUTPUT_TEXT_ANNOTATION_ADDED)
    ]
    assert len(annotation_events) == 1
    message_added = events[message_added_idx]
    assert annotation_events[0].item_id == message_added.item.id


URL_CITATION_ANNOTATION: Final = {
    "type": "url_citation",
    "url_citation": {
        "start_index": 0,
        "end_index": 6,
        "url": "https://example.com/citation",
        "title": "Example",
    },
}


@pytest.mark.asyncio
async def test_annotation_only_stream_still_delivers_the_annotation():
    """
    Regression: an annotation-only chunk carries no content, so deferring the
    message item past it strands the annotation -- annotation events are
    buffered until their item is announced, and the stream would reach
    response.completed with the buffer never drained.
    """
    events = await _collect_events(
        [
            _chunk(annotations=[URL_CITATION_ANNOTATION]),
            _chunk(finish_reason="stop"),
        ]
    )
    types = [_event_type(event) for event in events]
    annotation_added = str(ResponsesAPIStreamEvents.OUTPUT_TEXT_ANNOTATION_ADDED)
    assert annotation_added in types, f"annotation-only stream dropped the annotation: types={types}"
    assert types.index(annotation_added) > types.index(str(ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED)), (
        f"annotation.added must follow the message output_item.added: types={types}"
    )


@pytest.mark.asyncio
async def test_annotation_during_reasoning_is_drained_at_announcement():
    """
    An annotation arriving while the message item is still deferred (the
    stream is inside the reasoning item) is buffered and flushed when the
    message item is finally announced, so it still lands after that item's
    output_item.added instead of being stranded in the buffer.
    """
    events = await _collect_events(
        [
            _chunk(reasoning_content="Thinking", annotations=[URL_CITATION_ANNOTATION]),
            _chunk(content="Answer"),
            _chunk(finish_reason="stop"),
        ]
    )
    types = [_event_type(event) for event in events]
    annotation_added = str(ResponsesAPIStreamEvents.OUTPUT_TEXT_ANNOTATION_ADDED)
    assert annotation_added in types, f"buffered annotation was dropped: types={types}"

    message_added_idx = next(
        i
        for i, event in enumerate(events)
        if _event_type(event) == str(ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED) and _item_type(event) == "message"
    )
    assert types.index(annotation_added) > message_added_idx, (
        f"drained annotation must follow the message output_item.added: types={types}"
    )


@pytest.mark.asyncio
async def test_annotation_on_the_final_chunk_is_not_dropped():
    """
    The per-chunk emit path only runs while chunks are still arriving, so an
    annotation queued from the final chunk has to be drained during
    finalization or it never reaches the consumer.
    """
    events = await _collect_events(
        [
            _chunk(content="Answer with a citation"),
            _chunk(annotations=[URL_CITATION_ANNOTATION], finish_reason="stop"),
        ]
    )
    types = [_event_type(event) for event in events]
    annotation_added = str(ResponsesAPIStreamEvents.OUTPUT_TEXT_ANNOTATION_ADDED)
    assert annotation_added in types, f"annotation from the final chunk was dropped: types={types}"
    assert types.index(annotation_added) < types.index(str(ResponsesAPIStreamEvents.OUTPUT_TEXT_DONE)), (
        f"annotation.added must precede output_text.done: types={types}"
    )
