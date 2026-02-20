"""
Regression tests for #20975 — Responses API streaming omits required SSE
event types (``response.output_item.added`` and ``response.content_part.added``).

The OpenAI Responses API SSE contract requires the following sequence
before any ``response.output_text.delta`` events:

    response.created
    response.in_progress
    response.output_item.added
    response.content_part.added
    response.output_text.delta  (repeated)
    ...
    response.completed

Without the fix, ``response.content_part.added`` was never emitted and
the first delta could arrive before ``response.output_item.added`` in
the sync path.
"""

import os
import sys
from typing import Any, List, Optional

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import pytest

from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)
from litellm.types.llms.openai import (
    ResponsesAPIStreamEvents,
)
from litellm.types.utils import (
    Delta,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_text_chunk(
    text: str,
    chunk_id: str = "chatcmpl-test",
    model: str = "gpt-4",
    finish_reason: Optional[str] = None,
) -> ModelResponseStream:
    """Build a minimal ``ModelResponseStream`` that carries a text delta."""
    delta = Delta(content=text)
    choice = StreamingChoices(
        index=0,
        delta=delta,
        finish_reason=finish_reason,
    )
    return ModelResponseStream(
        id=chunk_id,
        model=model,
        choices=[choice],
    )


def _make_finish_chunk(
    chunk_id: str = "chatcmpl-test",
    model: str = "gpt-4",
) -> ModelResponseStream:
    """Build a finishing chunk with ``finish_reason='stop'`` and usage."""
    delta = Delta(content=None)
    choice = StreamingChoices(
        index=0,
        delta=delta,
        finish_reason="stop",
    )
    chunk = ModelResponseStream(
        id=chunk_id,
        model=model,
        choices=[choice],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    return chunk


class FakeSyncStreamWrapper:
    """Fake sync stream that yields pre-built chunks then raises
    ``StopIteration``."""

    def __init__(self, chunks: List[ModelResponseStream]):
        self._chunks = list(chunks)
        self._idx = 0
        self.logging_obj = None

    def __iter__(self):
        return self

    def __next__(self) -> ModelResponseStream:
        if self._idx >= len(self._chunks):
            raise StopIteration
        chunk = self._chunks[self._idx]
        self._idx += 1
        return chunk


class FakeAsyncStreamWrapper:
    """Fake async stream that yields pre-built chunks then raises
    ``StopAsyncIteration``."""

    def __init__(self, chunks: List[ModelResponseStream]):
        self._chunks = list(chunks)
        self._idx = 0
        self.logging_obj = None

    def __aiter__(self):
        return self

    async def __anext__(self) -> ModelResponseStream:
        if self._idx >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._idx]
        self._idx += 1
        return chunk


def _build_iterator(
    chunks: List[ModelResponseStream],
    sync: bool = True,
) -> LiteLLMCompletionStreamingIterator:
    wrapper = (
        FakeSyncStreamWrapper(chunks) if sync else FakeAsyncStreamWrapper(chunks)
    )
    it = LiteLLMCompletionStreamingIterator(
        model="gpt-4",
        litellm_custom_stream_wrapper=wrapper,  # type: ignore[arg-type]
        request_input="Hello",
        responses_api_request={},
        custom_llm_provider="openai",
    )
    return it


def _collect_sync_events(
    it: LiteLLMCompletionStreamingIterator,
) -> List[Any]:
    events: List[Any] = []
    try:
        while True:
            event = next(it)
            events.append(event)
    except StopIteration:
        pass
    return events


async def _collect_async_events(
    it: LiteLLMCompletionStreamingIterator,
) -> List[Any]:
    events: List[Any] = []
    try:
        async for event in it:
            events.append(event)
    except StopAsyncIteration:
        pass
    return events


def _event_types(events: List[Any]) -> List[str]:
    """Extract the ``type`` string from each event."""
    return [getattr(e, "type", None) for e in events]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSSEEventSequence:
    """Verify the full SSE setup event sequence is emitted."""

    def test_sync_emits_content_part_added(self):
        """Sync iteration must emit ``response.content_part.added`` before
        any text delta."""
        chunks = [
            _make_text_chunk("Hello"),
            _make_text_chunk(" world"),
            _make_finish_chunk(),
        ]
        it = _build_iterator(chunks, sync=True)
        events = _collect_sync_events(it)
        types = _event_types(events)

        assert ResponsesAPIStreamEvents.CONTENT_PART_ADDED in types

    def test_sync_correct_event_order(self):
        """The first four events must follow the OpenAI Responses API SSE
        contract."""
        chunks = [
            _make_text_chunk("Hi"),
            _make_finish_chunk(),
        ]
        it = _build_iterator(chunks, sync=True)
        events = _collect_sync_events(it)
        types = _event_types(events)

        # First 4 events must be the setup sequence
        assert types[0] == ResponsesAPIStreamEvents.RESPONSE_CREATED
        assert types[1] == ResponsesAPIStreamEvents.RESPONSE_IN_PROGRESS
        assert types[2] == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED
        assert types[3] == ResponsesAPIStreamEvents.CONTENT_PART_ADDED

    def test_sync_delta_after_setup_events(self):
        """Text deltas must only appear after all four setup events."""
        chunks = [
            _make_text_chunk("Test"),
            _make_finish_chunk(),
        ]
        it = _build_iterator(chunks, sync=True)
        events = _collect_sync_events(it)
        types = _event_types(events)

        delta_idx = types.index(ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA)
        assert delta_idx >= 4, (
            f"output_text.delta at index {delta_idx}, expected >= 4"
        )

    def test_sync_first_chunk_not_lost(self):
        """The text content from the first chunk must appear as a delta
        event (regression: sync path previously lost the first chunk when
        returning queued output_item.added)."""
        chunks = [
            _make_text_chunk("first"),
            _make_text_chunk("second"),
            _make_finish_chunk(),
        ]
        it = _build_iterator(chunks, sync=True)
        events = _collect_sync_events(it)

        deltas = [
            e.delta
            for e in events
            if getattr(e, "type", None)
            == ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA
        ]
        assert "first" in deltas, f"First chunk content lost; deltas={deltas}"
        assert "second" in deltas

    @pytest.mark.asyncio
    async def test_async_emits_content_part_added(self):
        """Async iteration must emit ``response.content_part.added``."""
        chunks = [
            _make_text_chunk("Hello"),
            _make_finish_chunk(),
        ]
        it = _build_iterator(chunks, sync=False)
        events = await _collect_async_events(it)
        types = _event_types(events)

        assert ResponsesAPIStreamEvents.CONTENT_PART_ADDED in types

    @pytest.mark.asyncio
    async def test_async_correct_event_order(self):
        """Async path must also emit the full setup event sequence."""
        chunks = [
            _make_text_chunk("Hi"),
            _make_finish_chunk(),
        ]
        it = _build_iterator(chunks, sync=False)
        events = await _collect_async_events(it)
        types = _event_types(events)

        assert types[0] == ResponsesAPIStreamEvents.RESPONSE_CREATED
        assert types[1] == ResponsesAPIStreamEvents.RESPONSE_IN_PROGRESS
        assert types[2] == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED
        assert types[3] == ResponsesAPIStreamEvents.CONTENT_PART_ADDED

    def test_async_emits_content_part_added_sync_compat(self):
        """Same as async test but using asyncio.run for environments
        without pytest-asyncio."""
        import asyncio

        chunks = [
            _make_text_chunk("Hello"),
            _make_finish_chunk(),
        ]
        it = _build_iterator(chunks, sync=False)
        events = asyncio.get_event_loop().run_until_complete(
            _collect_async_events(it)
        )
        types = _event_types(events)

        assert ResponsesAPIStreamEvents.CONTENT_PART_ADDED in types

    def test_async_correct_event_order_sync_compat(self):
        """Same as async test but using asyncio.run for environments
        without pytest-asyncio."""
        import asyncio

        chunks = [
            _make_text_chunk("Hi"),
            _make_finish_chunk(),
        ]
        it = _build_iterator(chunks, sync=False)
        events = asyncio.get_event_loop().run_until_complete(
            _collect_async_events(it)
        )
        types = _event_types(events)

        assert types[0] == ResponsesAPIStreamEvents.RESPONSE_CREATED
        assert types[1] == ResponsesAPIStreamEvents.RESPONSE_IN_PROGRESS
        assert types[2] == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED
        assert types[3] == ResponsesAPIStreamEvents.CONTENT_PART_ADDED

    def test_sync_completed_event_emitted(self):
        """``response.completed`` must be the last event."""
        chunks = [
            _make_text_chunk("done"),
            _make_finish_chunk(),
        ]
        it = _build_iterator(chunks, sync=True)
        events = _collect_sync_events(it)
        types = _event_types(events)

        assert types[-1] == ResponsesAPIStreamEvents.RESPONSE_COMPLETED

    def test_content_part_added_has_correct_fields(self):
        """The ``content_part.added`` event must carry the expected
        ``output_index``, ``content_index``, and ``part`` fields."""
        chunks = [
            _make_text_chunk("x"),
            _make_finish_chunk(),
        ]
        it = _build_iterator(chunks, sync=True)
        events = _collect_sync_events(it)
        cpa = [
            e
            for e in events
            if getattr(e, "type", None)
            == ResponsesAPIStreamEvents.CONTENT_PART_ADDED
        ]
        assert len(cpa) == 1
        event = cpa[0]
        assert event.output_index == 0
        assert event.content_index == 0
        # part should be an object with type="output_text"
        assert getattr(event.part, "type", None) == "output_text"
