"""
Tests for AnthropicResponsesStreamWrapper
(litellm/llms/anthropic/experimental_pass_through/responses_adapters/streaming_iterator.py)
"""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../.."))
)

from litellm.llms.anthropic.experimental_pass_through.responses_adapters.streaming_iterator import (
    AnthropicResponsesStreamWrapper,
)


def _process_all(events: list) -> list:
    wrapper = AnthropicResponsesStreamWrapper(responses_stream=None, model="m")
    for event in events:
        wrapper._process_event(event)
    return list(wrapper._chunk_queue)


class _FakeResponsesStream:
    def __init__(self, events: list) -> None:
        self._it = iter(events)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


async def _drain(events: list) -> list:
    wrapper = AnthropicResponsesStreamWrapper(
        responses_stream=_FakeResponsesStream(events), model="m"
    )
    return [chunk async for chunk in wrapper]


class TestProcessEventTextDeltaWithoutOutputItemAdded:
    """Streams that skip response.output_item.added (e.g. LMStudio) must still
    open a text block before any delta and never emit index -1."""

    def test_process_event_synthesizes_content_block_start_before_delta(self):
        chunks = _process_all(
            [
                {"type": "response.output_text.delta", "item_id": "i1", "delta": "Hel"},
                {"type": "response.output_text.delta", "item_id": "i1", "delta": "lo"},
            ]
        )
        assert [c["type"] for c in chunks] == [
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
        ]
        assert chunks[0]["content_block"] == {"type": "text", "text": ""}
        assert [c["index"] for c in chunks] == [0, 0, 0]
        assert chunks[1]["delta"] == {"type": "text_delta", "text": "Hel"}

    def test_process_event_delta_without_item_id_never_yields_negative_index(self):
        chunks = _process_all([{"type": "response.output_text.delta", "delta": "Hi"}])
        assert [(c["type"], c["index"]) for c in chunks] == [
            ("content_block_start", 0),
            ("content_block_delta", 0),
        ]

    def test_process_event_unregistered_item_id_opens_new_text_block(self):
        chunks = _process_all(
            [
                {
                    "type": "response.output_item.added",
                    "item": {"type": "reasoning", "id": "rs_1"},
                },
                {"type": "response.output_text.delta", "item_id": "m1", "delta": "Hi"},
            ]
        )
        assert [c["type"] for c in chunks] == [
            "content_block_start",
            "content_block_stop",
            "content_block_start",
            "content_block_delta",
        ]
        assert chunks[0]["content_block"] == {"type": "thinking", "thinking": ""}
        assert chunks[2]["content_block"] == {"type": "text", "text": ""}
        assert [c["index"] for c in chunks] == [0, 0, 1, 1]

    def test_process_event_message_item_opens_text_block_exactly_once(self):
        chunks = _process_all(
            [
                {
                    "type": "response.output_item.added",
                    "item": {"type": "message", "id": "m1"},
                },
                {"type": "response.output_text.delta", "item_id": "m1", "delta": "Hi"},
            ]
        )
        assert [(c["type"], c["index"]) for c in chunks] == [
            ("content_block_start", 0),
            ("content_block_delta", 0),
        ]


def _reasoning_first_bridge_events() -> list:
    """The event sequence a chat-completions -> Responses API bridge emits for
    an OpenAI-compatible reasoning backend: a message output item is announced
    first (from the role-only chunk), reasoning arrives as
    reasoning_summary_text.delta events with unrelated item_ids, then the text
    answer arrives on the message item."""
    return [
        {"type": "response.created"},
        {"type": "response.in_progress"},
        {"type": "response.output_item.added", "item": {"type": "message", "id": "msg_1"}},
        {"type": "response.content_part.added", "item_id": "msg_1"},
        {"type": "response.reasoning_summary_text.delta", "item_id": "rs_a", "delta": "I"},
        {"type": "response.reasoning_summary_text.delta", "item_id": "rs_b", "delta": " am"},
        {"type": "response.reasoning_summary_text.delta", "item_id": "rs_c", "delta": " thinking"},
        {"type": "response.output_text.delta", "item_id": "msg_1", "delta": "OK"},
        {"type": "response.output_text.done", "item_id": "msg_1"},
        {"type": "response.content_part.done", "item_id": "msg_1"},
        {"type": "response.output_item.done", "item": {"type": "message", "id": "msg_1"}},
        {"type": "response.completed"},
    ]


class TestReasoningContentIsNotStreamedIntoTextBlock:
    """Regression tests for https://github.com/BerriAI/litellm/issues/32357"""

    @pytest.mark.asyncio
    async def test_message_start_emitted_exactly_once(self):
        chunks = await _drain(_reasoning_first_bridge_events())
        assert [c["type"] for c in chunks].count("message_start") == 1

    @pytest.mark.asyncio
    async def test_reasoning_streams_into_its_own_thinking_block(self):
        chunks = await _drain(_reasoning_first_bridge_events())

        assert [c["type"] for c in chunks] == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]

        thinking_block = chunks[1]
        assert thinking_block["content_block"] == {"type": "thinking", "thinking": ""}
        assert thinking_block["index"] == 0
        assert [c["delta"]["type"] for c in chunks[2:5]] == [
            "thinking_delta",
            "thinking_delta",
            "thinking_delta",
        ]
        assert all(c["index"] == 0 for c in chunks[2:5])

        text_block = chunks[6]
        assert text_block["content_block"] == {"type": "text", "text": ""}
        assert text_block["index"] == 1
        assert chunks[7]["delta"] == {"type": "text_delta", "text": "OK"}
        assert chunks[7]["index"] == 1

    @pytest.mark.asyncio
    async def test_no_thinking_delta_is_ever_emitted_into_a_text_block(self):
        chunks = await _drain(_reasoning_first_bridge_events())

        text_block_indexes = {
            c["index"]
            for c in chunks
            if c["type"] == "content_block_start" and c["content_block"]["type"] == "text"
        }
        thinking_delta_indexes = {
            c["index"]
            for c in chunks
            if c["type"] == "content_block_delta" and c["delta"]["type"] == "thinking_delta"
        }
        assert text_block_indexes.isdisjoint(thinking_delta_indexes)
