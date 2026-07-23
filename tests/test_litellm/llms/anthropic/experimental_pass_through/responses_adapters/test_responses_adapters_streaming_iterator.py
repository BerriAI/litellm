"""
Tests for AnthropicResponsesStreamWrapper
(litellm/llms/anthropic/experimental_pass_through/responses_adapters/streaming_iterator.py)
"""

import os
import sys

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


def _assert_delta_block_types_are_valid(chunks: list) -> None:
    block_types = {}
    for chunk in chunks:
        if chunk["type"] == "content_block_start":
            block_types[chunk["index"]] = chunk["content_block"]["type"]
        elif chunk["type"] == "content_block_delta":
            delta_type = chunk["delta"]["type"]
            block_type = block_types[chunk["index"]]
            if delta_type in ("thinking_delta", "signature_delta"):
                assert block_type == "thinking"
            elif delta_type == "text_delta":
                assert block_type == "text"
            elif delta_type == "input_json_delta":
                assert block_type == "tool_use"


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
        assert chunks[1]["type"] == "content_block_start"
        assert chunks[1]["content_block"] == {"type": "text", "text": ""}
        assert [c["index"] for c in chunks[1:]] == [1, 1]

    def test_process_event_registered_item_id_does_not_synthesize_start(self):
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


class TestProcessEventReasoningDeltaBlockType:
    def test_reasoning_delta_after_text_block_opens_thinking_block(self):
        chunks = _process_all(
            [
                {
                    "type": "response.output_item.added",
                    "item": {"type": "message", "id": "m1"},
                },
                {"type": "response.output_text.delta", "item_id": "m1", "delta": "Hi"},
                {
                    "type": "response.reasoning_summary_text.delta",
                    "item_id": "rs1",
                    "delta": "Thinking",
                },
            ]
        )

        assert [(c["type"], c["index"]) for c in chunks] == [
            ("content_block_start", 0),
            ("content_block_delta", 0),
            ("content_block_start", 1),
            ("content_block_delta", 1),
        ]
        assert chunks[2]["content_block"] == {"type": "thinking", "thinking": ""}
        assert chunks[3]["delta"] == {
            "type": "thinking_delta",
            "thinking": "Thinking",
        }
        _assert_delta_block_types_are_valid(chunks)

    def test_reasoning_delta_without_output_item_added_opens_thinking_block(self):
        chunks = _process_all(
            [
                {
                    "type": "response.reasoning_summary_text.delta",
                    "item_id": "rs1",
                    "delta": "Thinking",
                },
            ]
        )

        assert [(c["type"], c["index"]) for c in chunks] == [
            ("content_block_start", 0),
            ("content_block_delta", 0),
        ]
        assert chunks[0]["content_block"] == {"type": "thinking", "thinking": ""}
        assert chunks[1]["delta"] == {
            "type": "thinking_delta",
            "thinking": "Thinking",
        }
        _assert_delta_block_types_are_valid(chunks)

    def test_text_delta_after_reasoning_block_opens_text_block(self):
        chunks = _process_all(
            [
                {
                    "type": "response.output_item.added",
                    "item": {"type": "reasoning", "id": "rs1"},
                },
                {
                    "type": "response.reasoning_summary_text.delta",
                    "item_id": "rs1",
                    "delta": "Thinking",
                },
                {"type": "response.output_text.delta", "item_id": "m1", "delta": "Hi"},
            ]
        )

        assert [(c["type"], c["index"]) for c in chunks] == [
            ("content_block_start", 0),
            ("content_block_delta", 0),
            ("content_block_start", 1),
            ("content_block_delta", 1),
        ]
        assert chunks[2]["content_block"] == {"type": "text", "text": ""}
        assert chunks[3]["delta"] == {"type": "text_delta", "text": "Hi"}
        _assert_delta_block_types_are_valid(chunks)
