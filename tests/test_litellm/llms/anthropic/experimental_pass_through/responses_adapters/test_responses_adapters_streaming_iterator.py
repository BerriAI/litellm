"""
Tests for AnthropicResponsesStreamWrapper
(litellm/llms/anthropic/experimental_pass_through/responses_adapters/streaming_iterator.py)
"""

import os
import sys
from types import SimpleNamespace

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


def _completed_event_with_usage(usage: object) -> SimpleNamespace:
    return SimpleNamespace(
        type="response.completed",
        response=SimpleNamespace(status="completed", usage=usage, output=[]),
    )


def _message_delta_usage(chunks: list) -> dict:
    for chunk in chunks:
        if chunk.get("type") == "message_delta":
            return chunk["usage"]
    raise AssertionError("message_delta chunk not found")


class TestProcessEventUsageMapping:
    """Responses usage is translated to Anthropic Messages usage."""

    def test_openai_responses_cached_tokens_map_to_cache_read_tokens(self):
        usage = SimpleNamespace(
            input_tokens=1000,
            output_tokens=75,
            input_tokens_details=SimpleNamespace(cached_tokens=800),
        )

        chunks = _process_all([_completed_event_with_usage(usage)])

        assert _message_delta_usage(chunks) == {
            "input_tokens": 200,
            "output_tokens": 75,
            "cache_read_input_tokens": 800,
        }

    def test_dict_responses_cached_tokens_map_to_cache_read_tokens(self):
        usage = SimpleNamespace(
            input_tokens=1000,
            output_tokens=75,
            input_tokens_details={"cached_tokens": 800},
        )

        chunks = _process_all([_completed_event_with_usage(usage)])

        assert _message_delta_usage(chunks) == {
            "input_tokens": 200,
            "output_tokens": 75,
            "cache_read_input_tokens": 800,
        }

    def test_anthropic_usage_fallback_does_not_double_subtract_input_tokens(self):
        usage = SimpleNamespace(
            input_tokens=200,
            output_tokens=75,
            input_tokens_details=None,
            cache_creation_input_tokens=10,
            cache_read_input_tokens=800,
        )

        chunks = _process_all([_completed_event_with_usage(usage)])

        assert _message_delta_usage(chunks) == {
            "input_tokens": 200,
            "output_tokens": 75,
            "cache_creation_input_tokens": 10,
            "cache_read_input_tokens": 800,
        }


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
