"""
Tests for AnthropicResponsesStreamWrapper
(litellm/llms/anthropic/experimental_pass_through/responses_adapters/streaming_iterator.py)
"""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../..")))

from litellm.llms.anthropic.experimental_pass_through.responses_adapters.streaming_iterator import (
    AnthropicResponsesStreamWrapper,
)


def _process_all(events: list) -> list:
    wrapper = AnthropicResponsesStreamWrapper(responses_stream=None, model="m")
    for event in events:
        wrapper._process_event(event)
    return list(wrapper._chunk_queue)


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


class TestDictShapedCompletedEvents:
    """Usage, status, and output must be read from dict-shaped
    `response.completed` payloads, not only attribute-shaped ones.

    Previously this branch used getattr-only access, so dict-shaped events
    always produced usage 0/0 (disabling spend tracking and TPM enforcement,
    https://github.com/BerriAI/litellm/issues/32086) and mapped
    `response.incomplete` to end_turn instead of max_tokens.
    """

    def test_dict_usage_is_extracted(self):
        chunks = _process_all(
            [
                {
                    "type": "response.completed",
                    "response": {
                        "status": "completed",
                        "usage": {
                            "input_tokens": 11,
                            "output_tokens": 42,
                            "cache_read_input_tokens": 7,
                        },
                    },
                }
            ]
        )
        assert chunks[0]["type"] == "message_delta"
        assert chunks[0]["usage"] == {
            "input_tokens": 11,
            "output_tokens": 42,
            "cache_read_input_tokens": 7,
        }

    def test_dict_incomplete_maps_to_max_tokens(self):
        chunks = _process_all([{"type": "response.incomplete", "response": {"status": "incomplete"}}])
        assert chunks[0]["delta"]["stop_reason"] == "max_tokens"

    def test_dict_function_call_output_maps_to_tool_use(self):
        chunks = _process_all(
            [
                {
                    "type": "response.completed",
                    "response": {
                        "status": "completed",
                        "output": [{"type": "function_call", "name": "get_weather"}],
                    },
                }
            ]
        )
        assert chunks[0]["delta"]["stop_reason"] == "tool_use"

    def test_object_shaped_usage_still_extracted(self):
        usage = SimpleNamespace(
            input_tokens=3,
            output_tokens=5,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
        response = SimpleNamespace(status="completed", usage=usage, output=[])
        chunks = _process_all([{"type": "response.completed", "response": response}])
        assert chunks[0]["usage"] == {"input_tokens": 3, "output_tokens": 5}
