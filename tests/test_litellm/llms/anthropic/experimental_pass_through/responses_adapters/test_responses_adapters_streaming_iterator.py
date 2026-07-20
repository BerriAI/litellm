"""
Tests for AnthropicResponsesStreamWrapper
(litellm/llms/anthropic/experimental_pass_through/responses_adapters/streaming_iterator.py)
"""

import asyncio
import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../..")))

from litellm.llms.anthropic.experimental_pass_through.responses_adapters.streaming_iterator import (
    AnthropicResponsesStreamWrapper,
)


def _process_all(events: list) -> list:
    wrapper = AnthropicResponsesStreamWrapper(responses_stream=None, model="m")
    for event in events:
        wrapper._process_event(event)
    return list(wrapper._chunk_queue)


def _drain_async(events: list) -> list:
    async def _gen():
        for event in events:
            yield event

    async def _run() -> list:
        wrapper = AnthropicResponsesStreamWrapper(responses_stream=_gen(), model="m")
        return [chunk async for chunk in wrapper]

    return asyncio.run(_run())


class TestMessageStartEmittedExactlyOnce:
    """The ``__anext__`` fallback emits ``message_start`` before consuming the
    stream, so ``_process_event`` must not emit a second one when
    ``response.created`` later arrives. Two ``message_start`` events (byte
    identical, same id) break strict Anthropic SDK clients (e.g. Claude Code)
    with 'Content block is not a thinking block' once thinking blocks follow."""

    def test_response_created_does_not_duplicate_message_start(self):
        chunks = _drain_async(
            [
                {"type": "response.created"},
                {"type": "response.output_text.delta", "item_id": "m1", "delta": "hi"},
            ]
        )
        message_starts = [c for c in chunks if c["type"] == "message_start"]
        assert len(message_starts) == 1

    def test_message_start_is_first_event(self):
        chunks = _drain_async([{"type": "response.created"}])
        assert chunks[0]["type"] == "message_start"


class TestProcessEventResponseCreatedGuard:
    """``_process_event`` must emit ``message_start`` exactly once even if
    ``response.created`` arrives more than once. The guard mirrors the
    ``__anext__`` fallback's ``_sent_message_start`` flag, so a direct caller
    and the async fallback can never double-emit. This also exercises the
    guard's emit-branch, which the async path never reaches because the
    fallback sets the flag before the upstream stream is consumed."""

    def test_first_response_created_emits_message_start(self):
        chunks = _process_all([{"type": "response.created"}])
        assert len(chunks) == 1
        assert chunks[0]["type"] == "message_start"
        assert chunks[0]["message"]["model"] == "m"

    def test_second_response_created_is_skipped(self):
        chunks = _process_all([{"type": "response.created"}, {"type": "response.created"}])
        message_starts = [c for c in chunks if c["type"] == "message_start"]
        assert len(message_starts) == 1


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


class TestUpstreamFailuresEmitAnthropicErrorEvents:
    """Mid-stream upstream failures must surface as Anthropic `error` SSE
    events, not as a well-formed empty stream (message_delta(end_turn) +
    message_stop) that clients can't distinguish from a model that
    legitimately produced no output.

    https://github.com/BerriAI/litellm/issues/32086
    """

    def test_response_failed_emits_error_event_not_end_turn(self):
        chunks = _process_all(
            [
                {
                    "type": "response.failed",
                    "response": {
                        "status": "failed",
                        "error": {
                            "code": "server_error",
                            "message": "The model is overloaded.",
                        },
                    },
                }
            ]
        )
        assert [c["type"] for c in chunks] == ["error"]
        assert chunks[0]["error"]["type"] == "api_error"
        assert "server_error" in chunks[0]["error"]["message"]
        assert "The model is overloaded." in chunks[0]["error"]["message"]

    def test_response_failed_without_error_details_still_emits_error_event(self):
        chunks = _process_all([{"type": "response.failed", "response": {"status": "failed"}}])
        assert [c["type"] for c in chunks] == ["error"]
        assert chunks[0]["error"]["type"] == "api_error"
        assert chunks[0]["error"]["message"]

    def test_error_event_without_any_message_uses_fallback(self):
        chunks = _process_all([{"type": "error"}])
        assert [c["type"] for c in chunks] == ["error"]
        assert chunks[0]["error"]["message"]

    def test_top_level_error_event_emits_error_event(self):
        chunks = _process_all([{"type": "error", "code": "rate_limit_exceeded", "message": "Rate limit reached."}])
        assert [c["type"] for c in chunks] == ["error"]
        assert chunks[0]["error"]["message"] == "Rate limit reached."

    def test_response_completed_still_emits_message_delta_and_stop(self):
        chunks = _process_all([{"type": "response.completed", "response": {"status": "completed"}}])
        assert [c["type"] for c in chunks] == ["message_delta", "message_stop"]
        assert chunks[0]["delta"]["stop_reason"] == "end_turn"

    def test_response_incomplete_still_maps_to_max_tokens(self):
        # NOTE: the completed/incomplete branch reads `status` via getattr,
        # so it only sees object-shaped responses (dict-shaped events fall
        # back to end_turn) — mirror that here with an object.
        response = SimpleNamespace(status="incomplete", usage=None, output=[])
        chunks = _process_all([{"type": "response.incomplete", "response": response}])
        assert [c["type"] for c in chunks] == ["message_delta", "message_stop"]
        assert chunks[0]["delta"]["stop_reason"] == "max_tokens"

    @pytest.mark.asyncio
    async def test_mid_stream_exception_emits_error_event_before_ending(self):
        async def exploding_stream():
            yield {"type": "response.created"}
            raise RuntimeError("connection reset by https://user:secret@host/v1")

        wrapper = AnthropicResponsesStreamWrapper(responses_stream=exploding_stream(), model="m")
        chunks = []
        async for chunk in wrapper:
            chunks.append(chunk)

        assert chunks[0]["type"] == "message_start"
        assert chunks[-1]["type"] == "error"
        # A non-empty error event is surfaced, but the raw transport exception
        # (which can carry upstream URLs/credentials) must not reach the client.
        assert chunks[-1]["error"]["message"] == "Upstream provider error while streaming."
        assert "secret" not in chunks[-1]["error"]["message"]

    @pytest.mark.asyncio
    async def test_sse_wrapper_renders_event_error(self):
        async def failing_stream():
            yield {"type": "response.created"}
            yield {
                "type": "response.failed",
                "response": {
                    "status": "failed",
                    "error": {"code": "server_error", "message": "boom"},
                },
            }

        wrapper = AnthropicResponsesStreamWrapper(responses_stream=failing_stream(), model="m")
        payloads = []
        async for raw in wrapper.async_anthropic_sse_wrapper():
            payloads.append(raw.decode())

        assert any(p.startswith("event: error\n") for p in payloads)
        error_payload = next(p for p in payloads if p.startswith("event: error\n"))
        data = json.loads(error_payload.split("data: ", 1)[1].strip())
        assert data["error"]["type"] == "api_error"
        assert "boom" in data["error"]["message"]


class TestReviewFindings:
    """Regression tests for review findings on the error-event handling."""

    def test_error_event_with_nested_error_message(self):
        chunks = _process_all(
            [
                {
                    "type": "error",
                    "error": {"code": "overloaded", "message": "Nested boom."},
                }
            ]
        )
        assert [c["type"] for c in chunks] == ["error"]
        assert chunks[0]["error"]["message"] == "Nested boom."

    @pytest.mark.asyncio
    async def test_persistently_failing_upstream_emits_exactly_one_error(self):
        class AlwaysFailingStream:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise RuntimeError("max streaming duration exceeded")

        wrapper = AnthropicResponsesStreamWrapper(responses_stream=AlwaysFailingStream(), model="m")
        chunks = []
        async for chunk in wrapper:
            chunks.append(chunk)

        assert [c["type"] for c in chunks] == ["message_start", "error"]

    @pytest.mark.asyncio
    async def test_no_chunks_after_message_stop(self):
        async def completed_stream():
            yield {"type": "response.created"}
            yield {"type": "response.completed", "response": {"status": "completed"}}

        wrapper = AnthropicResponsesStreamWrapper(responses_stream=completed_stream(), model="m")
        chunks = [c async for c in wrapper]
        assert chunks[-1]["type"] == "message_stop"
        assert [c["type"] for c in chunks].count("message_stop") == 1
