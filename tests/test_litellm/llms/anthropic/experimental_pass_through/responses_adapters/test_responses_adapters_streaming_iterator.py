"""
Tests for AnthropicResponsesStreamWrapper
(litellm/llms/anthropic/experimental_pass_through/responses_adapters/streaming_iterator.py)
"""

import asyncio
import os
import sys

from pydantic import BaseModel, ConfigDict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../..")))

from litellm.llms.anthropic.experimental_pass_through.responses_adapters.streaming_iterator import (
    AnthropicResponsesStreamWrapper,
)


class _Details(BaseModel):
    model_config = ConfigDict(extra="allow")
    cached_tokens: int = 0


class _Usage(BaseModel):
    model_config = ConfigDict(extra="allow")
    input_tokens: int
    output_tokens: int
    input_tokens_details: _Details


class _Response(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: str = "completed"
    usage: _Usage
    output: list = []


class _Event(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str = "response.completed"
    response: _Response


def _completed_usage(events: list) -> dict:
    return next(c for c in _process_all(events) if c.get("type") == "message_delta")["usage"]


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


class TestResponseCompletedCacheTokens:
    """message_delta must surface OpenAI Responses cache-read/write counts. OpenAI reports
    them under input_tokens_details (cached_tokens / cache_write_tokens), not as the
    Anthropic-native cache_*_input_tokens names the adapter previously read, so cache reads
    were always billed at the full input rate. See issue #35127."""

    def test_openai_cached_tokens_from_model_details(self):
        event = _Event(
            response=_Response(
                usage=_Usage(
                    input_tokens=12000,
                    output_tokens=40,
                    input_tokens_details=_Details(cached_tokens=11008),
                )
            )
        )
        assert _completed_usage([event]) == {
            "input_tokens": 12000,
            "output_tokens": 40,
            "cache_read_input_tokens": 11008,
        }

    def test_openai_cached_tokens_from_dict_details(self):
        class _DictDetailsUsage(BaseModel):
            model_config = ConfigDict(extra="allow")
            input_tokens: int
            output_tokens: int
            input_tokens_details: dict

        event = _Event.model_construct(
            response=_Response.model_construct(
                status="completed",
                output=[],
                usage=_DictDetailsUsage(
                    input_tokens=12000,
                    output_tokens=40,
                    input_tokens_details={"cached_tokens": 11008},
                ),
            )
        )
        assert _completed_usage([event]) == {
            "input_tokens": 12000,
            "output_tokens": 40,
            "cache_read_input_tokens": 11008,
        }

    def test_cache_write_tokens_pydantic_extra_becomes_cache_creation(self):
        event = _Event(
            response=_Response(
                usage=_Usage(
                    input_tokens=172000,
                    output_tokens=40,
                    input_tokens_details=_Details(cached_tokens=155000, cache_write_tokens=3000),
                )
            )
        )
        assert _completed_usage([event]) == {
            "input_tokens": 172000,
            "output_tokens": 40,
            "cache_read_input_tokens": 155000,
            "cache_creation_input_tokens": 3000,
        }

    def test_anthropic_native_fields_take_precedence(self):
        class _AnthropicUsage(BaseModel):
            model_config = ConfigDict(extra="allow")
            input_tokens: int
            output_tokens: int
            cache_read_input_tokens: int
            cache_creation_input_tokens: int
            input_tokens_details: _Details

        event = _Event.model_construct(
            response=_Response.model_construct(
                status="completed",
                output=[],
                usage=_AnthropicUsage(
                    input_tokens=500,
                    output_tokens=40,
                    cache_read_input_tokens=100,
                    cache_creation_input_tokens=25,
                    input_tokens_details=_Details(cached_tokens=999),
                ),
            )
        )
        usage = _completed_usage([event])
        assert usage["cache_read_input_tokens"] == 100
        assert usage["cache_creation_input_tokens"] == 25

    def test_absent_cache_fields_omit_keys(self):
        event = _Event(
            response=_Response(
                usage=_Usage(
                    input_tokens=500,
                    output_tokens=40,
                    input_tokens_details=_Details(cached_tokens=0),
                )
            )
        )
        assert _completed_usage([event]) == {"input_tokens": 500, "output_tokens": 40}
