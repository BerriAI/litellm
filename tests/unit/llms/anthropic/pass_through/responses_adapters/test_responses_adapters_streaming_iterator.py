"""
Tests for AnthropicResponsesStreamWrapper
(litellm/llms/anthropic/pass_through/responses_adapters/streaming_iterator.py)
"""

import asyncio
import itertools
import json
import os
import sys
import threading
from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from typing import Annotated, Final, Literal

import httpx
import pytest
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../..")))

import litellm
from litellm.exceptions import MidStreamFallbackError
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging as LitellmLogging
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    encrypted_reasoning_signature,
)
from litellm.llms.anthropic.pass_through.messages.utils import INCOMPLETE_STREAM_ERROR_MESSAGE
from litellm.llms.anthropic.pass_through.responses_adapters.streaming_iterator import (
    AnthropicResponsesSSEByteStream,
    AnthropicResponsesStreamWrapper,
)
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator, SyncResponsesAPIStreamingIterator
from litellm.types.llms.openai import ResponseFailedEvent, ResponsesAPIResponse


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


def _drain_sync_upstream(events: list) -> list:
    async def _run() -> list:
        wrapper = AnthropicResponsesStreamWrapper(responses_stream=iter(events), model="m")
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

    def test_sync_upstream_iterator_is_consumed(self):
        chunks = _drain_sync_upstream(
            [
                {"type": "response.created"},
                {"type": "response.output_text.delta", "item_id": "m1", "delta": "hi"},
            ]
        )
        assert any(chunk.get("delta", {}).get("text") == "hi" for chunk in chunks)


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


class TestReasoningItemWithoutSummaryText:
    """Regression: a reasoning item whose summary never produces text must not
    surface as a thinking content block.

    OpenAI emits ``response.output_item.added`` with ``type: "reasoning"`` on
    every reasoning turn, but only emits
    ``response.reasoning_summary_text.delta`` when a summary was requested and
    the model actually produced one. Eagerly opening the block on
    ``output_item.added`` left ``{"type": "thinking", "thinking": ""}`` in the
    assistant turn, which clients persist in their session transcript. Replaying
    that transcript against an Anthropic model (what ``claude --resume`` does
    once the resumed session falls back to the default Anthropic model) fails
    with::

        400 invalid_request_error - messages.2.content.0.thinking:
        each thinking block must contain thinking

    So the thinking block is opened on the first non-empty summary delta.
    """

    @staticmethod
    def _gpt_turn(reasoning_summary_deltas: list, encrypted_content: str | None = None) -> list:
        return [
            {"type": "response.created"},
            {"type": "response.output_item.added", "item": {"type": "reasoning", "id": "rs_1"}},
            *(
                {"type": "response.reasoning_summary_text.delta", "item_id": "rs_1", "delta": delta}
                for delta in reasoning_summary_deltas
            ),
            {
                "type": "response.output_item.done",
                "item": {"type": "reasoning", "id": "rs_1", "encrypted_content": encrypted_content},
            },
            {"type": "response.output_item.added", "item": {"type": "message", "id": "msg_1"}},
            {"type": "response.output_text.delta", "item_id": "msg_1", "delta": "Hello"},
            {"type": "response.output_item.done", "item": {"type": "message", "id": "msg_1"}},
            {"type": "response.completed"},
        ]

    def test_reasoning_without_summary_emits_no_thinking_block(self):
        chunks = _drain_async(self._gpt_turn(reasoning_summary_deltas=[]))

        assert not [
            c for c in chunks if c["type"] == "content_block_start" and c["content_block"]["type"] == "thinking"
        ]
        assert [(c["type"], c.get("index")) for c in chunks[1:]] == [
            ("content_block_start", 0),
            ("content_block_delta", 0),
            ("content_block_stop", 0),
            ("message_delta", None),
            ("message_stop", None),
        ]
        assert chunks[1]["content_block"] == {"type": "text", "text": ""}

    def test_reasoning_with_only_empty_summary_deltas_emits_no_thinking_block(self):
        chunks = _drain_async(self._gpt_turn(reasoning_summary_deltas=["", ""]))

        assert not [c for c in chunks if c["type"] == "content_block_delta" and c["delta"]["type"] == "thinking_delta"]
        assert not [
            c for c in chunks if c["type"] == "content_block_start" and c["content_block"]["type"] == "thinking"
        ]

    def test_reasoning_with_summary_text_still_emits_a_thinking_block(self):
        chunks = _drain_async(self._gpt_turn(reasoning_summary_deltas=["Weigh", "ing options"]))

        assert [(c["type"], c.get("index")) for c in chunks[1:]] == [
            ("content_block_start", 0),
            ("content_block_delta", 0),
            ("content_block_delta", 0),
            ("content_block_delta", 0),
            ("content_block_stop", 0),
            ("content_block_start", 1),
            ("content_block_delta", 1),
            ("content_block_stop", 1),
            ("message_delta", None),
            ("message_stop", None),
        ]
        assert chunks[1]["content_block"] == {"type": "thinking", "thinking": "", "signature": ""}
        assert "".join(c["delta"]["thinking"] for c in chunks[2:4]) == "Weighing options"

    def test_the_reasoning_item_id_is_never_streamed_as_a_signature(self):
        """A stand-in signature would be replayed as a real one, so the block closes with an empty one."""
        chunks = _drain_async(self._gpt_turn(reasoning_summary_deltas=["Weighing options"]))

        signature_deltas: Final = [c for c in chunks if c.get("delta", {}).get("type") == "signature_delta"]
        assert [(c["index"], c["delta"]["signature"]) for c in signature_deltas] == [(0, "")]
        assert chunks[chunks.index(signature_deltas[0]) + 1] == {"type": "content_block_stop", "index": 0}


class TestRawReasoningTextIsStreamedAsThinking:
    """Servers that return a reasoning item's raw text (vLLM and OpenRouter for gpt-oss)
    stream it as ``response.reasoning_text.delta`` instead of a summary."""

    def test_reasoning_text_deltas_become_one_closed_thinking_block(self):
        chunks: Final = _drain_async(
            [
                {"type": "response.created"},
                {"type": "response.output_item.added", "item": {"type": "reasoning", "id": "rs_1"}},
                {"type": "response.reasoning_text.delta", "item_id": "rs_1", "content_index": 0, "delta": "17*23"},
                {"type": "response.reasoning_text.delta", "item_id": "rs_1", "content_index": 0, "delta": " is 391."},
                {"type": "response.output_item.done", "item": {"type": "reasoning", "id": "rs_1"}},
                {"type": "response.output_item.added", "item": {"type": "message", "id": "msg_1"}},
                {"type": "response.output_text.delta", "item_id": "msg_1", "delta": "391"},
                {"type": "response.output_item.done", "item": {"type": "message", "id": "msg_1"}},
                {"type": "response.completed"},
            ]
        )

        assert [(c["type"], c.get("index"), c.get("content_block") or c.get("delta")) for c in chunks[1:-2]] == [
            ("content_block_start", 0, {"type": "thinking", "thinking": "", "signature": ""}),
            ("content_block_delta", 0, {"type": "thinking_delta", "thinking": "17*23"}),
            ("content_block_delta", 0, {"type": "thinking_delta", "thinking": " is 391."}),
            ("content_block_delta", 0, {"type": "signature_delta", "signature": ""}),
            ("content_block_stop", 0, None),
            ("content_block_start", 1, {"type": "text", "text": ""}),
            ("content_block_delta", 1, {"type": "text_delta", "text": "391"}),
            ("content_block_stop", 1, None),
        ]

    def test_a_whole_reasoning_item_without_deltas_becomes_one_closed_thinking_block(self):
        reasoning_item: Final = {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": "Just answer 391."}],
            "encrypted_content": None,
        }
        chunks: Final = _drain_async(
            [
                {"type": "response.created"},
                {"type": "response.output_item.added", "item": reasoning_item},
                {"type": "response.output_item.done", "item": reasoning_item},
                {"type": "response.output_item.added", "item": {"type": "message", "id": "msg_1"}},
                {"type": "response.output_text.delta", "item_id": "msg_1", "delta": "391"},
                {"type": "response.output_item.done", "item": {"type": "message", "id": "msg_1"}},
                {"type": "response.completed"},
            ]
        )

        assert [(c["type"], c.get("index"), c.get("content_block") or c.get("delta")) for c in chunks[1:-2]] == [
            ("content_block_start", 0, {"type": "thinking", "thinking": "", "signature": ""}),
            ("content_block_delta", 0, {"type": "thinking_delta", "thinking": "Just answer 391."}),
            ("content_block_delta", 0, {"type": "signature_delta", "signature": ""}),
            ("content_block_stop", 0, None),
            ("content_block_start", 1, {"type": "text", "text": ""}),
            ("content_block_delta", 1, {"type": "text_delta", "text": "391"}),
            ("content_block_stop", 1, None),
        ]


_ENCRYPTED_REASONING = "gAAAAABp_encrypted_reasoning_bytes_only_openai_can_read"


class TestEncryptedReasoningIsStreamedForReplay:
    """Regression for https://github.com/BerriAI/litellm/issues/40288.

    The client echoes a thinking block's signature (or a redacted block's data) back on the
    next turn, so the item's ``encrypted_content`` has to reach it through one of those.
    """

    def test_encrypted_content_is_streamed_as_the_signature_before_the_block_closes(self):
        chunks = _drain_async(
            TestReasoningItemWithoutSummaryText._gpt_turn(
                reasoning_summary_deltas=["Weighing options"], encrypted_content=_ENCRYPTED_REASONING
            )
        )

        assert [(c["type"], c.get("index"), c.get("delta", {}).get("type")) for c in chunks[1:5]] == [
            ("content_block_start", 0, None),
            ("content_block_delta", 0, "thinking_delta"),
            ("content_block_delta", 0, "signature_delta"),
            ("content_block_stop", 0, None),
        ]
        assert chunks[3]["delta"]["signature"] == encrypted_reasoning_signature(_ENCRYPTED_REASONING)

    def test_reasoning_without_summary_streams_a_redacted_thinking_block(self):
        chunks = _drain_async(
            TestReasoningItemWithoutSummaryText._gpt_turn(
                reasoning_summary_deltas=[], encrypted_content=_ENCRYPTED_REASONING
            )
        )

        assert [(c["type"], c.get("index")) for c in chunks[1:]] == [
            ("content_block_start", 0),
            ("content_block_stop", 0),
            ("content_block_start", 1),
            ("content_block_delta", 1),
            ("content_block_stop", 1),
            ("message_delta", None),
            ("message_stop", None),
        ]
        assert chunks[1]["content_block"] == {
            "type": "redacted_thinking",
            "data": encrypted_reasoning_signature(_ENCRYPTED_REASONING),
        }

    def test_summary_parts_are_separated_inside_the_one_thinking_block(self):
        """Two summary parts read as two paragraphs, not as one run-on sentence."""
        events = [
            {"type": "response.created"},
            {"type": "response.output_item.added", "item": {"type": "reasoning", "id": "rs_1"}},
            {"type": "response.reasoning_summary_part.added", "item_id": "rs_1", "summary_index": 0},
            {"type": "response.reasoning_summary_text.delta", "item_id": "rs_1", "delta": "First."},
            {"type": "response.reasoning_summary_part.added", "item_id": "rs_1", "summary_index": 1},
            {"type": "response.reasoning_summary_text.delta", "item_id": "rs_1", "delta": "Second."},
            {"type": "response.output_item.done", "item": {"type": "reasoning", "id": "rs_1"}},
        ]
        chunks = _process_all(events)

        thinking = "".join(c["delta"]["thinking"] for c in chunks if c.get("delta", {}).get("type") == "thinking_delta")
        assert thinking == "First.\n\nSecond."
        assert [c["type"] for c in chunks].count("content_block_start") == 1


class TestToolUseBlockClosedExactlyOnce:
    """Regression for https://github.com/BerriAI/litellm/issues/37273.

    With ``custom_llm_provider: openai`` + ``use_chat_completions_api: true``,
    ``/v1/messages`` streams through ``LiteLLMCompletionStreamingIterator``,
    which ends a tool-call turn with two ``response.output_item.done`` events:
    one for the function_call item (id = call_id) and one for a synthetic
    message item whose id is the upstream chatcmpl id and was never opened as a
    content block. Resolving that unknown item id to ``_current_block_index``
    closed the tool_use block a second time::

        content_block_start[0](tool_use) -> content_block_stop[0]
        -> content_block_stop[0] -> message_delta(stop_reason=tool_use)

    Anthropic SDK clients (e.g. Claude Code) materialize one tool_use block per
    ``content_block_stop``, so the tool executed twice. An ``output_item.done``
    for an item that never opened a block must emit nothing.
    """

    @staticmethod
    def _chat_completions_bridge_tool_turn() -> list[dict[str, object]]:
        return [
            {"type": "response.created"},
            {
                "type": "response.output_item.added",
                "item": {"type": "function_call", "id": "call_1", "call_id": "call_1", "name": "get_weather"},
            },
            {"type": "response.function_call_arguments.delta", "item_id": "call_1", "delta": '{"city": "'},
            {"type": "response.function_call_arguments.delta", "item_id": "call_1", "delta": 'Tokyo"}'},
            {
                "type": "response.function_call_arguments.done",
                "item_id": "call_1",
                "arguments": '{"city": "Tokyo"}',
            },
            {
                "type": "response.output_item.done",
                "item": {"type": "function_call", "id": "call_1", "call_id": "call_1", "status": "completed"},
            },
            {
                "type": "response.output_item.done",
                "item": {"type": "message", "id": "chatcmpl-123", "status": "completed"},
            },
            {"type": "response.completed"},
        ]

    def test_one_content_block_stop_per_content_block_start(self):
        chunks = _drain_async(self._chat_completions_bridge_tool_turn())

        starts = [c["index"] for c in chunks if c["type"] == "content_block_start"]
        stops = [c["index"] for c in chunks if c["type"] == "content_block_stop"]
        assert starts == [0]
        assert stops == [0]

    def test_tool_turn_event_order(self):
        chunks = _drain_async(self._chat_completions_bridge_tool_turn())

        assert [(c["type"], c.get("index")) for c in chunks] == [
            ("message_start", None),
            ("content_block_start", 0),
            ("content_block_delta", 0),
            ("content_block_delta", 0),
            ("content_block_stop", 0),
            ("message_delta", None),
            ("message_stop", None),
        ]
        assert chunks[1]["content_block"] == {
            "type": "tool_use",
            "id": "call_1",
            "name": "get_weather",
            "input": {},
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
                {"type": "response.reasoning_summary_text.delta", "item_id": "rs_1", "delta": "hm"},
                {"type": "response.output_text.delta", "item_id": "m1", "delta": "Hi"},
            ]
        )
        assert chunks[2]["type"] == "content_block_start"
        assert chunks[2]["content_block"] == {"type": "text", "text": ""}
        assert [c["index"] for c in chunks[2:]] == [1, 1]

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


class TestResponseCompletedUsage:
    """The Anthropic ``message_delta`` usage must report cache reads/writes and
    exclude them from ``input_tokens``, so spend is not billed at the uncached
    input rate."""

    def test_response_completed_usage_carries_cache_tokens(self):
        from litellm.types.llms.openai import ResponseAPIUsage

        response = SimpleNamespace(
            status="completed",
            output=[],
            usage=ResponseAPIUsage(
                input_tokens=4017,
                input_tokens_details={"cached_tokens": 4004, "cache_write_tokens": 10},
                output_tokens=5,
                total_tokens=4022,
            ),
        )
        chunks = _process_all([{"type": "response.completed", "response": response}])
        message_delta = next(c for c in chunks if c["type"] == "message_delta")
        assert message_delta["usage"] == {
            "input_tokens": 3,
            "output_tokens": 5,
            "cache_creation_input_tokens": 10,
            "cache_read_input_tokens": 4004,
        }


class TestRefusalStreamEvents:
    def test_refusal_event_sequence_emits_refusal_text_and_stop_details(self):
        response = SimpleNamespace(
            status="completed",
            output=[{"type": "message", "content": [{"type": "refusal", "refusal": "I cannot fulfill this."}]}],
            usage=None,
        )
        chunks = _process_all(
            [
                {"type": "response.created"},
                {"type": "response.output_item.added", "item": {"type": "message", "id": "msg_1"}},
                {"type": "response.refusal.delta", "item_id": "msg_1", "delta": "I cannot fulfill this."},
                {"type": "response.output_item.done", "item": {"type": "message", "id": "msg_1"}},
                {"type": "response.completed", "response": response},
            ]
        )
        assert [chunk["type"] for chunk in chunks] == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
        assert chunks[2]["delta"] == {"type": "text_delta", "text": "I cannot fulfill this."}
        assert chunks[4]["delta"] == {
            "stop_reason": "refusal",
            "stop_sequence": None,
            "stop_details": {
                "type": "refusal",
                "category": None,
                "explanation": "I cannot fulfill this.",
            },
        }

    def test_response_completed_with_refusal_sets_stop_reason_refusal(self):
        response = SimpleNamespace(
            status="completed",
            output=[{"type": "message", "content": [{"type": "refusal", "refusal": "Policy violation"}]}],
            usage=None,
        )
        chunks = _process_all([{"type": "response.completed", "response": response}])
        message_delta = next(c for c in chunks if c["type"] == "message_delta")
        assert message_delta["delta"]["stop_reason"] == "refusal"

    def test_incomplete_status_takes_precedence_over_refusal(self):
        response = SimpleNamespace(
            status="incomplete",
            output=[{"type": "message", "content": [{"type": "refusal", "refusal": "Partial refusal"}]}],
            usage=None,
        )
        chunks = _process_all([{"type": "response.incomplete", "response": response}])
        message_delta = next(c for c in chunks if c["type"] == "message_delta")
        assert message_delta["delta"]["stop_reason"] == "max_tokens"
        assert "stop_details" not in message_delta["delta"]


def _collect(stream) -> list:
    async def _run() -> list:
        wrapper = AnthropicResponsesStreamWrapper(responses_stream=stream, model="m")
        return [chunk async for chunk in wrapper]

    return asyncio.run(_run())


class TestUpstreamFailureEndsStreamWithErrorEvent:
    """A provider failure must reach the Anthropic client as an ``error`` event that
    ends the stream, never as a fabricated ``end_turn`` or a silent close."""

    def test_response_failed_event_emits_error_event_and_stops_pulling_upstream(self):
        failed = SimpleNamespace(
            status="failed",
            output=[],
            usage=None,
            error={"code": "rate_limit_exceeded", "message": "Rate limit reached for gpt-5.5, try again in 20s."},
        )

        async def _gen():
            yield {"type": "response.created"}
            yield {"type": "response.failed", "response": failed}
            raise AssertionError("upstream was pulled again after the failure")

        async def _run() -> list:
            wrapper = AnthropicResponsesStreamWrapper(responses_stream=_gen(), model="m")
            return [frame async for frame in wrapper.async_anthropic_sse_wrapper()]

        frames = asyncio.run(_run())
        assert [frame.split(b"\n", 1)[0] for frame in frames] == [b"event: message_start", b"event: error"]
        error_payload = json.loads(frames[1].split(b"data: ", 1)[1])
        assert error_payload["type"] == "error"
        assert error_payload["error"] == {
            "type": "rate_limit_error",
            "message": "Rate limit reached for gpt-5.5, try again in 20s.",
        }

    def test_raised_mid_stream_fallback_error_is_unwrapped_to_the_provider_failure(self):
        rate_limit = litellm.RateLimitError(message="You have no credits remaining.", llm_provider="openai", model="m")
        wrapped = MidStreamFallbackError(
            message=str(rate_limit),
            model="m",
            llm_provider="openai",
            original_exception=rate_limit,
            is_pre_first_chunk=True,
        )

        async def _gen():
            yield {"type": "response.created"}
            raise wrapped

        chunks = _collect(_gen())
        assert [chunk["type"] for chunk in chunks] == ["message_start", "error"]
        assert chunks[1]["error"] == {"type": "rate_limit_error", "message": rate_limit.message}

    def test_sync_upstream_transport_error_after_content_becomes_api_error_event(self):
        def _events():
            yield {"type": "response.created"}
            yield {"type": "response.output_item.added", "item": {"type": "message", "id": "msg_1"}}
            yield {"type": "response.output_text.delta", "item_id": "msg_1", "delta": "Hi"}
            raise ConnectionResetError("Response payload is not completed")

        chunks = _collect(_events())
        assert [chunk["type"] for chunk in chunks] == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "error",
        ]
        assert chunks[-1]["error"] == {"type": "api_error", "message": "Response payload is not completed"}

    def test_error_event_message_is_redacted_before_it_reaches_the_client(self):
        async def _gen():
            yield {"type": "response.created"}
            raise RuntimeError("upstream failed with key sk-proj-abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJ")

        chunks = _collect(_gen())
        assert chunks[-1]["type"] == "error"
        assert "sk-proj-" not in chunks[-1]["error"]["message"]
        assert chunks[-1]["error"]["message"].startswith("upstream failed with key")

    @pytest.mark.parametrize(
        ("raised", "expected_error"),
        [
            (
                MidStreamFallbackError(message="boom", model="m", llm_provider="openai"),
                {"type": "api_error", "message": "litellm.MidStreamFallbackError: boom"},
            ),
            (
                type("StringStatusError", (Exception,), {"status_code": "429"})("throttled"),
                {"type": "rate_limit_error", "message": "throttled"},
            ),
            (
                type("NonErrorStatusError", (Exception,), {"status_code": 200})("odd status"),
                {"type": "api_error", "message": "odd status"},
            ),
        ],
        ids=["mid-stream-fallback-without-original", "digit-string-status", "status-outside-4xx-5xx"],
    )
    def test_raised_failure_status_is_normalized_into_the_error_type(self, raised, expected_error):
        async def _gen():
            yield {"type": "response.created"}
            raise raised

        chunks = _collect(_gen())
        assert [chunk["type"] for chunk in chunks] == ["message_start", "error"]
        assert chunks[1]["error"] == expected_error

    def test_pydantic_response_failed_event_is_mapped_like_a_dict_event(self):
        failed = ResponsesAPIResponse(
            id="resp_1",
            created_at=1,
            error={"code": "server_error", "message": "The server had an error while processing your request."},
            status="failed",
            output=[],
            model="m",
            object="response",
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
        )

        async def _gen():
            yield {"type": "response.created"}
            yield ResponseFailedEvent(type="response.failed", response=failed)

        chunks = _collect(_gen())
        assert [chunk["type"] for chunk in chunks] == ["message_start", "error"]
        assert chunks[1]["error"] == {
            "type": "api_error",
            "message": "The server had an error while processing your request.",
        }

    def test_upstream_ending_without_a_terminal_event_is_an_error_not_a_silent_close(self):
        async def _gen():
            yield {"type": "response.created"}
            yield {"type": "response.output_item.added", "item": {"type": "message", "id": "msg_1"}}
            yield {"type": "response.output_text.delta", "item_id": "msg_1", "delta": "Hi"}

        chunks = _collect(_gen())
        assert [chunk["type"] for chunk in chunks] == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "error",
        ]
        assert chunks[-1]["error"] == {"type": "api_error", "message": INCOMPLETE_STREAM_ERROR_MESSAGE}

    def test_sync_upstream_ending_before_any_event_is_an_error_not_a_silent_close(self):
        chunks = _collect(iter(()))
        assert [chunk["type"] for chunk in chunks] == ["message_start", "error"]
        assert chunks[1]["error"] == {"type": "api_error", "message": INCOMPLETE_STREAM_ERROR_MESSAGE}


def _text_then_completed() -> Iterator[Mapping[str, object]]:
    yield {"type": "response.created"}
    yield {"type": "response.output_item.added", "item": {"type": "message", "id": "msg_1"}}
    yield {"type": "response.output_text.delta", "item_id": "msg_1", "delta": "Hi"}
    yield {"type": "response.output_item.done", "item": {"type": "message", "id": "msg_1"}}
    yield {"type": "response.completed"}


def _text_then_transport_error() -> Iterator[Mapping[str, object]]:
    yield {"type": "response.created"}
    yield {"type": "response.output_item.added", "item": {"type": "message", "id": "msg_1"}}
    yield {"type": "response.output_text.delta", "item_id": "msg_1", "delta": "Hi"}
    raise ConnectionResetError("Response payload is not completed")


def _text_without_a_terminal_event() -> Iterator[Mapping[str, object]]:
    yield {"type": "response.created"}
    yield {"type": "response.output_item.added", "item": {"type": "message", "id": "msg_1"}}
    yield {"type": "response.output_text.delta", "item_id": "msg_1", "delta": "Hi"}


def _failed_then_pulled_again() -> Iterator[Mapping[str, object]]:
    yield {"type": "response.created"}
    yield {
        "type": "response.failed",
        "response": {"status": "failed", "output": [], "error": {"code": "server_error", "message": "boom"}},
    }
    raise AssertionError("upstream was pulled again after the failure")


def _message_start_without_id(chunk: Mapping[str, object]) -> Mapping[str, object]:
    message: Final = chunk.get("message")
    if chunk["type"] != "message_start" or not isinstance(message, Mapping):
        return chunk
    return {**chunk, "message": {**message, "id": None}}


def _without_message_id(chunks: Iterable[Mapping[str, object]]) -> tuple[Mapping[str, object], ...]:
    return tuple(_message_start_without_id(chunk) for chunk in chunks)


@pytest.mark.parametrize(
    ("upstream", "last_event"),
    [
        (_text_then_completed, "message_stop"),
        (_text_then_transport_error, "error"),
        (_text_without_a_terminal_event, "error"),
        (_failed_then_pulled_again, "error"),
        (lambda: iter(()), "error"),
    ],
    ids=["completed", "transport-error", "no-terminal-event", "response-failed", "no-events"],
)
def test_a_for_loop_gets_the_events_an_async_caller_gets(
    upstream: Callable[[], Iterator[Mapping[str, object]]], last_event: str
) -> None:
    sync_chunks: Final = tuple(AnthropicResponsesStreamWrapper(responses_stream=upstream(), model="m"))

    assert _without_message_id(sync_chunks) == _without_message_id(_collect(upstream()))
    assert sync_chunks[0]["type"] == "message_start"
    assert sync_chunks[-1]["type"] == last_event


def test_a_for_loop_reads_the_upstream_only_as_far_as_the_caller_has_read() -> None:
    upstream: Final = _text_then_completed()
    stream: Final = AnthropicResponsesStreamWrapper(responses_stream=upstream, model="m")

    first_three: Final = tuple(chunk["type"] for chunk in itertools.islice(stream, 3))

    assert first_three == ("message_start", "content_block_start", "content_block_delta")
    assert next(upstream)["type"] == "response.output_item.done"


_RESPONSE_CREATED: Final = (
    b'data: {"type": "response.created", "response": {"id": "resp_1", "status": "in_progress", "output": []}}\n\n'
)
_MESSAGE_ADDED: Final = (
    b'data: {"type": "response.output_item.added", "output_index": 0, "item": '
    b'{"type": "message", "id": "msg_1", "role": "assistant", "content": [], "status": "in_progress"}}\n\n'
)
_TEXT_DELTA: Final = (
    b'data: {"type": "response.output_text.delta", "item_id": "msg_1", "output_index": 0, '
    b'"content_index": 0, "delta": "word "}\n\n'
)


class _EndlessProviderStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield _RESPONSE_CREATED
        yield _MESSAGE_ADDED
        while True:
            yield _TEXT_DELTA

    async def aclose(self) -> None:
        self.closed = True


class _EndlessSyncProviderStream(httpx.SyncByteStream):
    def __init__(self) -> None:
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        yield _RESPONSE_CREATED
        yield _MESSAGE_ADDED
        while True:
            yield _TEXT_DELTA

    def close(self) -> None:
        self.closed = True


class _ProviderStreamThatPausesAfterTheFirstTextDelta(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.closed = False
        self.paused = asyncio.Event()
        self.resume = asyncio.Event()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield _RESPONSE_CREATED
        yield _MESSAGE_ADDED
        yield _TEXT_DELTA
        self.paused.set()
        await self.resume.wait()
        yield _TEXT_DELTA

    async def aclose(self) -> None:
        self.closed = True


class _FailureRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.failures: tuple[object, ...] = ()

    async def async_log_failure_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.failures = (*self.failures, kwargs.get("exception"))


def _logging_obj_that_records_failures(failure_recorder: _FailureRecorder) -> LitellmLogging:
    return LitellmLogging(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type="anthropic_messages",
        start_time=datetime(2026, 1, 1),
        litellm_call_id="close-test",
        function_id="close-test",
        dynamic_async_failure_callbacks=[failure_recorder],
    )


@pytest.mark.parametrize(
    ("provider_stream", "upstream_class"),
    [
        (_EndlessProviderStream(), ResponsesAPIStreamingIterator),
        (_EndlessSyncProviderStream(), SyncResponsesAPIStreamingIterator),
    ],
    ids=["async_responses_call", "sync_responses_call"],
)
def test_closing_the_sse_stream_mid_response_closes_the_provider_response(
    provider_stream: _EndlessProviderStream | _EndlessSyncProviderStream,
    upstream_class: type[ResponsesAPIStreamingIterator] | type[SyncResponsesAPIStreamingIterator],
):
    async def _close_after_the_first_text_delta() -> bool:
        upstream: Final = upstream_class(
            response=httpx.Response(200, stream=provider_stream),
            model="m",
            responses_api_provider_config=OpenAIResponsesAPIConfig(),
            logging_obj=_logging_obj_that_records_failures(_FailureRecorder()),
        )
        sse: Final = AnthropicResponsesStreamWrapper(responses_stream=upstream, model="m").async_anthropic_sse_wrapper()
        async for frame in sse:
            if b'"text_delta"' in frame:
                await sse.aclose()
                return provider_stream.closed
        pytest.fail("the stream ended without a text delta")

    assert asyncio.run(_close_after_the_first_text_delta())


def test_a_close_refused_while_another_task_reads_leaves_the_provider_response_to_that_read():
    async def _close_from_another_task_mid_read() -> tuple[bool, bytes]:
        provider_stream: Final = _ProviderStreamThatPausesAfterTheFirstTextDelta()
        upstream: Final = ResponsesAPIStreamingIterator(
            response=httpx.Response(200, stream=provider_stream),
            model="m",
            responses_api_provider_config=OpenAIResponsesAPIConfig(),
            logging_obj=_logging_obj_that_records_failures(_FailureRecorder()),
        )
        sse: Final = AnthropicResponsesStreamWrapper(responses_stream=upstream, model="m").async_anthropic_sse_wrapper()
        text_deltas: Final = (frame async for frame in sse if b'"text_delta"' in frame)
        await text_deltas.__anext__()
        second_text_delta: Final = asyncio.ensure_future(text_deltas.__anext__())
        await provider_stream.paused.wait()
        with pytest.raises(RuntimeError, match="already running"):
            await sse.aclose()
        closed_by_the_refused_close: Final = provider_stream.closed
        provider_stream.resume.set()
        return closed_by_the_refused_close, await second_text_delta

    closed_by_the_refused_close, second_frame = asyncio.run(_close_from_another_task_mid_read())
    assert not closed_by_the_refused_close
    assert b'"text_delta"' in second_frame


class _SyncProviderStreamThatGoesQuietAfterTheFirstTextDelta(httpx.SyncByteStream):
    def __init__(self) -> None:
        self.closed = False
        self.quiet = threading.Event()
        self.next_bytes = threading.Event()
        self.sent_next_bytes = False

    def __iter__(self) -> Iterator[bytes]:
        yield _RESPONSE_CREATED
        yield _MESSAGE_ADDED
        yield _TEXT_DELTA
        self.quiet.set()
        self.sent_next_bytes = self.next_bytes.wait(timeout=5)
        if self.closed:
            raise httpx.ReadError("[Errno 9] Bad file descriptor")
        yield _TEXT_DELTA

    def close(self) -> None:
        self.closed = True


async def _close_after_cancelling_a_quiet_read(
    provider_stream: _SyncProviderStreamThatGoesQuietAfterTheFirstTextDelta,
    logging_obj: LitellmLogging,
    sync_result: bool,
) -> None:
    upstream: Final = SyncResponsesAPIStreamingIterator(
        response=httpx.Response(200, stream=provider_stream),
        model="m",
        responses_api_provider_config=OpenAIResponsesAPIConfig(),
        logging_obj=logging_obj,
    )
    wrapper: Final = AnthropicResponsesStreamWrapper(responses_stream=upstream, model="m")
    sse: Final = wrapper.anthropic_sse_wrapper() if sync_result else wrapper.async_anthropic_sse_wrapper()
    while b'"text_delta"' not in await sse.__anext__():
        pass
    quiet_read: Final = asyncio.ensure_future(sse.__anext__())
    assert await asyncio.to_thread(provider_stream.quiet.wait, 5)
    _ = quiet_read.cancel()
    with pytest.raises(asyncio.CancelledError):
        await quiet_read
    await sse.aclose()


@pytest.mark.parametrize("sync_result", [False, True])
def test_closing_a_sync_stream_after_a_cancelled_read_closes_the_provider_response_once_that_read_returns(
    sync_result: bool,
):
    provider_stream: Final = _SyncProviderStreamThatGoesQuietAfterTheFirstTextDelta()
    failure_recorder: Final = _FailureRecorder()

    async def _give_up_while_the_provider_is_quiet() -> None:
        await _close_after_cancelling_a_quiet_read(
            provider_stream, _logging_obj_that_records_failures(failure_recorder), sync_result
        )
        provider_stream.next_bytes.set()

    asyncio.run(_give_up_while_the_provider_is_quiet())
    assert provider_stream.sent_next_bytes
    assert provider_stream.closed
    assert failure_recorder.failures == ()


@pytest.mark.parametrize("sync_result", [False, True])
def test_closing_a_sync_stream_during_a_quiet_read_leaves_the_other_worker_threads_free(sync_result: bool):
    provider_stream: Final = _SyncProviderStreamThatGoesQuietAfterTheFirstTextDelta()

    async def _run_other_work_while_the_provider_is_quiet() -> int:
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=2))
        try:
            await _close_after_cancelling_a_quiet_read(
                provider_stream, _logging_obj_that_records_failures(_FailureRecorder()), sync_result
            )
            return await asyncio.wait_for(asyncio.to_thread(threading.get_ident), timeout=2)
        finally:
            provider_stream.next_bytes.set()

    _ = asyncio.run(_run_other_work_while_the_provider_is_quiet())
    assert provider_stream.closed


_RESPONSE_COMPLETED: Final = (
    b'data: {"type": "response.completed", "response": {"id": "resp_1", "status": "completed", "output": []}}\n\n'
)
_PARALLEL_CALLS: Final = (
    ("fc_a", "call_98145f1b9bdb40fc82c5b2415ed291a5", "read_file", '{"path": "notes/a.md"}'),
    ("fc_b", "call_0c8a1d2e3f4b5a6978695a4b3c2d1e0f", "run_command", '{"cmd": ["ls", "-la"]}'),
)


@dataclass(frozen=True, slots=True)
class _ToolCallStreamShape:
    argument_deltas: tuple[str, ...] = ()
    deltas_carry_item_id: bool = True
    sends_arguments_done: bool = True
    output_item_done_carries_arguments: bool = True


def _halves(arguments: str) -> tuple[str, ...]:
    return (arguments[: len(arguments) // 2], arguments[len(arguments) // 2 :])


class _ToolCallProviderStream(httpx.AsyncByteStream):
    def __init__(
        self,
        shape: _ToolCallStreamShape,
        split_arguments: Callable[[str], tuple[str, ...]],
        created: bytes = _RESPONSE_CREATED,
        hangs_before_completing: bool = False,
    ) -> None:
        self._shape: Final = shape
        self._split_arguments: Final = split_arguments
        self._created: Final = created
        self._hangs_before_completing: Final = hangs_before_completing

    def call_events(
        self, output_index: int, item_id: str, call_id: str, name: str, arguments: str
    ) -> Iterator[Mapping[str, object]]:
        item: Final = {"type": "function_call", "id": item_id, "call_id": call_id, "name": name}
        yield {
            "type": "response.output_item.added",
            "output_index": output_index,
            "item": {**item, "arguments": "", "status": "in_progress"},
        }
        for piece in self._split_arguments(arguments):
            yield {
                "type": "response.function_call_arguments.delta",
                "output_index": output_index,
                "delta": piece,
                **({"item_id": item_id} if self._shape.deltas_carry_item_id else {}),
            }
        if self._shape.sends_arguments_done:
            yield {
                "type": "response.function_call_arguments.done",
                "output_index": output_index,
                "item_id": item_id,
                "arguments": arguments,
            }
        yield {
            "type": "response.output_item.done",
            "output_index": output_index,
            "item": {
                **item,
                "arguments": arguments if self._shape.output_item_done_carries_arguments else "",
                "status": "completed",
            },
        }

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._created
        for output_index, call in enumerate(_PARALLEL_CALLS):
            for event in self.call_events(output_index, *call):
                yield f"data: {json.dumps(event)}\n\n".encode()
        if self._hangs_before_completing:
            await asyncio.Event().wait()
        yield _RESPONSE_COMPLETED


def _streamed_tool_uses(provider_stream: _ToolCallProviderStream) -> tuple[tuple[str, str, str | None], ...]:
    async def _run() -> list[Mapping[str, object]]:
        upstream: Final = ResponsesAPIStreamingIterator(
            response=httpx.Response(200, stream=provider_stream),
            model="m",
            responses_api_provider_config=OpenAIResponsesAPIConfig(),
            logging_obj=_logging_obj_that_records_failures(_FailureRecorder()),
        )
        return [chunk async for chunk in AnthropicResponsesStreamWrapper(responses_stream=upstream, model="m")]

    events: Final = tuple(
        _BLOCK_EVENT.validate_python(chunk) for chunk in asyncio.run(_run()) if chunk["type"] in _BLOCK_EVENT_TYPES
    )
    return tuple(
        (event.content_block.id, event.content_block.name, _input_before_block_stop(events, event.index))
        for event in events
        if isinstance(event, _BlockStart)
    )


class _ToolUseBlock(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    name: str


class _BlockStart(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["content_block_start"]
    index: int
    content_block: _ToolUseBlock


class _InputJsonDelta(BaseModel):
    model_config = ConfigDict(frozen=True)
    partial_json: str


class _BlockDelta(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["content_block_delta"]
    index: int
    delta: _InputJsonDelta


class _BlockStop(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["content_block_stop"]
    index: int


_BlockEvent = Annotated[_BlockStart | _BlockDelta | _BlockStop, Field(discriminator="type")]
_BLOCK_EVENT: Final[TypeAdapter[_BlockEvent]] = TypeAdapter(_BlockEvent)
_BLOCK_EVENT_TYPES: Final = frozenset({"content_block_start", "content_block_delta", "content_block_stop"})


def _input_before_block_stop(events: tuple[_BlockEvent, ...], index: int) -> str | None:
    stop: Final = next(
        (pos for pos, event in enumerate(events) if isinstance(event, _BlockStop) and event.index == index), None
    )
    if stop is None:
        return None
    return "".join(
        event.delta.partial_json for event in events[:stop] if isinstance(event, _BlockDelta) and event.index == index
    )


def _no_deltas(arguments: str) -> tuple[str, ...]:
    return ()


def _one_empty_delta(arguments: str) -> tuple[str, ...]:
    return ("",)


@pytest.mark.parametrize(
    ("shape", "split_arguments"),
    [
        (_ToolCallStreamShape(), _no_deltas),
        (_ToolCallStreamShape(sends_arguments_done=False), _no_deltas),
        (_ToolCallStreamShape(output_item_done_carries_arguments=False), _no_deltas),
        (_ToolCallStreamShape(output_item_done_carries_arguments=False), _one_empty_delta),
        (_ToolCallStreamShape(), _halves),
        (_ToolCallStreamShape(sends_arguments_done=False), _halves),
        (_ToolCallStreamShape(deltas_carry_item_id=False), _halves),
    ],
    ids=[
        "done_events_only",
        "output_item_done_only",
        "arguments_done_only",
        "empty_delta_then_arguments_done",
        "deltas_then_done_events",
        "deltas_then_output_item_done",
        "deltas_without_item_id",
    ],
)
def test_each_parallel_tool_use_streams_its_arguments_exactly_once(
    shape: _ToolCallStreamShape, split_arguments: Callable[[str], tuple[str, ...]]
):
    tool_uses: Final = _streamed_tool_uses(_ToolCallProviderStream(shape, split_arguments))

    assert tool_uses == tuple((call_id, name, arguments) for _, call_id, name, arguments in _PARALLEL_CALLS)


_CREATED_TOOL_RESPONSE_EVENT: Final[Mapping[str, object]] = {
    "type": "response.created",
    "sequence_number": 0,
    "response": {
        "id": "resp_tools",
        "object": "response",
        "created_at": 1700000000.0,
        "model": "gpt-5.4-nano",
        "status": "in_progress",
        "output": [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    },
}


async def _read_until_blocks_close(sse: AnthropicResponsesSSEByteStream, open_blocks: int) -> None:
    if open_blocks == 0:
        return
    frame: Final = await sse.__anext__()
    await _read_until_blocks_close(sse, open_blocks - (b'"content_block_stop"' in frame))


def _output_tokens_billed_on_disconnect(
    shape: _ToolCallStreamShape, split_arguments: Callable[[str], tuple[str, ...]]
) -> int | None:
    async def _disconnect_after_the_tool_calls() -> int | None:
        upstream: Final = ResponsesAPIStreamingIterator(
            response=httpx.Response(
                200,
                stream=_ToolCallProviderStream(
                    shape,
                    split_arguments,
                    created=f"data: {json.dumps(_CREATED_TOOL_RESPONSE_EVENT)}\n\n".encode(),
                    hangs_before_completing=True,
                ),
            ),
            model="gpt-5.4-nano",
            responses_api_provider_config=OpenAIResponsesAPIConfig(),
            logging_obj=_logging_obj_that_records_failures(_FailureRecorder()),
            request_data={"model": "gpt-5.4-nano", "input": "read the notes and list the files"},
            custom_llm_provider="openai",
        )
        sse: Final = AnthropicResponsesStreamWrapper(responses_stream=upstream, model="m").async_anthropic_sse_wrapper()
        await _read_until_blocks_close(sse, len(_PARALLEL_CALLS))
        partial: Final = sse.partial_response_on_disconnect()
        await sse.aclose()
        return partial.response.usage.output_tokens if partial is not None and partial.response.usage else None

    return asyncio.run(_disconnect_after_the_tool_calls())


@pytest.mark.parametrize(
    "shape",
    [
        _ToolCallStreamShape(),
        _ToolCallStreamShape(sends_arguments_done=False),
        _ToolCallStreamShape(output_item_done_carries_arguments=False),
    ],
    ids=["done_events_only", "output_item_done_only", "arguments_done_only"],
)
def test_a_disconnect_after_done_only_tool_calls_bills_the_arguments_the_client_received(shape: _ToolCallStreamShape):
    billed_for_done_only: Final = _output_tokens_billed_on_disconnect(shape, _no_deltas)
    billed_for_deltas: Final = _output_tokens_billed_on_disconnect(_ToolCallStreamShape(), _halves)

    assert billed_for_deltas
    assert billed_for_done_only == billed_for_deltas


class TestClientToolSearchStream:
    @staticmethod
    def _search_turn() -> list[object]:
        done_item: Final = {
            "type": "tool_search_call",
            "id": "tsc_1",
            "call_id": "call_search",
            "execution": "client",
            "arguments": {"query": "weather forecast"},
            "status": "completed",
        }
        return [
            {"type": "response.created"},
            {
                "type": "response.output_item.added",
                "item": {
                    "type": "tool_search_call",
                    "id": "tsc_1",
                    "call_id": "call_search",
                    "execution": "client",
                    "arguments": {},
                    "status": "in_progress",
                },
            },
            {"type": "response.output_item.done", "item": done_item},
            {"type": "response.completed", "response": SimpleNamespace(output=[done_item], status="completed")},
        ]

    @staticmethod
    def _drain(events: list[object], tool_search_name: str | None) -> list:
        async def _gen():
            for event in events:
                yield event

        async def _run() -> list:
            wrapper = AnthropicResponsesStreamWrapper(
                responses_stream=_gen(), model="m", tool_search_name=tool_search_name
            )
            return [chunk async for chunk in wrapper]

        return asyncio.run(_run())

    def test_search_call_streams_as_one_search_tool_use_and_stops_for_the_tool(self):
        chunks = self._drain(self._search_turn(), "ToolSearch")

        assert [c["content_block"] for c in chunks if c["type"] == "content_block_start"] == [
            {"type": "tool_use", "id": "call_search", "name": "ToolSearch", "input": {}}
        ]
        streamed_input = "".join(
            c["delta"]["partial_json"]
            for c in chunks
            if c["type"] == "content_block_delta" and c["delta"]["type"] == "input_json_delta"
        )
        assert json.loads(streamed_input) == {"query": "weather forecast"}
        assert [c["type"] for c in chunks if c["type"] == "content_block_stop"] == ["content_block_stop"]
        assert [c["delta"]["stop_reason"] for c in chunks if c["type"] == "message_delta"] == ["tool_use"]

    def test_search_call_is_not_a_tool_use_without_a_client_search_tool(self):
        chunks = self._drain(self._search_turn(), None)

        assert not [c for c in chunks if c["type"] == "content_block_start"]
        assert [c["delta"]["stop_reason"] for c in chunks if c["type"] == "message_delta"] == ["end_turn"]
