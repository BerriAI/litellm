"""
Tests for AgenticAnthropicStreamingIterator and SSE rebuild helpers.
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.anthropic.experimental_pass_through.messages.agentic_streaming_iterator import (
    AgenticAnthropicStreamingIterator,
    _handle_content_block_delta,
    _handle_content_block_start,
    _handle_content_block_stop,
    _handle_message_delta,
    _handle_message_start,
    _parse_sse_events,
)


# ---------------------------------------------------------------------------
# Helpers to build SSE byte payloads
# ---------------------------------------------------------------------------


def _sse_event(event_type: str, data: dict) -> bytes:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()


def _build_simple_text_stream() -> List[bytes]:
    """Produce SSE bytes for a simple text response (no tool calls)."""
    chunks = []
    chunks.append(
        _sse_event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_123",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-20250514",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                },
            },
        )
    )
    chunks.append(
        _sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        )
    )
    chunks.append(
        _sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hello, world!"},
            },
        )
    )
    chunks.append(
        _sse_event("content_block_stop", {"type": "content_block_stop", "index": 0})
    )
    chunks.append(
        _sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 5},
            },
        )
    )
    chunks.append(_sse_event("message_stop", {"type": "message_stop"}))
    return chunks


def _build_tool_use_stream() -> List[bytes]:
    """Produce SSE bytes for a response with a tool_use block."""
    chunks = []
    chunks.append(
        _sse_event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_tool_456",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-20250514",
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 50, "output_tokens": 0},
                },
            },
        )
    )
    # thinking block
    chunks.append(
        _sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "thinking",
                    "thinking": "",
                    "signature": "",
                },
            },
        )
    )
    chunks.append(
        _sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "thinking_delta",
                    "thinking": "I need to retrieve...",
                },
            },
        )
    )
    chunks.append(
        _sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "sig_abc"},
            },
        )
    )
    chunks.append(
        _sse_event("content_block_stop", {"type": "content_block_stop", "index": 0})
    )
    # tool_use block
    chunks.append(
        _sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_001",
                    "name": "litellm_content_retrieve",
                    "input": {},
                },
            },
        )
    )
    chunks.append(
        _sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"key": "section_',
                },
            },
        )
    )
    chunks.append(
        _sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '1"}'},
            },
        )
    )
    chunks.append(
        _sse_event("content_block_stop", {"type": "content_block_stop", "index": 1})
    )
    chunks.append(
        _sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 20},
            },
        )
    )
    chunks.append(_sse_event("message_stop", {"type": "message_stop"}))
    return chunks


# ---------------------------------------------------------------------------
# Mock async stream
# ---------------------------------------------------------------------------


class MockAsyncStream:
    """Async iterator that yields a list of byte chunks."""

    def __init__(self, chunks: List[bytes]):
        self._chunks = list(chunks)
        self._idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        if self._idx >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._idx]
        self._idx += 1
        return chunk


# ---------------------------------------------------------------------------
# Tests for _parse_sse_events
# ---------------------------------------------------------------------------


class TestParseSSEEvents:
    def test_should_parse_single_event(self):
        raw = _sse_event(
            "message_start", {"type": "message_start", "message": {"id": "1"}}
        )
        events = _parse_sse_events(raw)
        assert len(events) == 1
        assert events[0][0] == "message_start"
        assert events[0][1]["message"]["id"] == "1"

    def test_should_parse_multiple_events(self):
        raw = b"".join(_build_simple_text_stream())
        events = _parse_sse_events(raw)
        event_types = [e[0] for e in events]
        assert "message_start" in event_types
        assert "content_block_start" in event_types
        assert "content_block_delta" in event_types
        assert "content_block_stop" in event_types
        assert "message_delta" in event_types
        assert "message_stop" in event_types

    def test_should_skip_malformed_json(self):
        raw = b"event: message_start\ndata: {invalid json}\n\n"
        events = _parse_sse_events(raw)
        assert len(events) == 0

    def test_should_handle_empty_bytes(self):
        events = _parse_sse_events(b"")
        assert events == []


# ---------------------------------------------------------------------------
# Tests for _handle_* helpers
# ---------------------------------------------------------------------------


class TestHandleMessageStart:
    def test_should_populate_envelope(self):
        response: Dict[str, Any] = {
            "id": "",
            "model": "",
            "role": "assistant",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        data = {
            "message": {
                "id": "msg_abc",
                "model": "claude-sonnet-4-20250514",
                "role": "assistant",
                "usage": {
                    "input_tokens": 42,
                    "cache_creation_input_tokens": 100,
                },
            }
        }
        _handle_message_start(data, response)
        assert response["id"] == "msg_abc"
        assert response["model"] == "claude-sonnet-4-20250514"
        assert response["usage"]["input_tokens"] == 42
        assert response["usage"]["cache_creation_input_tokens"] == 100


class TestHandleContentBlockStart:
    def test_should_create_text_block(self):
        blocks: Dict[int, Dict] = {}
        data = {"index": 0, "content_block": {"type": "text", "text": ""}}
        _handle_content_block_start(data, blocks)
        assert blocks[0] == {"type": "text", "text": ""}

    def test_should_create_tool_use_block(self):
        blocks: Dict[int, Dict] = {}
        data = {
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_x",
                "name": "my_tool",
                "input": {},
            },
        }
        _handle_content_block_start(data, blocks)
        assert blocks[1]["type"] == "tool_use"
        assert blocks[1]["name"] == "my_tool"
        assert blocks[1]["_partial_json"] == ""

    def test_should_create_thinking_block(self):
        blocks: Dict[int, Dict] = {}
        data = {
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        }
        _handle_content_block_start(data, blocks)
        assert blocks[0]["type"] == "thinking"


class TestHandleContentBlockDelta:
    def test_should_accumulate_text(self):
        blocks = {0: {"type": "text", "text": "Hello"}}
        _handle_content_block_delta(
            {"index": 0, "delta": {"type": "text_delta", "text": " World"}},
            blocks,
        )
        assert blocks[0]["text"] == "Hello World"

    def test_should_accumulate_json(self):
        blocks = {0: {"type": "tool_use", "_partial_json": '{"key":'}}
        _handle_content_block_delta(
            {
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '"val"}'},
            },
            blocks,
        )
        assert blocks[0]["_partial_json"] == '{"key":"val"}'

    def test_should_ignore_missing_block(self):
        blocks: Dict[int, Dict] = {}
        _handle_content_block_delta(
            {"index": 99, "delta": {"type": "text_delta", "text": "x"}},
            blocks,
        )
        assert 99 not in blocks


class TestHandleContentBlockStop:
    def test_should_parse_tool_input_json(self):
        blocks = {
            0: {
                "type": "tool_use",
                "input": {},
                "_partial_json": '{"key": "section_1"}',
            }
        }
        _handle_content_block_stop({"index": 0}, blocks)
        assert blocks[0]["input"] == {"key": "section_1"}
        assert "_partial_json" not in blocks[0]

    def test_should_handle_invalid_json_gracefully(self):
        blocks = {
            0: {
                "type": "tool_use",
                "input": {},
                "_partial_json": "not valid json",
            }
        }
        _handle_content_block_stop({"index": 0}, blocks)
        assert blocks[0]["input"] == {"_raw": "not valid json"}


class TestHandleMessageDelta:
    def test_should_set_stop_reason_and_usage(self):
        response: Dict[str, Any] = {
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        _handle_message_delta(
            {
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 15},
            },
            response,
        )
        assert response["stop_reason"] == "end_turn"
        assert response["usage"]["output_tokens"] == 15


# ---------------------------------------------------------------------------
# Tests for _rebuild_anthropic_response_from_sse
# ---------------------------------------------------------------------------


class TestRebuildAnthropicResponse:
    def test_should_rebuild_simple_text_response(self):
        raw_bytes = _build_simple_text_stream()
        result = AgenticAnthropicStreamingIterator._rebuild_anthropic_response_from_sse(
            raw_bytes
        )
        assert result is not None
        assert result["id"] == "msg_123"
        assert result["model"] == "claude-sonnet-4-20250514"
        assert result["stop_reason"] == "end_turn"
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Hello, world!"
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 5

    def test_should_rebuild_tool_use_response(self):
        raw_bytes = _build_tool_use_stream()
        result = AgenticAnthropicStreamingIterator._rebuild_anthropic_response_from_sse(
            raw_bytes
        )
        assert result is not None
        assert result["id"] == "msg_tool_456"
        assert result["stop_reason"] == "tool_use"
        assert len(result["content"]) == 2

        thinking = result["content"][0]
        assert thinking["type"] == "thinking"
        assert thinking["thinking"] == "I need to retrieve..."
        assert thinking["signature"] == "sig_abc"

        tool = result["content"][1]
        assert tool["type"] == "tool_use"
        assert tool["id"] == "toolu_001"
        assert tool["name"] == "litellm_content_retrieve"
        assert tool["input"] == {"key": "section_1"}

    def test_should_return_none_without_message_start(self):
        raw_bytes = [
            _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text"},
                },
            )
        ]
        result = AgenticAnthropicStreamingIterator._rebuild_anthropic_response_from_sse(
            raw_bytes
        )
        assert result is None

    def test_should_handle_empty_bytes(self):
        result = AgenticAnthropicStreamingIterator._rebuild_anthropic_response_from_sse(
            []
        )
        assert result is None

    def test_should_handle_multi_event_chunks(self):
        """When multiple SSE events arrive in a single bytes chunk."""
        combined = b"".join(_build_simple_text_stream())
        result = AgenticAnthropicStreamingIterator._rebuild_anthropic_response_from_sse(
            [combined]
        )
        assert result is not None
        assert result["content"][0]["text"] == "Hello, world!"

    def test_should_preserve_cache_usage_fields(self):
        raw_bytes = [
            _sse_event(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_cache",
                        "model": "claude-sonnet-4-20250514",
                        "role": "assistant",
                        "usage": {
                            "input_tokens": 100,
                            "cache_creation_input_tokens": 50,
                            "cache_read_input_tokens": 30,
                        },
                    },
                },
            ),
            _sse_event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 10},
                },
            ),
            _sse_event("message_stop", {"type": "message_stop"}),
        ]
        result = AgenticAnthropicStreamingIterator._rebuild_anthropic_response_from_sse(
            raw_bytes
        )
        assert result is not None
        assert result["usage"]["cache_creation_input_tokens"] == 50
        assert result["usage"]["cache_read_input_tokens"] == 30

    def test_should_handle_redacted_thinking_block(self):
        raw_bytes = [
            _sse_event(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_redact",
                        "model": "claude-sonnet-4-20250514",
                        "role": "assistant",
                        "usage": {"input_tokens": 5},
                    },
                },
            ),
            _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "redacted_thinking", "data": "abc123"},
                },
            ),
            _sse_event(
                "content_block_stop",
                {"type": "content_block_stop", "index": 0},
            ),
            _sse_event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 1},
                },
            ),
            _sse_event("message_stop", {"type": "message_stop"}),
        ]
        result = AgenticAnthropicStreamingIterator._rebuild_anthropic_response_from_sse(
            raw_bytes
        )
        assert result is not None
        assert result["content"][0]["type"] == "redacted_thinking"


# ---------------------------------------------------------------------------
# Tests for AgenticAnthropicStreamingIterator (Phase 1 / Phase 2)
# ---------------------------------------------------------------------------


class TestAgenticStreamingIteratorPhase1:
    @pytest.mark.asyncio
    async def test_should_yield_all_chunks_when_no_hook_fires(self):
        """When hooks return None, the wrapper should yield all original chunks."""
        chunks = _build_simple_text_stream()
        mock_stream = MockAsyncStream(chunks)

        mock_handler = MagicMock()
        mock_handler._call_agentic_completion_hooks = AsyncMock(return_value=None)

        iterator = AgenticAnthropicStreamingIterator(
            completion_stream=mock_stream,
            http_handler=mock_handler,
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "hi"}],
            anthropic_messages_provider_config=MagicMock(),
            anthropic_messages_optional_request_params={},
            logging_obj=MagicMock(),
            custom_llm_provider="anthropic",
            kwargs={},
        )

        collected = []
        async for chunk in iterator:
            collected.append(chunk)

        assert len(collected) == len(chunks)
        for orig, got in zip(chunks, collected):
            assert orig == got

        mock_handler._call_agentic_completion_hooks.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_should_pass_rebuilt_response_to_hooks(self):
        """The rebuilt dict passed to hooks should match the original stream content."""
        chunks = _build_tool_use_stream()
        mock_stream = MockAsyncStream(chunks)

        captured_response = {}

        async def mock_hooks(**kwargs):
            captured_response.update(kwargs["response"])
            return None

        mock_handler = MagicMock()
        mock_handler._call_agentic_completion_hooks = mock_hooks

        iterator = AgenticAnthropicStreamingIterator(
            completion_stream=mock_stream,
            http_handler=mock_handler,
            model="claude-sonnet-4-20250514",
            messages=[],
            anthropic_messages_provider_config=MagicMock(),
            anthropic_messages_optional_request_params={},
            logging_obj=MagicMock(),
            custom_llm_provider="anthropic",
            kwargs={},
        )

        async for _ in iterator:
            pass

        assert captured_response["id"] == "msg_tool_456"
        assert captured_response["stop_reason"] == "tool_use"
        assert captured_response["content"][1]["name"] == "litellm_content_retrieve"


class TestAgenticStreamingIteratorPhase2:
    @pytest.mark.asyncio
    async def test_should_chain_follow_up_async_iterator(self):
        """When hooks return an async iterator, Phase 2 should yield from it."""
        phase1_chunks = _build_simple_text_stream()
        phase2_chunks = [b"follow-up-chunk-1", b"follow-up-chunk-2"]

        mock_stream = MockAsyncStream(phase1_chunks)
        follow_up = MockAsyncStream(phase2_chunks)

        mock_handler = MagicMock()
        mock_handler._call_agentic_completion_hooks = AsyncMock(return_value=follow_up)

        iterator = AgenticAnthropicStreamingIterator(
            completion_stream=mock_stream,
            http_handler=mock_handler,
            model="claude-sonnet-4-20250514",
            messages=[],
            anthropic_messages_provider_config=MagicMock(),
            anthropic_messages_optional_request_params={},
            logging_obj=MagicMock(),
            custom_llm_provider="anthropic",
            kwargs={},
        )

        collected = []
        async for chunk in iterator:
            collected.append(chunk)

        assert len(collected) == len(phase1_chunks) + len(phase2_chunks)
        assert collected[-2:] == phase2_chunks

    @pytest.mark.asyncio
    async def test_should_convert_dict_response_to_fake_stream(self):
        """When hooks return a dict, it should be wrapped in FakeAnthropicMessagesStreamIterator."""
        phase1_chunks = _build_simple_text_stream()
        mock_stream = MockAsyncStream(phase1_chunks)

        fake_response = {
            "id": "msg_followup",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-20250514",
            "content": [{"type": "text", "text": "follow-up answer"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 100, "output_tokens": 20},
        }

        mock_handler = MagicMock()
        mock_handler._call_agentic_completion_hooks = AsyncMock(
            return_value=fake_response
        )

        iterator = AgenticAnthropicStreamingIterator(
            completion_stream=mock_stream,
            http_handler=mock_handler,
            model="claude-sonnet-4-20250514",
            messages=[],
            anthropic_messages_provider_config=MagicMock(),
            anthropic_messages_optional_request_params={},
            logging_obj=MagicMock(),
            custom_llm_provider="anthropic",
            kwargs={},
        )

        collected = []
        async for chunk in iterator:
            collected.append(chunk)

        # Phase 1 chunks + Phase 2 fake-stream chunks
        assert len(collected) > len(phase1_chunks)
        # The follow-up chunks should contain the text from the dict response
        phase2_bytes = b"".join(collected[len(phase1_chunks) :])
        assert b"follow-up answer" in phase2_bytes


class TestAgenticStreamingIteratorErrorHandling:
    @pytest.mark.asyncio
    async def test_should_swallow_hook_errors(self):
        """Errors in hook processing should be swallowed; Phase 1 chunks are still yielded."""
        chunks = _build_simple_text_stream()
        mock_stream = MockAsyncStream(chunks)

        mock_handler = MagicMock()
        mock_handler._call_agentic_completion_hooks = AsyncMock(
            side_effect=RuntimeError("hook exploded")
        )

        mock_logging = MagicMock()
        mock_logging.litellm_call_id = "test_call_123"

        iterator = AgenticAnthropicStreamingIterator(
            completion_stream=mock_stream,
            http_handler=mock_handler,
            model="claude-sonnet-4-20250514",
            messages=[],
            anthropic_messages_provider_config=MagicMock(),
            anthropic_messages_optional_request_params={},
            logging_obj=mock_logging,
            custom_llm_provider="anthropic",
            kwargs={},
        )

        collected = []
        async for chunk in iterator:
            collected.append(chunk)

        # All Phase 1 chunks should still have been yielded
        assert len(collected) == len(chunks)

    @pytest.mark.asyncio
    async def test_should_handle_empty_stream(self):
        """An empty upstream stream should not crash."""
        mock_stream = MockAsyncStream([])

        mock_handler = MagicMock()
        mock_handler._call_agentic_completion_hooks = AsyncMock(return_value=None)

        iterator = AgenticAnthropicStreamingIterator(
            completion_stream=mock_stream,
            http_handler=mock_handler,
            model="claude-sonnet-4-20250514",
            messages=[],
            anthropic_messages_provider_config=MagicMock(),
            anthropic_messages_optional_request_params={},
            logging_obj=MagicMock(),
            custom_llm_provider="anthropic",
            kwargs={},
        )

        collected = []
        async for chunk in iterator:
            collected.append(chunk)

        assert collected == []
        # hooks should not be called since no bytes were collected
        mock_handler._call_agentic_completion_hooks.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_should_pass_stream_true_to_hooks(self):
        """The wrapper should always pass stream=True to hooks."""
        chunks = _build_simple_text_stream()
        mock_stream = MockAsyncStream(chunks)

        mock_handler = MagicMock()
        mock_handler._call_agentic_completion_hooks = AsyncMock(return_value=None)

        iterator = AgenticAnthropicStreamingIterator(
            completion_stream=mock_stream,
            http_handler=mock_handler,
            model="claude-sonnet-4-20250514",
            messages=[],
            anthropic_messages_provider_config=MagicMock(),
            anthropic_messages_optional_request_params={},
            logging_obj=MagicMock(),
            custom_llm_provider="anthropic",
            kwargs={},
        )

        async for _ in iterator:
            pass

        call_kwargs = mock_handler._call_agentic_completion_hooks.call_args
        assert call_kwargs.kwargs["stream"] is True
