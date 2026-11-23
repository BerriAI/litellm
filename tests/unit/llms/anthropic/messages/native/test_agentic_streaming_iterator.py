"""
Tests for AgenticAnthropicStreamingIterator and SSE rebuild helpers.
"""

import asyncio
import functools
import json
import selectors
from typing import Any, AsyncGenerator, Callable, Coroutine, Dict, Final, List, Optional, ParamSpec, Tuple, TypeVar
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm
from litellm.constants import STREAM_SSE_KEEPALIVE_PING_BYTES
from litellm.llms.anthropic.messages.native.agentic_streaming_iterator import (
    SERVER_FULFILLED_TOOL_LEAK_ERROR_SSE_BYTES,
    AgenticAnthropicStreamingIterator,
    _handle_content_block_delta,
    _handle_content_block_start,
    _handle_content_block_stop,
    _handle_message_delta,
    _handle_message_start,
    _parse_sse_events,
)
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.utils import ProxyLogging


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
    chunks.append(_sse_event("content_block_stop", {"type": "content_block_stop", "index": 0}))
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


def _sse_frames(wire: str) -> list[tuple[str, str]]:
    return [
        (frame.split("\n")[0].removeprefix("event: "), frame.split("data: ", 1)[1])
        for frame in wire.split("\n\n")
        if frame.startswith("event: ") and "data: " in frame
    ]


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
    chunks.append(_sse_event("content_block_stop", {"type": "content_block_stop", "index": 0}))
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
    chunks.append(_sse_event("content_block_stop", {"type": "content_block_stop", "index": 1}))
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


class MockSlowAsyncStream(MockAsyncStream):
    """Async iterator that sleeps before every chunk."""

    def __init__(self, chunks: List[bytes], delay_seconds: float):
        super().__init__(chunks)
        self._delay_seconds = delay_seconds

    async def __anext__(self) -> bytes:
        await asyncio.sleep(self._delay_seconds)
        return await super().__anext__()


class _IdleSkippingSelector(selectors.DefaultSelector):
    def __init__(self) -> None:
        super().__init__()
        self.skipped_seconds = 0.0

    def select(self, timeout: Optional[float] = None) -> List[Tuple[selectors.SelectorKey, int]]:
        ready = super().select(0)
        if ready or timeout == 0:
            return ready
        if timeout is None:
            raise AssertionError("every task is waiting and no timer is left to fire")
        self.skipped_seconds += timeout
        return ready


class _VirtualClockLoop(asyncio.SelectorEventLoop):
    def __init__(self) -> None:
        self._idle_skipping_selector = _IdleSkippingSelector()
        super().__init__(self._idle_skipping_selector)

    def time(self) -> float:
        return super().time() + self._idle_skipping_selector.skipped_seconds


_P = ParamSpec("_P")
_T = TypeVar("_T")


def _on_virtual_clock(test: Callable[_P, Coroutine[object, object, _T]]) -> Callable[_P, _T]:
    @functools.wraps(test)
    def run(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        loop: Final = _VirtualClockLoop()
        try:
            return loop.run_until_complete(test(*args, **kwargs))
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    return run


class MockFailingAsyncStream(MockAsyncStream):
    """Async iterator that raises after yielding its chunks."""

    def __init__(self, chunks: List[bytes], error: Exception):
        super().__init__(chunks)
        self._error = error

    async def __anext__(self) -> bytes:
        if self._idx >= len(self._chunks):
            raise self._error
        return await super().__anext__()


PING_INTERVAL_SECONDS = 15.0


def _build_hold_back_iterator(
    stream: MockAsyncStream,
    mock_handler: MagicMock,
    ping_interval_seconds: float = PING_INTERVAL_SECONDS,
    server_fulfilled_tool_names: frozenset = frozenset({"litellm_content_retrieve"}),
) -> AgenticAnthropicStreamingIterator:
    return AgenticAnthropicStreamingIterator(
        completion_stream=stream,
        http_handler=mock_handler,
        model="claude-sonnet-4-20250514",
        messages=[],
        anthropic_messages_provider_config=MagicMock(),
        anthropic_messages_optional_request_params={},
        logging_obj=MagicMock(),
        custom_llm_provider="anthropic",
        kwargs={},
        hold_back=True,
        server_fulfilled_tool_names=server_fulfilled_tool_names,
        ping_interval_seconds=ping_interval_seconds,
    )


def _compact_sse_event(event_type: str, data: dict) -> bytes:
    return f"event: {event_type}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode()


def _text_then_tool_use_stream(tool_name: str = "litellm_content_retrieve") -> List[bytes]:
    return [
        _sse_event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_live",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-20250514",
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 30, "output_tokens": 0},
                },
            },
        ),
        _sse_event(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        _sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Let me look that up."},
            },
        ),
        _sse_event("content_block_stop", {"type": "content_block_stop", "index": 0}),
        _sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "tool_use", "id": "toolu_1", "name": tool_name, "input": {}},
            },
        ),
        _sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '{"key": "section_1"}'},
            },
        ),
        _sse_event("content_block_stop", {"type": "content_block_stop", "index": 1}),
        _sse_event(
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 12}},
        ),
        _sse_event("message_stop", {"type": "message_stop"}),
    ]


LIVE_TEXT_FRAMES = 4


def _follow_up_stream() -> List[bytes]:
    return [
        _compact_sse_event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_follow_up",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-20250514",
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 60, "output_tokens": 0},
                },
            },
        ),
        _compact_sse_event(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        _compact_sse_event(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Section 1 says hi."}},
        ),
        _compact_sse_event("content_block_stop", {"type": "content_block_stop", "index": 0}),
        _compact_sse_event(
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 7}},
        ),
        _compact_sse_event("message_stop", {"type": "message_stop"}),
    ]


class GatedStream:
    """Yields `before`, then waits for `gate` before yielding `after`; `closed` records that it ended or was closed."""

    def __init__(self, before: List[bytes], after: List[bytes]):
        self.gate = asyncio.Event()
        self.closed = False
        self._chunks = self._run(before, after)

    async def _run(self, before: List[bytes], after: List[bytes]) -> AsyncGenerator[bytes, None]:
        try:
            for chunk in before:
                yield chunk
            await self.gate.wait()
            for chunk in after:
                yield chunk
        finally:
            self.closed = True

    def __aiter__(self) -> "GatedStream":
        return self

    async def __anext__(self) -> bytes:
        return await self._chunks.__anext__()

    async def aclose(self) -> None:
        await self._chunks.aclose()


def _hooks_returning(result: object) -> MagicMock:
    handler = MagicMock()
    handler._call_agentic_completion_hooks = AsyncMock(return_value=result)
    return handler


def _hooks_running(hooks: Callable[..., Coroutine[object, object, object]]) -> MagicMock:
    handler = MagicMock()
    handler._call_agentic_completion_hooks = AsyncMock(side_effect=hooks)
    return handler


async def _read_until_ping(iterator: AgenticAnthropicStreamingIterator) -> List[bytes]:
    chunks = []
    while (chunk := await iterator.__anext__()) != STREAM_SSE_KEEPALIVE_PING_BYTES:
        chunks.append(chunk)
    return chunks


async def _drain_without_pings(iterator: AgenticAnthropicStreamingIterator) -> List[bytes]:
    return [chunk async for chunk in iterator if chunk != STREAM_SSE_KEEPALIVE_PING_BYTES]


def _wire_events(chunks: List[bytes]) -> List[Tuple[str, dict]]:
    return [(event, json.loads(data)) for event, data in _sse_frames(b"".join(chunks).decode())]


def _block_starts(events: List[Tuple[str, dict]]) -> List[Tuple[int, str]]:
    return [(data["index"], data["content_block"]["type"]) for event, data in events if event == "content_block_start"]


# ---------------------------------------------------------------------------
# Tests for _parse_sse_events
# ---------------------------------------------------------------------------


class TestParseSSEEvents:
    def test_should_parse_single_event(self):
        raw = _sse_event("message_start", {"type": "message_start", "message": {"id": "1"}})
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
        result = AgenticAnthropicStreamingIterator._rebuild_anthropic_response_from_sse(raw_bytes)
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
        result = AgenticAnthropicStreamingIterator._rebuild_anthropic_response_from_sse(raw_bytes)
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
        result = AgenticAnthropicStreamingIterator._rebuild_anthropic_response_from_sse(raw_bytes)
        assert result is None

    def test_should_handle_empty_bytes(self):
        result = AgenticAnthropicStreamingIterator._rebuild_anthropic_response_from_sse([])
        assert result is None

    def test_should_handle_multi_event_chunks(self):
        """When multiple SSE events arrive in a single bytes chunk."""
        combined = b"".join(_build_simple_text_stream())
        result = AgenticAnthropicStreamingIterator._rebuild_anthropic_response_from_sse([combined])
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
        result = AgenticAnthropicStreamingIterator._rebuild_anthropic_response_from_sse(raw_bytes)
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
        result = AgenticAnthropicStreamingIterator._rebuild_anthropic_response_from_sse(raw_bytes)
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

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "non_object_frame",
        [
            b"data: 5\n\n",
            b"event: content_block_delta\ndata: [1]\n\n",
            b"data: null\n\n",
            b"data: " + b'{"a": ' * 100_000 + b"1" + b"}" * 100_000 + b"\n\n",
        ],
        ids=["number", "array", "null", "deep-object"],
    )
    async def test_should_skip_a_data_line_it_cannot_read_as_an_object(self, non_object_frame: bytes):
        text_stream = _build_simple_text_stream()
        chunks = [*text_stream[:2], non_object_frame, *text_stream[2:]]
        hooks = AsyncMock(return_value=None)
        mock_handler = MagicMock()
        mock_handler._call_agentic_completion_hooks = hooks

        iterator = AgenticAnthropicStreamingIterator(
            completion_stream=MockAsyncStream(chunks),
            http_handler=mock_handler,
            model="claude-sonnet-4-20250514",
            messages=[],
            anthropic_messages_provider_config=MagicMock(),
            anthropic_messages_optional_request_params={},
            logging_obj=MagicMock(),
            custom_llm_provider="anthropic",
            kwargs={},
        )

        collected = [chunk async for chunk in iterator]

        assert collected == chunks
        rebuilt = hooks.await_args.kwargs["response"]
        assert rebuilt["id"] == "msg_123"
        assert [(block["type"], block["text"]) for block in rebuilt["content"]] == [("text", "Hello, world!")]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "skipped_frame",
        [
            b"event: ping\ndata: not json\n\n",
            b"event: ping\ndata: " + b"9" * 5000 + b"\n\n",
            b'event: ping\ndata: {"a": ' + b'{"a": ' * 100_000 + b"1" + b"}" * 100_001 + b"\n\n",
        ],
        ids=["not-json", "huge-number", "deep-object"],
    )
    async def test_should_not_carry_a_skipped_line_event_name_to_the_next_data_line(self, skipped_frame: bytes):
        text_stream = _build_simple_text_stream()
        delta_without_event_line = text_stream[2].split(b"\n", 1)[1]
        chunks = [*text_stream[:2], skipped_frame, delta_without_event_line, *text_stream[3:]]
        hooks = AsyncMock(return_value=None)
        mock_handler = MagicMock()
        mock_handler._call_agentic_completion_hooks = hooks

        iterator = AgenticAnthropicStreamingIterator(
            completion_stream=MockAsyncStream(chunks),
            http_handler=mock_handler,
            model="claude-sonnet-4-20250514",
            messages=[],
            anthropic_messages_provider_config=MagicMock(),
            anthropic_messages_optional_request_params={},
            logging_obj=MagicMock(),
            custom_llm_provider="anthropic",
            kwargs={},
        )

        collected = [chunk async for chunk in iterator]

        assert collected == chunks
        rebuilt = hooks.await_args.kwargs["response"]
        assert [(block["type"], block["text"]) for block in rebuilt["content"]] == [("text", "Hello, world!")]


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
        mock_handler._call_agentic_completion_hooks = AsyncMock(return_value=fake_response)

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
    async def test_a_hook_failure_reaches_the_client_after_the_streamed_message(self):
        chunks = _build_simple_text_stream()
        mock_stream = MockAsyncStream(chunks)

        mock_handler = MagicMock()
        mock_handler._call_agentic_completion_hooks = AsyncMock(side_effect=RuntimeError("hook exploded"))

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

        async def _drain():
            async for chunk in iterator:
                collected.append(chunk)

        with pytest.raises(RuntimeError, match="hook exploded"):
            await _drain()

        assert collected == chunks

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


class TestAgenticStreamingIteratorHoldBack:
    @pytest.mark.asyncio
    async def test_text_before_a_tool_use_reaches_the_client_while_the_model_is_still_writing(self):
        chunks = _text_then_tool_use_stream()
        upstream = GatedStream(chunks[:LIVE_TEXT_FRAMES], chunks[LIVE_TEXT_FRAMES:])
        iterator = _build_hold_back_iterator(upstream, _hooks_returning(None))

        for expected in chunks[:LIVE_TEXT_FRAMES]:
            assert await iterator.__anext__() == expected

        upstream.gate.set()
        assert await _drain_without_pings(iterator) == [SERVER_FULFILLED_TOOL_LEAK_ERROR_SSE_BYTES]

    @pytest.mark.asyncio
    async def test_a_follow_up_continues_the_message_the_client_is_already_reading(self):
        follow_up = _follow_up_stream()
        iterator = _build_hold_back_iterator(
            MockAsyncStream(_text_then_tool_use_stream()), _hooks_returning(MockAsyncStream(follow_up))
        )

        received = await _drain_without_pings(iterator)
        events = _wire_events(received)

        assert [data["message"]["id"] for event, data in events if event == "message_start"] == ["msg_live"]
        assert _block_starts(events) == [(0, "text"), (1, "text")]
        assert [(data["index"], data["delta"]["text"]) for event, data in events if event == "content_block_delta"] == [
            (0, "Let me look that up."),
            (1, "Section 1 says hi."),
        ]
        assert [data["index"] for event, data in events if event == "content_block_stop"] == [0, 1]
        assert received[-2:] == follow_up[-2:]
        assert b"tool_use" not in b"".join(received)

    @pytest.mark.asyncio
    async def test_a_follow_up_after_a_leading_tool_use_passes_through_unchanged_without_its_message_start(self):
        tool_use_first = _text_then_tool_use_stream()
        upstream = [tool_use_first[0], *tool_use_first[LIVE_TEXT_FRAMES:]]
        follow_up = _follow_up_stream()
        iterator = _build_hold_back_iterator(
            MockAsyncStream(upstream),
            _hooks_returning(MockAsyncStream([STREAM_SSE_KEEPALIVE_PING_BYTES, *follow_up])),
        )

        assert [chunk async for chunk in iterator] == [upstream[0], STREAM_SSE_KEEPALIVE_PING_BYTES, *follow_up[1:]]

    @pytest.mark.asyncio
    async def test_a_non_streaming_follow_up_continues_the_message_too(self):
        follow_up = {
            "id": "msg_follow_up",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-20250514",
            "content": [{"type": "text", "text": "Section 1 says hi."}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 60, "output_tokens": 7},
        }
        iterator = _build_hold_back_iterator(MockAsyncStream(_text_then_tool_use_stream()), _hooks_returning(follow_up))

        events = _wire_events(await _drain_without_pings(iterator))

        assert [data["message"]["id"] for event, data in events if event == "message_start"] == ["msg_live"]
        assert _block_starts(events) == [(0, "text"), (1, "text")]
        assert [event for event, _ in events][-2:] == ["message_delta", "message_stop"]

    @pytest.mark.asyncio
    async def test_a_hook_that_resends_the_live_message_does_not_repeat_what_the_client_already_has(self):
        resent = {
            "id": "msg_live",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-20250514",
            "content": [{"type": "text", "text": "Let me look that up."}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 30, "output_tokens": 9},
        }
        iterator = _build_hold_back_iterator(MockAsyncStream(_text_then_tool_use_stream()), _hooks_returning(resent))

        events = _wire_events(await _drain_without_pings(iterator))

        assert _block_starts(events) == [(0, "text")]
        assert [data["delta"]["text"] for event, data in events if event == "content_block_delta"] == [
            "Let me look that up."
        ]
        assert [data["delta"]["stop_reason"] for event, data in events if event == "message_delta"] == ["end_turn"]

    @pytest.mark.asyncio
    async def test_a_server_tool_use_the_hooks_left_in_place_fails_the_turn_after_the_live_text(self):
        chunks = _text_then_tool_use_stream()
        iterator = _build_hold_back_iterator(MockAsyncStream(chunks), _hooks_returning(None))

        assert [chunk async for chunk in iterator] == [
            *chunks[:LIVE_TEXT_FRAMES],
            SERVER_FULFILLED_TOOL_LEAK_ERROR_SSE_BYTES,
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("chunk_size", [1, 7, 64])
    async def test_a_plain_answer_reaches_the_client_as_the_upstream_frames_however_upstream_splits_them(
        self, chunk_size
    ):
        message_start, *rest = _build_simple_text_stream()
        frames = [message_start, b": upstream comment\n\n", *rest]
        wire = b"".join(frames)
        upstream = [wire[start : start + chunk_size] for start in range(0, len(wire), chunk_size)]
        handler = _hooks_returning(None)
        iterator = _build_hold_back_iterator(MockAsyncStream(upstream), handler)

        assert [chunk async for chunk in iterator] == frames
        handler._call_agentic_completion_hooks.assert_awaited_once()

    @_on_virtual_clock
    async def test_a_client_tool_use_waits_for_the_hooks_then_reaches_the_client_unchanged(self):
        chunks = _text_then_tool_use_stream(tool_name="get_weather")
        release_hooks = asyncio.Event()

        async def hooks(**_kwargs):
            await release_hooks.wait()

        iterator = _build_hold_back_iterator(MockAsyncStream(chunks), _hooks_running(hooks))

        assert await _read_until_ping(iterator) == chunks[:LIVE_TEXT_FRAMES]
        assert await iterator.__anext__() == STREAM_SSE_KEEPALIVE_PING_BYTES
        release_hooks.set()
        assert await _drain_without_pings(iterator) == chunks[LIVE_TEXT_FRAMES:]

    @pytest.mark.parametrize("first_ending_frame", ["message_delta", "message_stop"])
    @_on_virtual_clock
    async def test_the_end_of_an_answer_waits_for_the_hooks(self, first_ending_frame):
        frames = _build_simple_text_stream()
        chunks = frames if first_ending_frame == "message_delta" else [*frames[:LIVE_TEXT_FRAMES], frames[-1]]
        release_hooks = asyncio.Event()

        async def hooks(**_kwargs):
            await release_hooks.wait()

        iterator = _build_hold_back_iterator(MockAsyncStream(chunks), _hooks_running(hooks))

        assert await _read_until_ping(iterator) == chunks[:LIVE_TEXT_FRAMES]
        release_hooks.set()
        assert await _drain_without_pings(iterator) == chunks[LIVE_TEXT_FRAMES:]

    @_on_virtual_clock
    async def test_a_frame_the_stream_cut_off_waits_for_the_hooks(self):
        streamed = _build_simple_text_stream()[:3]
        cut_off_delta = _sse_event(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " and then"}},
        ).rstrip(b"\n")
        release_hooks = asyncio.Event()

        async def hooks(**_kwargs):
            await release_hooks.wait()

        iterator = _build_hold_back_iterator(MockAsyncStream([*streamed, cut_off_delta]), _hooks_running(hooks))

        assert await _read_until_ping(iterator) == streamed
        release_hooks.set()
        assert await _drain_without_pings(iterator) == [cut_off_delta]

    @_on_virtual_clock
    async def test_keepalive_pings_cover_a_stalled_upstream_without_losing_or_repeating_a_frame(self):
        chunks = _build_simple_text_stream()
        upstream = GatedStream(chunks[:2], chunks[2:])
        iterator = _build_hold_back_iterator(upstream, _hooks_returning(None))

        assert await _read_until_ping(iterator) == chunks[:2]
        assert await iterator.__anext__() == STREAM_SSE_KEEPALIVE_PING_BYTES
        upstream.gate.set()
        assert await _drain_without_pings(iterator) == chunks[2:]

    @_on_virtual_clock
    async def test_keepalive_pings_cover_a_slow_held_tail_and_slow_hooks_without_leaking_held_frames(self):
        chunks = _text_then_tool_use_stream()
        upstream = GatedStream(chunks[: LIVE_TEXT_FRAMES + 1], chunks[LIVE_TEXT_FRAMES + 1 :])
        hooks_started = asyncio.Event()
        release_hooks = asyncio.Event()

        async def hooks(**_kwargs):
            hooks_started.set()
            await release_hooks.wait()
            return MockAsyncStream(_follow_up_stream())

        iterator = _build_hold_back_iterator(upstream, _hooks_running(hooks))

        assert await _read_until_ping(iterator) == chunks[:LIVE_TEXT_FRAMES]
        assert await iterator.__anext__() == STREAM_SSE_KEEPALIVE_PING_BYTES
        upstream.gate.set()
        while not hooks_started.is_set():
            assert await iterator.__anext__() == STREAM_SSE_KEEPALIVE_PING_BYTES
        assert await iterator.__anext__() == STREAM_SSE_KEEPALIVE_PING_BYTES
        release_hooks.set()
        received = await _drain_without_pings(iterator)

        assert _block_starts(_wire_events(received)) == [(1, "text")]
        assert b"litellm_content_retrieve" not in b"".join(received)

    @_on_virtual_clock
    async def test_keepalive_pings_keep_flowing_while_a_held_tool_use_trickles_in_faster_than_the_interval(self):
        chunks = _text_then_tool_use_stream(tool_name="get_weather")
        input_delta = _sse_event(
            "content_block_delta",
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": " "}},
        )
        trickled = [*chunks[: LIVE_TEXT_FRAMES + 1], *[input_delta] * 10, *chunks[LIVE_TEXT_FRAMES + 1 :]]
        iterator = _build_hold_back_iterator(
            MockSlowAsyncStream(trickled, delay_seconds=PING_INTERVAL_SECONDS * 0.4), _hooks_returning(None)
        )

        received = [chunk async for chunk in iterator]

        assert received[:LIVE_TEXT_FRAMES] == chunks[:LIVE_TEXT_FRAMES]
        held = received[received.index(chunks[LIVE_TEXT_FRAMES - 1]) + 1 : received.index(chunks[LIVE_TEXT_FRAMES])]
        assert STREAM_SSE_KEEPALIVE_PING_BYTES in held
        assert [chunk for chunk in received if chunk != STREAM_SSE_KEEPALIVE_PING_BYTES] == trickled

    @pytest.mark.asyncio
    async def test_an_upstream_failure_before_any_output_fails_the_first_read_instead_of_answering_a_ping(self):
        handler = _hooks_returning(None)
        iterator = _build_hold_back_iterator(MockFailingAsyncStream([], RuntimeError("upstream died")), handler)

        with pytest.raises(RuntimeError, match="upstream died"):
            await iterator.__anext__()

        handler._call_agentic_completion_hooks.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("partial_frame_bytes", [0, 20], ids=["at_a_frame_boundary", "mid_frame"])
    async def test_an_upstream_failure_after_text_reaches_the_client_after_that_text_and_keeps_failing(
        self, partial_frame_bytes
    ):
        chunks = _text_then_tool_use_stream()
        handler = _hooks_returning(None)
        upstream = [*chunks[:LIVE_TEXT_FRAMES], chunks[LIVE_TEXT_FRAMES][:partial_frame_bytes]]
        iterator = _build_hold_back_iterator(MockFailingAsyncStream(upstream, RuntimeError("upstream died")), handler)

        received = []

        async def drain():
            async for chunk in iterator:
                received.append(chunk)

        with pytest.raises(RuntimeError, match="upstream died"):
            await drain()
        with pytest.raises(RuntimeError, match="upstream died"):
            await iterator.__anext__()

        assert received == chunks[:LIVE_TEXT_FRAMES]
        handler._call_agentic_completion_hooks.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_nested_agentic_follow_up_error_reaches_the_client_byte_identical(self):
        nested = _build_hold_back_iterator(MockAsyncStream(_text_then_tool_use_stream()), _hooks_returning(None))
        iterator = _build_hold_back_iterator(MockAsyncStream(_text_then_tool_use_stream()), _hooks_returning(nested))

        received = await _drain_without_pings(iterator)
        events = _wire_events(received[:-1])

        assert received[-1] == SERVER_FULFILLED_TOOL_LEAK_ERROR_SSE_BYTES
        assert [data["message"]["id"] for event, data in events if event == "message_start"] == ["msg_live"]
        assert _block_starts(events) == [(0, "text"), (1, "text")]

    @pytest.mark.asyncio
    async def test_a_rate_limited_follow_up_reaches_the_client_as_its_anthropic_error_event(self):
        chunks = _build_tool_use_stream()

        mock_handler = MagicMock()
        mock_handler._call_agentic_completion_hooks = AsyncMock(
            side_effect=litellm.RateLimitError(
                message="follow-up rate limited", llm_provider="anthropic", model="claude-sonnet-4-20250514"
            )
        )

        wire = "".join(
            [
                frame if isinstance(frame, str) else frame.decode()
                async for frame in ProxyBaseLLMRequestProcessing.async_sse_data_generator(
                    response=_build_hold_back_iterator(MockAsyncStream(chunks), mock_handler),
                    user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"),
                    request_data={"model": "claude-sonnet-4-20250514"},
                    proxy_logging_obj=ProxyLogging(user_api_key_cache=MagicMock()),
                )
            ]
        )

        error_events = [json.loads(data) for event, data in _sse_frames(wire) if event == "error"]
        assert [event["error"]["type"] for event in error_events] == ["rate_limit_error"]
        assert "follow-up rate limited" in error_events[0]["error"]["message"]
        assert "litellm_content_retrieve" not in wire

    @pytest.mark.asyncio
    async def test_a_capped_loop_tells_the_client_to_raise_max_agentic_loops_instead_of_retrying(self):
        mock_handler = MagicMock()
        mock_handler._call_agentic_completion_hooks = AsyncMock(
            side_effect=lambda **_: BaseLLMHTTPHandler._check_agentic_loop_safety(
                tool_calls=[{"name": "litellm_content_retrieve"}],
                fingerprints=[],
                depth=1,
                max_loops=1,
                model="claude-sonnet-4-20250514",
            )
        )

        iterator = _build_hold_back_iterator(MockAsyncStream(_build_tool_use_stream()), mock_handler)
        wire = b"".join([chunk async for chunk in iterator if chunk != STREAM_SSE_KEEPALIVE_PING_BYTES]).decode()

        error_messages = [json.loads(data)["error"]["message"] for event, data in _sse_frames(wire) if event == "error"]
        assert len(error_messages) == 1
        assert "max_agentic_loops" in error_messages[0]
        assert "retry" not in error_messages[0].lower()
        assert "litellm_content_retrieve" not in wire

    @pytest.mark.asyncio
    async def test_should_replay_client_owned_tool_use_verbatim(self):
        """Only server-fulfilled tools are withheld: a client's own tool_use still reaches it byte-identical."""
        chunks = _build_tool_use_stream()

        mock_handler = MagicMock()
        mock_handler._call_agentic_completion_hooks = AsyncMock(return_value=None)

        iterator = _build_hold_back_iterator(
            MockAsyncStream(chunks),
            mock_handler,
            server_fulfilled_tool_names=frozenset({"headroom_retrieve"}),
        )

        collected = []
        async for chunk in iterator:
            collected.append(chunk)

        assert [c for c in collected if c != STREAM_SSE_KEEPALIVE_PING_BYTES] == chunks

    @_on_virtual_clock
    async def test_keepalive_pings_cover_a_stalled_follow_up(self):
        follow_up = _follow_up_stream()
        stalled = GatedStream(follow_up[:2], follow_up[2:])
        chunks = _text_then_tool_use_stream()
        iterator = _build_hold_back_iterator(
            MockAsyncStream(chunks), _hooks_returning(stalled)
        )

        before_stall = await _read_until_ping(iterator)
        stalled.gate.set()
        after_stall = await _drain_without_pings(iterator)

        assert before_stall[:LIVE_TEXT_FRAMES] == chunks[:LIVE_TEXT_FRAMES]
        assert _block_starts(_wire_events(before_stall[LIVE_TEXT_FRAMES:])) == [(1, "text")]
        assert [(event, data.get("index")) for event, data in _wire_events(after_stall)] == [
            ("content_block_delta", 1),
            ("content_block_stop", 1),
            ("message_delta", None),
            ("message_stop", None),
        ]

    @pytest.mark.asyncio
    async def test_should_propagate_follow_up_stream_error(self):
        """A failing follow-up stream surfaces its error instead of hanging on pings forever."""
        chunks = _build_tool_use_stream()

        mock_handler = MagicMock()
        mock_handler._call_agentic_completion_hooks = AsyncMock(
            return_value=MockFailingAsyncStream([b"follow-up-chunk"], RuntimeError("follow-up died"))
        )

        iterator = _build_hold_back_iterator(MockAsyncStream(chunks), mock_handler, ping_interval_seconds=0.02)

        with pytest.raises(RuntimeError, match="follow-up died"):
            async for _ in iterator:
                pass

    @_on_virtual_clock
    async def test_closing_while_upstream_is_pending_closes_upstream_without_running_hooks(self):
        chunks = _text_then_tool_use_stream()
        upstream = GatedStream(chunks[:LIVE_TEXT_FRAMES], chunks[LIVE_TEXT_FRAMES:])
        handler = _hooks_returning(None)
        iterator = _build_hold_back_iterator(upstream, handler)

        await _read_until_ping(iterator)
        await iterator.aclose()

        assert upstream.closed
        handler._call_agentic_completion_hooks.assert_not_awaited()

    @pytest.mark.parametrize("follow_up_frame_pending", [True, False])
    @_on_virtual_clock
    async def test_closing_during_the_follow_up_closes_it(self, follow_up_frame_pending):
        follow_up_frames = _follow_up_stream()
        follow_up = GatedStream(follow_up_frames[:2], follow_up_frames[2:])
        iterator = _build_hold_back_iterator(
            MockAsyncStream(_text_then_tool_use_stream()), _hooks_returning(follow_up)
        )

        if follow_up_frame_pending:
            await _read_until_ping(iterator)
        else:
            for _ in range(LIVE_TEXT_FRAMES + 1):
                await iterator.__anext__()
        await iterator.aclose()

        assert follow_up.closed

    @_on_virtual_clock
    async def test_closing_while_hooks_run_cancels_them(self):
        hooks_started = asyncio.Event()
        hooks_cancelled = asyncio.Event()

        async def hooks(**_kwargs):
            hooks_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                hooks_cancelled.set()
                raise

        iterator = _build_hold_back_iterator(
            MockAsyncStream(_text_then_tool_use_stream()), _hooks_running(hooks)
        )

        while not hooks_started.is_set():
            await iterator.__anext__()
        await iterator.aclose()

        assert hooks_cancelled.is_set()
