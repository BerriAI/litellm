"""
Unit tests for the Command Code stream iterator.

Fully mocked - streams are built from in-memory httpx responses, no real
network calls.
"""

import json
import os
import sys
from typing import AsyncIterator, Iterator, List

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))  # Adds the parent directory to the system path

from litellm.llms.command_code.chat.sse_iterator import CommandCodeSSEStreamIterator
from litellm.llms.command_code.common_utils import CommandCodeError

MODEL = "command_code/gpt-5.5"
REQUEST = httpx.Request("POST", "https://api.commandcode.ai/alpha/generate")


class ChunkedByteStream(httpx.SyncByteStream):
    """Byte stream that yields the payload in caller-defined chunks."""

    def __init__(self, chunks: List[bytes]):
        self._chunks = chunks

    def __iter__(self) -> Iterator[bytes]:
        yield from self._chunks


class AsyncChunkedByteStream(httpx.AsyncByteStream):
    """Async byte stream that yields the payload in caller-defined chunks."""

    def __init__(self, chunks: List[bytes]):
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


def make_sync_iterator(payload: bytes) -> CommandCodeSSEStreamIterator:
    response = httpx.Response(status_code=200, content=payload, request=REQUEST)
    return iter(CommandCodeSSEStreamIterator(response=response, model=MODEL))


def collect_sync(payload: bytes) -> list:
    return list(make_sync_iterator(payload))


def event_line(event: dict) -> bytes:
    return (json.dumps(event) + "\n").encode("utf-8")


class TestEventTranslation:
    def test_text_delta(self):
        chunks = collect_sync(
            event_line({"type": "text-delta", "text": "Hello"}) + event_line({"type": "finish", "finishReason": "stop"})
        )
        assert chunks[0].choices[0].delta.content == "Hello"
        assert chunks[0].choices[0].finish_reason is None

    def test_reasoning_delta(self):
        chunks = collect_sync(
            event_line({"type": "reasoning-delta", "text": "thinking"})
            + event_line({"type": "finish", "finishReason": "stop"})
        )
        assert chunks[0].choices[0].delta.reasoning_content == "thinking"
        assert chunks[0].choices[0].delta.content is None

    def test_tool_call_complete_delta(self):
        chunks = collect_sync(
            event_line(
                {
                    "type": "tool-call",
                    "toolCallId": "call_1",
                    "toolName": "get_weather",
                    "input": {"city": "sf"},
                }
            )
            + event_line(
                {
                    "type": "tool-call",
                    "toolCallId": "call_2",
                    "toolName": "get_time",
                    "input": {"tz": "utc"},
                }
            )
            + event_line({"type": "finish", "finishReason": "tool-calls"})
        )
        first_call = chunks[0].choices[0].delta.tool_calls[0]
        assert first_call.id == "call_1"
        assert first_call.index == 0
        assert first_call.function.name == "get_weather"
        assert json.loads(first_call.function.arguments) == {"city": "sf"}

        second_call = chunks[1].choices[0].delta.tool_calls[0]
        assert second_call.id == "call_2"
        assert second_call.index == 1

        assert chunks[2].choices[0].finish_reason == "tool_calls"

    def test_tool_call_input_as_json_string(self):
        chunks = collect_sync(
            event_line(
                {
                    "type": "tool-call",
                    "toolCallId": "call_1",
                    "toolName": "get_weather",
                    "input": '{"city": "sf"}',
                }
            )
            + event_line({"type": "finish", "finishReason": "tool-calls"})
        )
        tool_call = chunks[0].choices[0].delta.tool_calls[0]
        assert json.loads(tool_call.function.arguments) == {"city": "sf"}

    def test_tool_result_event_is_noop(self):
        chunks = collect_sync(
            event_line({"type": "tool-result", "toolCallId": "call_1"})
            + event_line({"type": "text-delta", "text": "done"})
            + event_line({"type": "finish", "finishReason": "stop"})
        )
        # tool-result produced no chunk: only text-delta + finish
        assert len(chunks) == 2
        assert chunks[0].choices[0].delta.content == "done"

    def test_reasoning_start_end_are_noops(self):
        chunks = collect_sync(
            event_line({"type": "reasoning-start"})
            + event_line({"type": "reasoning-delta", "text": "hmm"})
            + event_line({"type": "reasoning-end"})
            + event_line({"type": "finish", "finishReason": "stop"})
        )
        assert len(chunks) == 2

    @pytest.mark.parametrize(
        "command_code_reason,expected",
        [
            ("tool-calls", "tool_calls"),
            ("length", "length"),
            ("max_tokens", "length"),
            ("max-tokens", "length"),
            ("max_output_tokens", "length"),
            ("something-new", "stop"),
        ],
    )
    def test_finish_reason_mapping(self, command_code_reason, expected):
        chunks = collect_sync(event_line({"type": "finish", "finishReason": command_code_reason}))
        assert chunks[-1].choices[0].finish_reason == expected

    def test_finish_usage_with_cache_tokens(self):
        chunks = collect_sync(
            event_line(
                {
                    "type": "finish",
                    "finishReason": "stop",
                    "totalUsage": {
                        "inputTokens": 100,
                        "outputTokens": 20,
                        "inputTokenDetails": {"cacheReadTokens": 30, "cacheWriteTokens": 10},
                    },
                }
            )
        )
        usage = chunks[-1].usage
        assert usage.prompt_tokens == 140  # 100 + 30 cache read + 10 cache write
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 160
        assert usage.cache_read_input_tokens == 30
        assert usage.cache_creation_input_tokens == 10
        assert usage.prompt_tokens_details.cached_tokens == 30

    def test_error_event_raises_command_code_error(self):
        iterator = make_sync_iterator(event_line({"type": "error", "error": {"message": "rate limited"}}))
        with pytest.raises(CommandCodeError, match="rate limited"):
            next(iterator)

    def test_error_event_with_string_error(self):
        iterator = make_sync_iterator(event_line({"type": "error", "error": "boom"}))
        with pytest.raises(CommandCodeError, match="boom"):
            next(iterator)

    def test_stream_end_without_finish_emits_stop(self):
        chunks = collect_sync(event_line({"type": "text-delta", "text": "partial"}))
        assert chunks[-1].choices[0].finish_reason == "stop"


class TestSSEFraming:
    def test_framing_variants(self):
        payload = (
            b'data: {"type": "text-delta", "text": "a"}\n'  # data: prefix
            b'{"type": "text-delta", "text": "b"}\n'  # bare JSON line
            b": keep-alive comment\n"  # comment line
            b"event: message\n"  # event: line
            b"\n"  # empty line
            b"data: [DONE]\n"  # DONE marker
            b'{"type": "finish", "finishReason": "stop"}\n'
        )
        chunks = collect_sync(payload)
        assert [c.choices[0].delta.content for c in chunks[:-1]] == ["a", "b"]
        assert chunks[-1].choices[0].finish_reason == "stop"

    def test_json_object_split_across_two_chunks(self):
        line = json.dumps({"type": "text-delta", "text": "split across chunks"}) + "\n"
        split_at = len(line) // 2
        stream = ChunkedByteStream(
            [
                line[:split_at].encode("utf-8"),
                line[split_at:].encode("utf-8"),
                event_line({"type": "finish", "finishReason": "stop"}),
            ]
        )
        response = httpx.Response(status_code=200, stream=stream, request=REQUEST)
        chunks = list(iter(CommandCodeSSEStreamIterator(response=response, model=MODEL)))
        assert chunks[0].choices[0].delta.content == "split across chunks"
        assert chunks[-1].choices[0].finish_reason == "stop"


class TestAsyncIteration:
    async def _collect_async(self, chunks: List[bytes]) -> list:
        response = httpx.Response(status_code=200, stream=AsyncChunkedByteStream(chunks), request=REQUEST)
        iterator = CommandCodeSSEStreamIterator(response=response, model=MODEL).__aiter__()
        collected = []
        async for chunk in iterator:
            collected.append(chunk)
        return collected

    @pytest.mark.asyncio
    async def test_async_matches_sync(self):
        payload = (
            event_line({"type": "reasoning-delta", "text": "hmm"})
            + event_line({"type": "text-delta", "text": "Hello"})
            + event_line(
                {
                    "type": "tool-call",
                    "toolCallId": "call_1",
                    "toolName": "get_weather",
                    "input": {"city": "sf"},
                }
            )
            + event_line(
                {
                    "type": "finish",
                    "finishReason": "tool-calls",
                    "totalUsage": {"inputTokens": 5, "outputTokens": 3},
                }
            )
        )
        sync_chunks = collect_sync(payload)
        async_chunks = await self._collect_async([payload])

        assert len(async_chunks) == len(sync_chunks)
        assert async_chunks[0].choices[0].delta.reasoning_content == "hmm"
        assert async_chunks[1].choices[0].delta.content == "Hello"
        tool_call = async_chunks[2].choices[0].delta.tool_calls[0]
        assert tool_call.function.name == "get_weather"
        assert async_chunks[3].choices[0].finish_reason == "tool_calls"
        assert async_chunks[3].usage.prompt_tokens == 5

    @pytest.mark.asyncio
    async def test_async_split_chunks(self):
        line = json.dumps({"type": "text-delta", "text": "async split"}) + "\n"
        split_at = len(line) // 2
        chunks = await self._collect_async(
            [
                line[:split_at].encode("utf-8"),
                line[split_at:].encode("utf-8"),
                event_line({"type": "finish", "finishReason": "stop"}),
            ]
        )
        assert chunks[0].choices[0].delta.content == "async split"
        assert chunks[-1].choices[0].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_async_error_event(self):
        response = httpx.Response(
            status_code=200,
            stream=AsyncChunkedByteStream([event_line({"type": "error", "error": {"message": "async boom"}})]),
            request=REQUEST,
        )
        iterator = CommandCodeSSEStreamIterator(response=response, model=MODEL).__aiter__()
        with pytest.raises(CommandCodeError, match="async boom"):
            await iterator.__anext__()
