"""
Regression tests for LIT-4313: the native `sagemaker/` streaming path must
forward each AWS event-stream frame as it arrives instead of buffering to a
fixed 1024-byte threshold and then draining a burst of tokens.

The buffering came from `response.aiter_bytes(chunk_size=1024)`: httpx's
ByteChunker withholds bytes until `chunk_size` accumulates, so the first token
could not be produced until enough later frames had arrived to cross 1024 bytes,
inflating TTFT and turning a steady provider stream into gap-then-burst delivery.
"""

import binascii
import json
import struct
from typing import AsyncIterator, Iterator
from unittest.mock import MagicMock

import httpx
import pytest

from litellm.llms.sagemaker.completion.handler import SagemakerLLM


def _encode_header(name: str, value: str) -> bytes:
    name_b = name.encode("utf-8")
    value_b = value.encode("utf-8")
    return struct.pack("B", len(name_b)) + name_b + struct.pack("B", 7) + struct.pack(">H", len(value_b)) + value_b


def _encode_event_frame(payload: bytes) -> bytes:
    """Encode one AWS event-stream message that botocore's EventStreamBuffer decodes."""
    headers = {
        ":event-type": "PayloadPart",
        ":content-type": "application/json",
        ":message-type": "event",
    }
    headers_b = b"".join(_encode_header(k, v) for k, v in headers.items())
    total_len = 16 + len(headers_b) + len(payload)
    prelude = struct.pack(">I", total_len) + struct.pack(">I", len(headers_b))
    prelude_crc = struct.pack(">I", binascii.crc32(prelude) & 0xFFFFFFFF)
    message = prelude + prelude_crc + headers_b + payload
    message_crc = struct.pack(">I", binascii.crc32(message) & 0xFFFFFFFF)
    return message + message_crc


def _token_frame(text: str) -> bytes:
    # SageMaker HF TGI streaming payloads are `{"token": {"text": ...}}` blobs.
    sse = "data: " + json.dumps({"token": {"text": text}}) + "\n\n"
    return _encode_event_frame(sse.encode("utf-8"))


def _make_frames(n: int) -> list[bytes]:
    frames = [_token_frame(f"token{i} ") for i in range(n)]
    assert all(len(f) < 1024 for f in frames)
    return frames


class _CountingSyncStream(httpx.SyncByteStream):
    """Yields provider frames one at a time and records how many have been pulled."""

    def __init__(self, frames: list[bytes]) -> None:
        self._frames = frames
        self.consumed = 0

    def __iter__(self) -> Iterator[bytes]:
        for frame in self._frames:
            self.consumed += 1
            yield frame


class _CountingAsyncStream(httpx.AsyncByteStream):
    """Yields provider frames one at a time and records how many have been pulled."""

    def __init__(self, frames: list[bytes]) -> None:
        self._frames = frames
        self.consumed = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for frame in self._frames:
            self.consumed += 1
            yield frame


class _FakeSyncClient:
    def __init__(self, response: httpx.Response) -> None:
        self._response = response

    def post(self, *args, **kwargs) -> httpx.Response:
        return self._response


class _FakeAsyncClient:
    def __init__(self, response: httpx.Response) -> None:
        self._response = response

    async def post(self, *args, **kwargs) -> httpx.Response:
        return self._response


def test_sync_native_streaming_forwards_each_frame_incrementally():
    """Each token must be emitted after exactly one newly-pulled source frame.

    With the old `chunk_size=1024` the httpx chunker would swallow several small
    frames before yielding, so the first token would arrive only after `consumed`
    had already crossed multiple frames, and tokens would then replay in a burst.
    """
    frames = _make_frames(24)
    stream = _CountingSyncStream(frames)
    response = httpx.Response(200, stream=stream)

    completion_stream = SagemakerLLM().make_sync_call(
        api_base="https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/phi-4/invocations-response-stream",
        headers={},
        data="",
        logging_obj=MagicMock(),
        client=_FakeSyncClient(response),
    )

    consumed_at_token = []
    texts = []
    for chunk in completion_stream:
        if chunk is not None and chunk["text"]:
            consumed_at_token.append(stream.consumed)
            texts.append(chunk["text"])

    assert texts == [f"token{i} " for i in range(len(frames))]
    assert consumed_at_token == list(range(1, len(frames) + 1))


@pytest.mark.asyncio
async def test_async_native_streaming_forwards_each_frame_incrementally():
    """Each token must be emitted after exactly one newly-pulled source frame.

    With the old `chunk_size=1024` the httpx chunker would swallow several small
    frames before yielding, so the first token would arrive only after `consumed`
    had already crossed multiple frames, and tokens would then replay in a burst.
    """
    frames = _make_frames(24)
    stream = _CountingAsyncStream(frames)
    response = httpx.Response(200, stream=stream)

    completion_stream = await SagemakerLLM().make_async_call(
        api_base="https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/phi-4/invocations-response-stream",
        headers={},
        data="",
        logging_obj=MagicMock(),
        client=_FakeAsyncClient(response),
    )

    consumed_at_token = []
    texts = []
    async for chunk in completion_stream:
        if chunk is not None and chunk["text"]:
            consumed_at_token.append(stream.consumed)
            texts.append(chunk["text"])

    assert texts == [f"token{i} " for i in range(len(frames))]
    assert consumed_at_token == list(range(1, len(frames) + 1))
