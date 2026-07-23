"""
Regression tests for LIT-4313: sagemaker_chat streaming must forward each AWS
event-stream frame as it arrives instead of buffering to a fixed 1024-byte
threshold and then draining a burst of deltas.

The buffering came from `response.iter_bytes(chunk_size=1024)` /
`response.aiter_bytes(chunk_size=1024)`: httpx's ByteChunker withholds bytes until
`chunk_size` accumulates, so the first client delta could not be produced until
enough later frames had arrived to cross 1024 bytes, inflating TTFT and turning a
steady provider stream into gap-then-burst delivery.
"""

import binascii
import json
import struct
from typing import AsyncIterator, Iterator
from unittest.mock import MagicMock

import httpx
import pytest

from litellm.llms.sagemaker.chat.transformation import SagemakerChatConfig


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


def _delta_frame(index: int, content: str) -> bytes:
    sse = (
        "data: "
        + json.dumps(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
            }
        )
        + "\n\n"
    )
    return _encode_event_frame(sse.encode("utf-8"))


def _make_frames(n: int) -> list[bytes]:
    # Small single-token frames (< 1024 bytes each) so a fixed 1024-byte chunker
    # would have to swallow several frames before releasing the first delta.
    frames = [_delta_frame(i, f"token{i} ") for i in range(n)]
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


def _content_of(chunk) -> str | None:
    return chunk.choices[0].delta.content


def test_sync_first_event_emitted_after_a_single_frame():
    """The first delta must be available after exactly one source frame is pulled.

    With the old chunk_size=1024 the httpx chunker would consume several small
    frames before yielding, so `consumed` would be > 1 at the first delta.
    """
    frames = _make_frames(24)
    stream = _CountingSyncStream(frames)
    response = httpx.Response(200, stream=stream)

    wrapper = SagemakerChatConfig().get_sync_custom_stream_wrapper(
        model="phi-4",
        custom_llm_provider="sagemaker_chat",
        logging_obj=MagicMock(),
        api_base="https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/phi-4/invocations-response-stream",
        headers={},
        data={},
        messages=[],
        client=_FakeSyncClient(response),
    )

    first = next(c for c in wrapper.completion_stream if c is not None and _content_of(c) is not None)
    assert _content_of(first) == "token0 "
    assert stream.consumed == 1


def test_sync_events_emitted_incrementally_without_bursting():
    """Each successive delta must correspond to exactly one newly-pulled frame."""
    frames = _make_frames(24)
    stream = _CountingSyncStream(frames)
    response = httpx.Response(200, stream=stream)

    wrapper = SagemakerChatConfig().get_sync_custom_stream_wrapper(
        model="phi-4",
        custom_llm_provider="sagemaker_chat",
        logging_obj=MagicMock(),
        api_base="https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/phi-4/invocations-response-stream",
        headers={},
        data={},
        messages=[],
        client=_FakeSyncClient(response),
    )

    consumed_at_delta = [
        stream.consumed for chunk in wrapper.completion_stream if chunk is not None and _content_of(chunk) is not None
    ]

    assert consumed_at_delta == list(range(1, len(frames) + 1))


@pytest.mark.asyncio
async def test_async_first_event_emitted_after_a_single_frame():
    frames = _make_frames(24)
    stream = _CountingAsyncStream(frames)
    response = httpx.Response(200, stream=stream)

    wrapper = await SagemakerChatConfig().get_async_custom_stream_wrapper(
        model="phi-4",
        custom_llm_provider="sagemaker_chat",
        logging_obj=MagicMock(),
        api_base="https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/phi-4/invocations-response-stream",
        headers={},
        data={},
        messages=[],
        client=_FakeAsyncClient(response),
    )

    consumed_at_delta = []
    async for chunk in wrapper.completion_stream:
        if chunk is not None and _content_of(chunk) is not None:
            consumed_at_delta.append(stream.consumed)

    assert consumed_at_delta[0] == 1
    assert consumed_at_delta == list(range(1, len(frames) + 1))


def test_signed_body_includes_stream_flag():
    """A streaming request must carry `stream: true` in the signed body sent to SageMaker.

    `stream` flows into the request body through the transformed request (`{**optional_params}`)
    and must survive SigV4 signing so the endpoint enables token-level streaming.
    """
    headers, signed_body = SagemakerChatConfig().sign_request(
        headers={},
        optional_params={
            "aws_access_key_id": "AKIATESTTESTTESTTEST",
            "aws_secret_access_key": "test-secret-key",
            "aws_region_name": "us-east-1",
        },
        request_data={"model": "phi-4", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        api_base="https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/phi-4/invocations-response-stream",
        model="phi-4",
        stream=True,
    )
    assert signed_body is not None
    assert json.loads(signed_body)["stream"] is True
