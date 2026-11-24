import json
import threading
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final, TypeVar

import httpx._client
import pytest

import litellm
from litellm.llms.custom_httpx.accept_encoding import (
    ACCEPT_ENCODING_HEADER,
    decodable_accept_encoding,
    httpx_accept_encoding,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

zstandard = pytest.importorskip("zstandard")

_TEXT_PIECES: Final = ("he", "llo")
_FIRST_EVENT_READ_TIMEOUT_SECONDS: Final = 5

T = TypeVar("T")


def _sse(payload: dict[str, object]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def _chat_events() -> tuple[bytes, ...]:
    return tuple(
        _sse(
            {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": piece},
                        "finish_reason": "stop" if index == len(_TEXT_PIECES) - 1 else None,
                    }
                ],
            }
        )
        for index, piece in enumerate(_TEXT_PIECES)
    ) + (b"data: [DONE]\n\n",)


def _text_events() -> tuple[bytes, ...]:
    return tuple(
        _sse(
            {
                "id": "cmpl-1",
                "object": "text_completion",
                "created": 1,
                "model": "gpt-3.5-turbo-instruct",
                "choices": [
                    {
                        "index": 0,
                        "text": piece,
                        "finish_reason": "stop" if index == len(_TEXT_PIECES) - 1 else None,
                    }
                ],
            }
        )
        for index, piece in enumerate(_TEXT_PIECES)
    ) + (b"data: [DONE]\n\n",)


class _FramePerEventServer(ThreadingHTTPServer):
    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _FramePerEventHandler)
        self.first_event_read: Final = threading.Event()


class _FramePerEventHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _write_chunk(self, frame: bytes) -> None:
        self.wfile.write(b"%X\r\n%s\r\n" % (len(frame), frame))
        self.wfile.flush()

    def do_POST(self) -> None:
        assert isinstance(self.server, _FramePerEventServer)
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        events: Final = _chat_events() if self.path.endswith("/chat/completions") else _text_events()
        offers_zstd: Final = "zstd" in self.headers.get("Accept-Encoding", "")
        frames: Final = tuple(zstandard.ZstdCompressor().compress(event) for event in events) if offers_zstd else events
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        if offers_zstd:
            self.send_header("Content-Encoding", "zstd")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        self._write_chunk(frames[0])
        self.server.first_event_read.wait(_FIRST_EVENT_READ_TIMEOUT_SECONDS)
        for frame in frames[1:]:
            self._write_chunk(frame)
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def log_message(self, format: str, *args: object) -> None:
        return


@dataclass(frozen=True, slots=True)
class _Upstream:
    api_base: str
    first_event_read: threading.Event


@pytest.fixture
def frame_per_event_upstream() -> Iterator[_Upstream]:
    server: Final = _FramePerEventServer()
    thread: Final = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield _Upstream(api_base=f"http://127.0.0.1:{server.server_port}/v1", first_event_read=server.first_event_read)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize(
    ("advertised", "expected"),
    [
        ("gzip, deflate, br, zstd", "gzip, deflate, br"),
        ("gzip;q=1.0, deflate, br, zstd;q=0.9", "gzip;q=1.0, deflate, br"),
        (" zstd , gzip ", "gzip"),
        ("zstd", "identity"),
        ("gzip, deflate", "gzip, deflate"),
    ],
)
def test_zstd_is_the_only_encoding_dropped(advertised: str, expected: str) -> None:
    assert decodable_accept_encoding(advertised) == expected


def _releasing_after_each(chunks: Iterator[T], release: threading.Event) -> Iterator[T]:
    for chunk in chunks:
        yield chunk
        release.set()


async def _areleasing_after_each(chunks: AsyncIterator[T], release: threading.Event) -> AsyncIterator[T]:
    async for chunk in chunks:
        yield chunk
        release.set()


def _encodings(header: str) -> frozenset[str]:
    return frozenset(part.strip() for part in header.split(","))


def test_the_header_offers_every_encoding_httpx_offers_but_zstd() -> None:
    assert _encodings(ACCEPT_ENCODING_HEADER["Accept-Encoding"]) == _encodings(httpx_accept_encoding()) - {"zstd"}


def test_an_httpx_without_its_accept_encoding_constant_falls_back_to_gzip_and_deflate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(httpx._client, "ACCEPT_ENCODING")

    assert httpx_accept_encoding() == "gzip, deflate"


@pytest.mark.asyncio
async def test_async_http_handler_streams_every_event_from_an_upstream_that_frames_each_one(
    frame_per_event_upstream: _Upstream,
) -> None:
    handler: Final = AsyncHTTPHandler()
    try:
        response: Final = await handler.post(
            f"{frame_per_event_upstream.api_base}/chat/completions", json={}, stream=True
        )
        body: Final = b"".join(
            [
                chunk
                async for chunk in _areleasing_after_each(
                    response.aiter_bytes(), frame_per_event_upstream.first_event_read
                )
            ]
        )
    finally:
        await handler.close()

    assert body == b"".join(_chat_events())


def test_http_handler_streams_every_event_from_an_upstream_that_frames_each_one(
    frame_per_event_upstream: _Upstream,
) -> None:
    handler: Final = HTTPHandler()
    try:
        response: Final = handler.post(f"{frame_per_event_upstream.api_base}/chat/completions", json={}, stream=True)
        body: Final = b"".join(_releasing_after_each(response.iter_bytes(), frame_per_event_upstream.first_event_read))
    finally:
        handler.close()

    assert body == b"".join(_chat_events())


def _chat_text(chunk: litellm.ModelResponseStream) -> str:
    return chunk.choices[0].delta.content or ""


def test_sync_openai_chat_stream_survives_an_upstream_that_frames_each_event(
    frame_per_event_upstream: _Upstream,
) -> None:
    stream: Final = litellm.completion(
        model="openai/gpt-4o",
        messages=[{"role": "user", "content": "say hello"}],
        stream=True,
        api_base=frame_per_event_upstream.api_base,
        api_key="sk-not-a-real-key",
    )

    assert "".join(
        _chat_text(chunk) for chunk in _releasing_after_each(iter(stream), frame_per_event_upstream.first_event_read)
    ) == "".join(_TEXT_PIECES)


@pytest.mark.asyncio
async def test_async_openai_chat_stream_survives_an_upstream_that_frames_each_event(
    frame_per_event_upstream: _Upstream,
) -> None:
    stream: Final = await litellm.acompletion(
        model="openai/gpt-4o",
        messages=[{"role": "user", "content": "say hello"}],
        stream=True,
        api_base=frame_per_event_upstream.api_base,
        api_key="sk-not-a-real-key",
    )

    assert "".join(
        [
            _chat_text(chunk)
            async for chunk in _areleasing_after_each(aiter(stream), frame_per_event_upstream.first_event_read)
        ]
    ) == "".join(_TEXT_PIECES)


def test_sync_openai_text_completion_stream_survives_an_upstream_that_frames_each_event(
    frame_per_event_upstream: _Upstream,
) -> None:
    stream: Final = litellm.text_completion(
        model="text-completion-openai/gpt-3.5-turbo-instruct",
        prompt="say hello",
        stream=True,
        api_base=frame_per_event_upstream.api_base,
        api_key="sk-not-a-real-key",
    )

    assert "".join(
        chunk.choices[0]["text"] or ""
        for chunk in _releasing_after_each(iter(stream), frame_per_event_upstream.first_event_read)
    ) == "".join(_TEXT_PIECES)


@pytest.mark.asyncio
async def test_async_openai_text_completion_stream_survives_an_upstream_that_frames_each_event(
    frame_per_event_upstream: _Upstream,
) -> None:
    stream: Final = await litellm.atext_completion(
        model="text-completion-openai/gpt-3.5-turbo-instruct",
        prompt="say hello",
        stream=True,
        api_base=frame_per_event_upstream.api_base,
        api_key="sk-not-a-real-key",
    )

    assert "".join(
        [
            chunk.choices[0]["text"] or ""
            async for chunk in _areleasing_after_each(aiter(stream), frame_per_event_upstream.first_event_read)
        ]
    ) == "".join(_TEXT_PIECES)
