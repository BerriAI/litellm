from collections.abc import AsyncIterator, Iterator, Mapping
from threading import Event
from typing import Final

import httpx
import pytest

import litellm
from litellm import aspeech, speech  # pyright: ignore[reportUnknownVariableType]  # public SDK accepts provider kwargs
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.llms.openai import HttpxBinaryResponseContent

_MODEL: Final = "gnani/timbre-v3.0"
_CHUNK: Final = b'event: chunk\r\ndata: {"audio":"AP8="}\r\n\r\n'
_COMPLETE: Final = b'event: complete\r\ndata: {"is_final":true,"audio":""}\r\n\r\n'


def test_audio_is_delivered_before_the_next_event_and_stops_at_completion() -> None:
    advance: Final = Event()

    def events() -> Iterator[bytes]:
        yield b": heartbeat\r\n\r\nevent: start\r\ndata: {}\r\n\r\n" + _CHUNK[:18]
        yield _CHUNK[18:]
        assert advance.is_set(), "The caller must receive audio before another event is read"
        yield _CHUNK + _COMPLETE
        pytest.fail("A complete event must release a still-open provider stream")

    upstream: Final = httpx.Response(200, headers={"content-type": "text/event-stream"}, content=events())

    def synthesize(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/tts/sse"
        assert request.headers["X-API-Key-ID"] == "key"
        return upstream

    with httpx.Client(transport=httpx.MockTransport(synthesize)) as client:
        result: Final = speech(
            model=_MODEL,
            voice="Shivani",
            input="Read aloud.",
            stream=True,
            api_key="key",
            response_format="pcm",
            client=HTTPHandler(client=client),
        )
        assert isinstance(result, HttpxBinaryResponseContent)
        assert result.content_type == "audio/pcm"
        chunks: Final = result.iter_bytes()
        assert next(chunks) == b"\x00\xff"
        advance.set()
        assert tuple(chunks) == (b"\x00\xff",)
        assert upstream.is_closed


@pytest.mark.parametrize(
    ("tail", "message"),
    (
        (b'event: error\ndata: {"type":"error","message":"Synthesis failed"}\n\n', "Synthesis failed"),
        (b'event: chunk\ndata: {"audio":"@@"}\n\n', "malformed"),
        (b"event: chunk\ndata: {\n\n", "malformed"),
        (b'event: chunk\ndata: {"audio":null}\n\n', "without an audio"),
        (b"", "before the complete"),
    ),
)
async def test_partial_audio_is_followed_by_a_stream_error_and_cleanup(tail: bytes, message: str) -> None:
    async def events() -> AsyncIterator[bytes]:
        yield _CHUNK + tail

    upstream: Final = httpx.Response(200, headers={"content-type": "text/event-stream"}, content=events())
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: upstream)) as client:
        result: Final = await aspeech(
            model=_MODEL,
            voice="Shivani",
            input="Read aloud.",
            stream=True,
            api_key="key",
            client=AsyncHTTPHandler(client=client),
        )
        chunks: Final = await result.aiter_bytes()
        assert await anext(chunks) == b"\x00\xff"
        with pytest.raises(litellm.BadGatewayError, match=message):
            await anext(chunks)
        assert upstream.is_closed


@pytest.mark.parametrize("read_first", (False, True))
async def test_caller_can_release_a_stream_without_draining_it(read_first: bool) -> None:
    async def events() -> AsyncIterator[bytes]:
        yield _CHUNK
        pytest.fail("Closing the response must not drain it")

    upstream: Final = httpx.Response(200, headers={"content-type": "text/event-stream"}, content=events())
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: upstream)) as client:
        result: Final = await aspeech(
            model=_MODEL,
            voice="Shivani",
            input="Read aloud.",
            stream=True,
            api_key="key",
            client=AsyncHTTPHandler(client=client),
        )
        if read_first:
            assert await anext(await result.aiter_bytes()) == b"\x00\xff"
        await result.aclose()
        assert upstream.is_closed


@pytest.mark.parametrize(
    ("config", "mime"),
    (
        ({}, "audio/pcm"),
        ({"container": "wav"}, "audio/wav"),
        ({"container": "mp3"}, "audio/mpeg"),
        ({"container": "raw", "encoding": "pcm_alaw"}, "audio/alaw"),
        ({"container": "mulaw"}, "audio/basic"),
        ({"encoding": "oggopus"}, "audio/ogg"),
        ({"container": "ogg", "encoding": "pcm_mulaw"}, "audio/ogg"),
    ),
)
async def test_stream_labels_the_audio_with_its_output_media_type(config: Mapping[str, str], mime: str) -> None:
    def synthesize(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://example.test/route/api/v1/tts/sse"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_CHUNK + _COMPLETE)

    async with httpx.AsyncClient(transport=httpx.MockTransport(synthesize)) as client:
        result: Final = await aspeech(
            model=_MODEL,
            input="Hello",
            voice="Shivani",
            api_key="key",
            api_base="https://example.test/route/api/v1/tts/inference",
            stream=True,
            extra_body={"audio_config": config},
            client=AsyncHTTPHandler(client=client),
        )
        assert result.content_type == mime
        assert await result.aread() == b"\x00\xff"


@pytest.mark.parametrize(
    ("mime", "content", "message"),
    (("application/json", b"{}", "non-SSE"), ("text/event-stream", _COMPLETE, "without sending audio")),
)
async def test_invalid_upstream_success_is_reported_as_a_gateway_error(mime: str, content: bytes, message: str) -> None:
    async def events() -> AsyncIterator[bytes]:
        yield content

    upstream: Final = httpx.Response(200, headers={"content-type": mime}, content=events())
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: upstream)) as client:
        request: Final = aspeech(
            model=_MODEL,
            input="Hello",
            voice="Shivani",
            api_key="key",
            stream=True,
            client=AsyncHTTPHandler(client=client),
        )
        if mime == "text/event-stream":
            result: Final = await request
            with pytest.raises(litellm.BadGatewayError, match=message):
                await result.aread()
        else:
            with pytest.raises(litellm.BadGatewayError, match=message):
                await request
        assert upstream.is_closed
