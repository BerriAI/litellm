"""Streaming behaviour of the OpenAI and Azure OpenAI ``/audio/speech`` paths."""

from __future__ import annotations

import base64
import json
from typing import Final
from unittest.mock import Mock

import httpx
import pytest
from openai import AsyncAzureOpenAI, AsyncOpenAI, OpenAI

import litellm
from litellm.llms.azure.azure import AzureChatCompletion
from litellm.llms.openai.openai import OpenAIChatCompletion


class _RecordingAudioStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes], produced: list[bytes]):
        self._chunks = chunks
        self._produced = produced

    async def __aiter__(self):
        for chunk in self._chunks:
            self._produced.append(chunk)
            yield chunk


def _audio_client(produced: list[bytes]) -> httpx.AsyncClient:
    def _respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            headers={"content-type": "audio/mpeg"},
            stream=_RecordingAudioStream([b"one", b"two", b"three"], produced),
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(_respond))


async def _openai_speech(produced: list[bytes], *, stream: bool):
    return await OpenAIChatCompletion().async_audio_speech(
        model="tts-1",
        input="hello",
        voice="alloy",
        optional_params={},
        api_key="fake-key",
        api_base=None,
        organization=None,
        project=None,
        max_retries=0,
        timeout=10.0,
        logging_obj=Mock(),
        client=AsyncOpenAI(api_key="fake-key", http_client=_audio_client(produced)),
        stream=stream,
    )


@pytest.mark.asyncio
async def test_openai_audio_speech_yields_chunks_before_upstream_finishes():
    produced: Final[list[bytes]] = []

    response: Final = await _openai_speech(produced, stream=True)
    audio: Final = await response.aiter_bytes()

    assert await anext(audio) == b"one"
    assert produced == [b"one"]


@pytest.mark.asyncio
async def test_openai_audio_speech_buffers_without_stream():
    produced: Final[list[bytes]] = []

    response: Final = await _openai_speech(produced, stream=False)

    assert produced == [b"one", b"two", b"three"]
    assert response.content == b"onetwothree"


@pytest.mark.asyncio
async def test_azure_audio_speech_yields_chunks_before_upstream_finishes():
    produced: Final[list[bytes]] = []

    response: Final = await AzureChatCompletion().async_audio_speech(
        model="tts-1",
        input="hello",
        voice="alloy",
        optional_params={},
        api_key="fake-key",
        api_base="https://fake.openai.azure.com",
        api_version="2024-02-01",
        azure_ad_token=None,
        azure_ad_token_provider=None,
        max_retries=0,
        timeout=10.0,
        logging_obj=Mock(),
        client=AsyncAzureOpenAI(
            api_key="fake-key",
            api_version="2024-02-01",
            azure_endpoint="https://fake.openai.azure.com",
            http_client=_audio_client(produced),
        ),
        stream=True,
    )
    audio: Final = await response.aiter_bytes()

    assert await anext(audio) == b"one"
    assert produced == [b"one"]


_SPEECH_SSE: Final = (
    b'data: {"type":"speech.audio.delta","audio":"%s"}\n\n'
    b'data: {"type":"speech.audio.delta","audio":"%s"}\n\n'
    b'data: {"type":"speech.audio.done","usage":{"input_tokens":6,"output_tokens":50,"total_tokens":56}}\n\n'
    b"data: [DONE]\n\n"
) % (base64.b64encode(b"first-half|"), base64.b64encode(b"second-half"))


class _SpeechSseStream(httpx.SyncByteStream, httpx.AsyncByteStream):
    def __init__(self, frames: list[bytes]):
        self._frames = frames

    def __iter__(self):
        yield from self._frames

    async def __aiter__(self):
        for frame in self._frames:
            yield frame


def _sse_transport(
    requests: list[dict[str, object]], frame_size: int, sse: bytes = _SPEECH_SSE, encodings: list[str] | None = None
) -> httpx.MockTransport:
    def _respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if encodings is not None:
            encodings.append(request.headers["accept-encoding"])
        frames: Final = [sse[start : start + frame_size] for start in range(0, len(sse), frame_size)]
        return httpx.Response(
            status_code=200,
            headers={"content-type": "text/event-stream; charset=utf-8"},
            stream=_SpeechSseStream(frames),
        )

    return httpx.MockTransport(_respond)


async def _token_priced_speech(
    requests: list[dict[str, object]], *, stream: bool, frame_size: int, optional_params: dict[str, object]
):
    return await OpenAIChatCompletion().async_audio_speech(
        model="gpt-4o-mini-tts",
        input="hello",
        voice="alloy",
        optional_params=optional_params,
        api_key="fake-key",
        api_base=None,
        organization=None,
        project=None,
        max_retries=0,
        timeout=10.0,
        logging_obj=Mock(),
        client=AsyncOpenAI(
            api_key="fake-key", http_client=httpx.AsyncClient(transport=_sse_transport(requests, frame_size))
        ),
        stream=stream,
    )


def _audio_output_cost(response: litellm.HttpxBinaryResponseContent) -> float:
    _, completion_cost = litellm.cost_per_token(
        model="gpt-4o-mini-tts", custom_llm_provider="openai", usage_object=response.usage, call_type="aspeech"
    )
    return completion_cost


@pytest.mark.asyncio
@pytest.mark.parametrize("frame_size", [7, 64, 4096])
async def test_streamed_token_priced_speech_returns_audio_and_bills_the_audio_tokens_openai_reports(frame_size: int):
    requests: Final[list[dict[str, object]]] = []

    response: Final = await _token_priced_speech(
        requests, stream=True, frame_size=frame_size, optional_params={"response_format": "wav"}
    )
    audio: Final = b"".join([chunk async for chunk in await response.aiter_bytes()])

    assert audio == b"first-half|second-half"
    assert response.content_type == "audio/wav"
    assert requests == [
        {
            "model": "gpt-4o-mini-tts",
            "voice": "alloy",
            "input": "hello",
            "response_format": "wav",
            "stream_format": "sse",
        }
    ]
    assert response.usage is not None
    assert (response.usage.prompt_tokens, response.usage.completion_tokens) == (6, 50)
    assert _audio_output_cost(response) == pytest.approx(
        50 * litellm.model_cost["gpt-4o-mini-tts"]["output_cost_per_audio_token"]
    )


@pytest.mark.asyncio
async def test_buffered_token_priced_speech_returns_audio_and_bills_the_audio_tokens_openai_reports():
    requests: Final[list[dict[str, object]]] = []

    response: Final = await _token_priced_speech(requests, stream=False, frame_size=64, optional_params={})

    assert response.content == b"first-half|second-half"
    assert response.content_type == "audio/mpeg"
    assert _audio_output_cost(response) == pytest.approx(
        50 * litellm.model_cost["gpt-4o-mini-tts"]["output_cost_per_audio_token"]
    )


def test_sync_streamed_token_priced_speech_returns_audio_and_the_usage_openai_reports():
    requests: Final[list[dict[str, object]]] = []

    response: Final = OpenAIChatCompletion().audio_speech(
        model="gpt-4o-mini-tts",
        input="hello",
        voice="alloy",
        optional_params={},
        api_key="fake-key",
        api_base=None,
        organization=None,
        project=None,
        max_retries=0,
        timeout=10.0,
        logging_obj=Mock(),
        client=OpenAI(api_key="fake-key", http_client=httpx.Client(transport=_sse_transport(requests, 7))),
        stream=True,
    )

    assert b"".join(response.iter_bytes()) == b"first-half|second-half"
    assert response.usage is not None
    assert (response.usage.prompt_tokens, response.usage.completion_tokens) == (6, 50)


@pytest.mark.asyncio
async def test_streamed_speech_drops_the_cost_stamped_before_the_audio_tokens_arrived():
    response: Final = await _token_priced_speech([], stream=True, frame_size=64, optional_params={})
    response.set_response_cost(3.6e-06)

    async for _ in await response.aiter_bytes():
        pass

    assert "response_cost" not in response.hidden_params


@pytest.mark.asyncio
async def test_speech_sse_asked_for_by_the_caller_is_passed_through_untouched():
    requests: Final[list[dict[str, object]]] = []

    response: Final = await _token_priced_speech(
        requests, stream=False, frame_size=4096, optional_params={"stream_format": "sse"}
    )

    assert requests == [{"model": "gpt-4o-mini-tts", "voice": "alloy", "input": "hello", "stream_format": "sse"}]
    assert response.content == _SPEECH_SSE


@pytest.mark.asyncio
async def test_speech_model_priced_per_character_is_not_asked_for_sse():
    requests: Final[list[dict[str, object]]] = []

    def _respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, headers={"content-type": "audio/mpeg"}, content=b"audio")

    response: Final = await OpenAIChatCompletion().async_audio_speech(
        model="tts-1",
        input="hello",
        voice="alloy",
        optional_params={},
        api_key="fake-key",
        api_base=None,
        organization=None,
        project=None,
        max_retries=0,
        timeout=10.0,
        logging_obj=Mock(),
        client=AsyncOpenAI(api_key="fake-key", http_client=httpx.AsyncClient(transport=httpx.MockTransport(_respond))),
        stream=False,
    )

    assert requests == [{"model": "tts-1", "voice": "alloy", "input": "hello"}]
    assert response.content == b"audio"


def _sync_streamed_speech(transport: httpx.MockTransport):
    return OpenAIChatCompletion().audio_speech(
        model="gpt-4o-mini-tts",
        input="hello",
        voice="alloy",
        optional_params={},
        api_key="fake-key",
        api_base=None,
        organization=None,
        project=None,
        max_retries=0,
        timeout=10.0,
        logging_obj=Mock(),
        client=OpenAI(api_key="fake-key", http_client=httpx.Client(transport=transport)),
        stream=True,
    )


def test_streamed_speech_keeps_the_usage_of_a_last_event_with_no_trailing_newline():
    response: Final = _sync_streamed_speech(
        _sse_transport([], 64, sse=_SPEECH_SSE.removesuffix(b"\n\ndata: [DONE]\n\n"))
    )

    assert b"".join(response.iter_bytes()) == b"first-half|second-half"
    assert response.usage is not None
    assert response.usage.completion_tokens == 50


def test_token_priced_speech_asks_the_provider_not_to_compress_the_events_it_decodes():
    encodings: Final[list[str]] = []

    response: Final = _sync_streamed_speech(_sse_transport([], 64, encodings=encodings))

    assert b"".join(response.iter_bytes()) == b"first-half|second-half"
    assert encodings == ["identity"]


_SECONDS: Final = 3
_REPORTED_TOKENS: Final = 50
# OpenAI's usage API billed gpt-4o-mini-tts 20 audio tokens per second of audio on 2026-11-20, one request per row
_BILLED_TOKENS: Final = 20 * _SECONDS
_PCM: Final = bytes(24000 * 2 * _SECONDS)
_WAV: Final = (
    b"RIFF\xff\xff\xff\xffWAVEfmt "
    + (16).to_bytes(4, "little")
    + (1).to_bytes(2, "little") * 2
    + (24000).to_bytes(4, "little")
    + (48000).to_bytes(4, "little")
    + (2).to_bytes(2, "little")
    + (16).to_bytes(2, "little")
    + b"data\xff\xff\xff\xff"
    + _PCM
)
_MP3: Final = (b"\xff\xf3\xc4\xc4" + bytes(380)) * (24000 * _SECONDS // 576)
_OPUS_PRE_SKIP: Final = 4800


def _ogg_page(granule: int, body: bytes) -> bytes:
    return b"OggS\x00\x00" + granule.to_bytes(8, "little") + bytes(12) + bytes([1, len(body)]) + body


_OPUS: Final = _ogg_page(0, b"OpusHead\x01\x01" + _OPUS_PRE_SKIP.to_bytes(2, "little") + bytes(7)) + _ogg_page(
    48000 * _SECONDS + _OPUS_PRE_SKIP, bytes(40)
)


def _speech_sse(audio: bytes) -> bytes:
    return b'data: {"type":"speech.audio.delta","audio":"%s"}\n\ndata: {"type":"speech.audio.done","usage":%s}\n\n' % (
        base64.b64encode(audio),
        json.dumps({"input_tokens": 6, "output_tokens": _REPORTED_TOKENS, "total_tokens": 56}).encode(),
    )


def _billed_audio_tokens(audio: bytes, optional_params: dict[str, object]) -> int:
    sse: Final = _speech_sse(audio)
    response: Final = OpenAIChatCompletion().audio_speech(
        model="gpt-4o-mini-tts",
        input="hello",
        voice="alloy",
        optional_params=optional_params,
        api_key="fake-key",
        api_base=None,
        organization=None,
        project=None,
        max_retries=0,
        timeout=10.0,
        logging_obj=Mock(),
        client=OpenAI(api_key="fake-key", http_client=httpx.Client(transport=_sse_transport([], 4096, sse=sse))),
        stream=True,
    )
    assert b"".join(response.iter_bytes()) == audio
    assert response.usage is not None
    assert response.usage.completion_tokens_details.audio_tokens == response.usage.completion_tokens
    assert response.usage.total_tokens == 6 + response.usage.completion_tokens
    return response.usage.completion_tokens


@pytest.mark.parametrize(
    ("audio", "optional_params", "billed"),
    [
        pytest.param(_PCM, {"response_format": "pcm"}, _BILLED_TOKENS, id="pcm"),
        pytest.param(_WAV, {"response_format": "wav"}, _BILLED_TOKENS, id="wav"),
        pytest.param(_OPUS, {"response_format": "opus"}, _BILLED_TOKENS, id="opus"),
        pytest.param(
            _OPUS[:-40] + b"OggS" + bytes(36),
            {"response_format": "opus"},
            _BILLED_TOKENS,
            id="opus whose last packet holds the bytes that start a page",
        ),
        pytest.param(_PCM[:50352], {"response_format": "pcm", "speed": 4.0}, 83, id="rounds once after speed"),
        pytest.param(_PCM[:48000], {"response_format": "pcm", "speed": 0.3}, 6, id="a decimal speed is exact"),
        pytest.param(_MP3, {"response_format": "mp3"}, _BILLED_TOKENS - 1, id="mp3 drops the encoder delay"),
        pytest.param(_MP3, {}, _BILLED_TOKENS - 1, id="mp3 is the default format"),
        pytest.param(_PCM, {"response_format": "pcm", "speed": 2.0}, _BILLED_TOKENS * 2, id="sped up audio"),
        pytest.param(_PCM, {"response_format": "pcm", "speed": 0.5}, _BILLED_TOKENS // 2, id="slowed audio"),
    ],
)
def test_token_priced_speech_bills_the_length_of_the_audio_not_the_tokens_the_done_event_reports(
    audio: bytes, optional_params: dict[str, object], billed: int
):
    assert _billed_audio_tokens(audio, optional_params) == billed


@pytest.mark.parametrize(
    ("audio", "response_format"),
    [
        pytest.param(_PCM, "aac", id="a format whose length is not read"),
        pytest.param(_PCM, "wav", id="wav without its header"),
        pytest.param(_MP3 + b"\x00", "mp3", id="mp3 that is not whole frames"),
        pytest.param(_OPUS.replace(b"OpusHead", b"OpusTags"), "opus", id="ogg without an opus header"),
    ],
)
def test_token_priced_speech_falls_back_to_the_reported_tokens_when_the_audio_length_is_unreadable(
    audio: bytes, response_format: str
):
    assert _billed_audio_tokens(audio, {"response_format": response_format}) == _REPORTED_TOKENS


@pytest.mark.parametrize("speed", [0.1, 4.5, 0, -1, True, "2"])
def test_token_priced_speech_falls_back_to_the_reported_tokens_for_a_speed_the_provider_does_not_accept(
    speed: object,
):
    assert _billed_audio_tokens(_PCM, {"response_format": "pcm", "speed": speed}) == _REPORTED_TOKENS


@pytest.mark.parametrize("speed", [0.25, 4.0])
def test_token_priced_speech_scales_by_the_slowest_and_fastest_speed_the_provider_accepts(speed: float):
    assert _billed_audio_tokens(_PCM, {"response_format": "pcm", "speed": speed}) == int(_BILLED_TOKENS * speed)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
async def test_async_token_priced_speech_bills_the_length_of_the_audio_streamed_or_buffered(stream: bool):
    response: Final = await OpenAIChatCompletion().async_audio_speech(
        model="gpt-4o-mini-tts",
        input="hello",
        voice="alloy",
        optional_params={"response_format": "wav"},
        api_key="fake-key",
        api_base=None,
        organization=None,
        project=None,
        max_retries=0,
        timeout=10.0,
        logging_obj=Mock(),
        client=AsyncOpenAI(
            api_key="fake-key",
            http_client=httpx.AsyncClient(transport=_sse_transport([], 4096, sse=_speech_sse(_WAV))),
        ),
        stream=stream,
    )
    audio: Final = b"".join([chunk async for chunk in await response.aiter_bytes()]) if stream else response.content

    assert audio == _WAV
    assert response.usage is not None
    assert response.usage.completion_tokens == _BILLED_TOKENS
