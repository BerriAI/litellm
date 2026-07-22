"""Unit tests for streaming Text-to-Speech (OpenAI ``stream_format="sse"``).

Regression coverage for #33974: streaming TTS must open the provider request in
streaming mode and forward ``speech.audio.delta`` SSE frames incrementally, instead
of buffering the full clip (which made proxy TTFT == full generation time).
"""

from typing import AsyncIterator, Iterator, cast
from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm.llms.openai.openai import OpenAIChatCompletion
from litellm.types.llms.openai import SpeechStreamingResponse


class _MockAsyncResponse:
    headers = {"content-type": "text/event-stream"}

    async def iter_bytes(self):
        yield b'event: speech.audio.delta\ndata: {"audio":"AAA"}\n\n'
        yield b"event: speech.audio.done\ndata: {}\n\n"


class _MockAsyncResponseCM:
    def __init__(self):
        self.entered = False
        self.exc_info = None

    async def __aenter__(self):
        self.entered = True
        return _MockAsyncResponse()

    async def __aexit__(self, exc_type, exc, tb):
        self.exc_info = (exc_type, exc, tb)


def _mock_async_client(response_cm, captured):
    speech = MagicMock()

    def _create(**kwargs):
        captured.update(kwargs)
        return response_cm

    speech.with_streaming_response.create = _create
    client = MagicMock()
    client.audio.speech = speech
    return client


@pytest.mark.asyncio
async def test_async_audio_speech_streaming_forwards_sse_frames():
    """The streaming path opens with_streaming_response and yields the SSE frames verbatim."""
    captured: dict = {}
    response_cm = _MockAsyncResponseCM()
    api = OpenAIChatCompletion()

    result = await api.async_audio_speech_streaming(
        model="gpt-4o-mini-tts",
        input="hello",
        voice="alloy",
        optional_params={"stream_format": "sse"},
        openai_client=_mock_async_client(response_cm, captured),  # type: ignore[arg-type]
    )

    assert isinstance(result, SpeechStreamingResponse)
    assert response_cm.entered is True
    # stream_format must reach the SDK create() call
    assert captured["stream_format"] == "sse"
    assert captured["model"] == "gpt-4o-mini-tts"

    stream = cast(AsyncIterator[bytes], result.stream_iterator)
    frames = [chunk async for chunk in stream]
    assert frames[0].startswith(b"event: speech.audio.delta")
    assert b"speech.audio.done" in frames[1]
    assert result.headers["content-type"] == "text/event-stream"
    # context manager closed after the stream is exhausted
    assert response_cm.exc_info == (None, None, None)


@pytest.mark.asyncio
async def test_async_audio_speech_streaming_closes_cm_on_error():
    """An error mid-stream must propagate and still close the streaming context manager."""

    class _RaisingResponse:
        headers: dict = {}

        async def iter_bytes(self):
            yield b"a"
            raise RuntimeError("stream failed")

    class _RaisingCM:
        def __init__(self):
            self.exc_info = None

        async def __aenter__(self):
            return _RaisingResponse()

        async def __aexit__(self, exc_type, exc, tb):
            self.exc_info = (exc_type, exc, tb)

    cm = _RaisingCM()
    client = MagicMock()
    client.audio.speech.with_streaming_response.create = lambda **kw: cm
    api = OpenAIChatCompletion()

    result = await api.async_audio_speech_streaming(
        model="gpt-4o-mini-tts",
        input="hi",
        voice="alloy",
        optional_params={"stream_format": "sse"},
        openai_client=client,  # type: ignore[arg-type]
    )
    stream = cast(AsyncIterator[bytes], result.stream_iterator)
    assert await stream.__anext__() == b"a"
    with pytest.raises(RuntimeError, match="stream failed"):
        await stream.__anext__()
    assert cm.exc_info is not None and cm.exc_info[0] is RuntimeError


def test_sync_audio_speech_streaming_forwards_sse_frames():
    """Sync path mirrors the async one via with_streaming_response."""

    class _SyncResponse:
        headers = {"content-type": "text/event-stream"}

        def iter_bytes(self) -> Iterator[bytes]:
            yield b"event: speech.audio.delta\ndata: {}\n\n"

    class _SyncCM:
        def __enter__(self):
            return _SyncResponse()

        def __exit__(self, *a):
            return None

    client = MagicMock()
    client.audio.speech.with_streaming_response.create = lambda **kw: _SyncCM()
    api = OpenAIChatCompletion()

    result = api.audio_speech_streaming(
        model="gpt-4o-mini-tts",
        input="hi",
        voice="alloy",
        optional_params={"stream_format": "sse"},
        openai_client=client,  # type: ignore[arg-type]
    )
    frames = list(cast(Iterator[bytes], result.stream_iterator))
    assert frames == [b"event: speech.audio.delta\ndata: {}\n\n"]


def test_speech_threads_stream_format_into_optional_params():
    """litellm.speech must forward stream_format to the provider (it was previously dropped)."""
    captured: dict = {}

    def _spy_audio_speech(**kwargs):
        captured.update(kwargs)
        return MagicMock(spec=litellm.llms.openai.openai.HttpxBinaryResponseContent)

    with patch(
        "litellm.main.openai_chat_completions.audio_speech",
        side_effect=_spy_audio_speech,
    ):
        litellm.speech(
            model="gpt-4o-mini-tts",
            input="hi",
            voice="alloy",
            stream_format="sse",
            api_key="sk-test",
        )

    assert captured["optional_params"].get("stream_format") == "sse"


@pytest.mark.asyncio
async def test_async_audio_speech_streams_plain_audio_without_stream_format():
    """stream_audio=True must stream even without stream_format="sse" (the common case).

    OpenAI's /v1/audio/speech streams audio bytes over chunked transfer for every request, so
    the proxy must forward incrementally to match calling OpenAI directly with
    with_streaming_response. Regression: streaming was previously gated on stream_format="sse",
    so a plain audio request still buffered the full clip."""
    captured: dict = {}
    response_cm = _MockAsyncResponseCM()
    api = OpenAIChatCompletion()

    result = await api.async_audio_speech(
        model="gpt-4o-mini-tts",
        input="hello",
        voice="alloy",
        optional_params={},  # no stream_format
        api_key="sk-test",
        api_base=None,
        organization=None,
        project=None,
        max_retries=0,
        timeout=600,
        stream_audio=True,
        client=_mock_async_client(response_cm, captured),
    )

    assert isinstance(result, SpeechStreamingResponse)
    assert response_cm.entered is True
    assert "stream_format" not in captured  # plain audio, no sse requested


@pytest.mark.asyncio
async def test_async_audio_speech_buffers_when_stream_audio_false():
    """stream_audio=False must keep the buffered HttpxBinaryResponseContent contract (SDK back-compat)."""
    from litellm.types.llms.openai import HttpxBinaryResponseContent

    created: dict = {}

    async def _create(**kwargs):
        created.update(kwargs)
        inner = MagicMock()
        inner.response = MagicMock()
        return inner

    client = MagicMock()
    client.audio.speech.create = _create
    # with_streaming_response must NOT be used on the buffered path
    client.audio.speech.with_streaming_response.create = MagicMock(
        side_effect=AssertionError("must not open a streaming request when stream_audio=False")
    )
    api = OpenAIChatCompletion()

    result = await api.async_audio_speech(
        model="gpt-4o-mini-tts",
        input="hello",
        voice="alloy",
        optional_params={},
        api_key="sk-test",
        api_base=None,
        organization=None,
        project=None,
        max_retries=0,
        timeout=600,
        stream_audio=False,
        client=client,
    )

    assert isinstance(result, HttpxBinaryResponseContent)
    assert created["model"] == "gpt-4o-mini-tts"


def test_speech_forwards_stream_audio_for_openai_compatible_provider():
    """stream_audio must reach the OpenAI handler for openai-compatible providers like hosted_vllm,
    not just custom_llm_provider="openai". Regression: voice-agents point at qwen3-tts served as
    hosted_vllm and need the same transparent default-path streaming as OpenAI."""
    captured: dict = {}

    def _spy_audio_speech(**kwargs):
        captured.update(kwargs)
        return MagicMock(spec=litellm.llms.openai.openai.HttpxBinaryResponseContent)

    with patch(
        "litellm.main.openai_chat_completions.audio_speech",
        side_effect=_spy_audio_speech,
    ):
        litellm.speech(
            model="hosted_vllm/qwen3-tts",
            input="hi",
            voice="alloy",
            stream_audio=True,
            api_base="http://vllm.local/v1",
            api_key="sk-test",
        )

    assert captured.get("stream_audio") is True
