"""Streaming behaviour of the OpenAI and Azure OpenAI ``/audio/speech`` paths."""

from __future__ import annotations

from typing import Final
from unittest.mock import Mock

import httpx
import pytest
from openai import AsyncAzureOpenAI, AsyncOpenAI

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
