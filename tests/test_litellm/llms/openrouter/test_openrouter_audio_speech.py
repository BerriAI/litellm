"""OpenRouter TTS is an OpenAI-compatible POST /api/v1/audio/speech passthrough."""

from __future__ import annotations

import json
from typing import Final

import httpx
import pytest
import respx

import litellm

OPENROUTER_SPEECH_URL: Final = "https://openrouter.ai/api/v1/audio/speech"
TTS_MODEL: Final = "openrouter/google/gemini-3.1-flash-tts-preview"
UPSTREAM_MODEL: Final = "google/gemini-3.1-flash-tts-preview"


@pytest.fixture(autouse=True)
def _flush_openai_client_cache():
    cache = getattr(litellm, "in_memory_llm_clients_cache", None)
    if cache is not None:
        cache.flush_cache()
    yield
    if cache is not None:
        cache.flush_cache()


def test_speech_openrouter_posts_to_openrouter_speech_endpoint(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    audio_bytes: Final = b"pcm-fake-audio"
    mock_route: Final = respx_mock.post(OPENROUTER_SPEECH_URL).mock(
        return_value=httpx.Response(200, content=audio_bytes, headers={"content-type": "audio/pcm"})
    )

    response: Final = litellm.speech(
        model=TTS_MODEL,
        input="Hello world",
        voice="Kore",
        response_format="pcm",
    )

    assert mock_route.called
    request: Final = mock_route.calls.last.request
    assert json.loads(request.content) == {
        "model": UPSTREAM_MODEL,
        "input": "Hello world",
        "voice": "Kore",
        "response_format": "pcm",
    }
    assert request.headers["authorization"] == "Bearer sk-or-test"
    assert request.headers["http-referer"] == "https://litellm.ai"
    assert request.headers["x-title"] == "liteLLM"
    assert response.content == audio_bytes


def test_speech_openrouter_uses_openrouter_api_base_env(respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENROUTER_API_BASE", "https://openrouter.env.example/api/v1")
    monkeypatch.setattr(litellm, "api_base", None)
    audio_bytes: Final = b"env-base-audio"
    env_route: Final = respx_mock.post("https://openrouter.env.example/api/v1/audio/speech").mock(
        return_value=httpx.Response(200, content=audio_bytes)
    )

    response: Final = litellm.speech(
        model=TTS_MODEL,
        input="Hello world",
        voice="Kore",
    )

    assert env_route.called
    assert response.content == audio_bytes


def test_speech_openrouter_uses_chat_ranking_header_env(respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OR_SITE_URL", "https://example.com")
    monkeypatch.setenv("OR_APP_NAME", "MyApp")
    mock_route: Final = respx_mock.post(OPENROUTER_SPEECH_URL).mock(return_value=httpx.Response(200, content=b"hdr"))

    litellm.speech(model=TTS_MODEL, input="Hello world", voice="Kore")

    request: Final = mock_route.calls.last.request
    assert request.headers["http-referer"] == "https://example.com"
    assert request.headers["x-title"] == "MyApp"


def test_speech_openrouter_uses_configured_api_base(respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    audio_bytes: Final = b"gateway-audio"
    gateway_route: Final = respx_mock.post("https://openrouter.gateway.internal/api/v1/audio/speech").mock(
        return_value=httpx.Response(200, content=audio_bytes)
    )

    response: Final = litellm.speech(
        model=TTS_MODEL,
        input="Hello world",
        voice="Kore",
        api_base="https://openrouter.gateway.internal/api/v1",
    )

    assert gateway_route.called
    assert response.content == audio_bytes


@pytest.mark.asyncio
async def test_aspeech_openrouter_posts_to_openrouter_speech_endpoint(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    audio_bytes: Final = b"async-pcm-audio"
    mock_route: Final = respx_mock.post(OPENROUTER_SPEECH_URL).mock(
        return_value=httpx.Response(200, content=audio_bytes)
    )

    response: Final = await litellm.aspeech(
        model=TTS_MODEL,
        input="Hello world",
        voice="Kore",
        response_format="pcm",
    )

    assert mock_route.called
    assert json.loads(mock_route.calls.last.request.content)["model"] == UPSTREAM_MODEL
    assert response.content == audio_bytes
