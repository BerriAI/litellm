import json
from typing import Final

import httpx
import pytest
import respx
from openai import AsyncOpenAI

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

LOKUTOR_API_BASE: Final = "https://api.lokutor.com/v1"
WAV_BYTES: Final = b"RIFF\x24\x00\x00\x00WAVEfmt "


@pytest.fixture(autouse=True)
def _lokutor_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOKUTOR_API_KEY", "lokutor-test-key")
    monkeypatch.delenv("LOKUTOR_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_prefixed_model_resolves_to_lokutor_base_and_env_key():
    model, provider, api_key, api_base = get_llm_provider(model="lokutor/tts-1")

    assert (model, provider, api_key, api_base) == ("tts-1", "lokutor", "lokutor-test-key", LOKUTOR_API_BASE)


def test_lokutor_is_a_registered_openai_compatible_provider():
    assert "lokutor" in litellm.openai_compatible_providers
    assert litellm.LlmProviders("lokutor") is litellm.LlmProviders.LOKUTOR


def test_speech_posts_to_lokutor_audio_speech_with_lokutor_key():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{LOKUTOR_API_BASE}/audio/speech").respond(
            200, content=WAV_BYTES, headers={"content-type": "audio/wav"}
        )

        response: Final = litellm.speech(
            model="lokutor/tts-1",
            voice="alloy",
            input="Hello from Lokutor.",
        )

    assert response.content == WAV_BYTES
    request: Final = route.calls.last.request
    assert request.headers["authorization"] == "Bearer lokutor-test-key"
    assert json.loads(request.content) == {
        "model": "tts-1",
        "input": "Hello from Lokutor.",
        "voice": "alloy",
    }


def test_explicit_api_key_wins_over_the_provider_env_key():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{LOKUTOR_API_BASE}/audio/speech").respond(200, content=WAV_BYTES)

        litellm.speech(model="lokutor/tts-1", voice="F1", input="Hi", api_key="explicit-key")

    assert route.calls.last.request.headers["authorization"] == "Bearer explicit-key"


def test_transcription_posts_multipart_to_lokutor_audio_transcriptions():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{LOKUTOR_API_BASE}/audio/transcriptions").respond(
            200, json={"text": "hello from lokutor"}
        )

        response: Final = litellm.transcription(
            model="lokutor/whisper-1",
            file=("hello.wav", WAV_BYTES, "audio/wav"),
            language="en",
        )

    assert response.text == "hello from lokutor"
    request: Final = route.calls.last.request
    assert request.headers["authorization"] == "Bearer lokutor-test-key"
    assert request.headers["content-type"].startswith("multipart/form-data")
    assert b'name="language"' in request.content
    assert b"\r\n\r\nen\r\n" in request.content
    assert b"hello.wav" in request.content


@pytest.mark.asyncio
async def test_async_speech_uses_lokutor_endpoint():
    requests: Final[list[httpx.Request]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=WAV_BYTES, headers={"content-type": "audio/wav"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client:
        client: Final = AsyncOpenAI(api_key="lokutor-test-key", base_url=LOKUTOR_API_BASE, http_client=http_client)
        response: Final = await litellm.aspeech(model="lokutor/tts-1", voice="nova", input="Hola", client=client)

    assert response.content == WAV_BYTES
    assert str(requests[0].url) == f"{LOKUTOR_API_BASE}/audio/speech"


def _only_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOKUTOR_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-must-not-leave")
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.setattr(litellm, "openai_key", None)


def test_speech_without_a_lokutor_key_never_sends_the_openai_key(monkeypatch: pytest.MonkeyPatch):
    _only_openai_key(monkeypatch)
    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.post(f"{LOKUTOR_API_BASE}/audio/speech").respond(200, content=WAV_BYTES)

        with pytest.raises(litellm.AuthenticationError):
            litellm.speech(model="lokutor/tts-1", voice="alloy", input="Hi")

    assert not route.called


def test_transcription_without_a_lokutor_key_never_sends_the_openai_key(monkeypatch: pytest.MonkeyPatch):
    _only_openai_key(monkeypatch)
    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.post(f"{LOKUTOR_API_BASE}/audio/transcriptions").respond(200, json={"text": "x"})

        with pytest.raises(litellm.AuthenticationError):
            litellm.transcription(model="lokutor/whisper-1", file=("hello.wav", WAV_BYTES, "audio/wav"))

    assert not route.called
