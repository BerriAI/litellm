"""
A JSON-configured provider may use the OpenAI `/v1/audio/*` transport only when it declares
that endpoint for itself, so `transcription()` and `speech()` cannot carry `OPENAI_API_KEY`
to a provider whose entry in `providers.json` advertises chat only.
"""

import io
from collections.abc import Sequence
from typing import Final

import pytest
import respx

import litellm
from litellm.constants import openai_compatible_providers
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.openai_like.json_loader import (
    OPENAI_AUDIO_ENDPOINTS,
    OPENAI_AUDIO_TRANSCRIPTION_PROVIDERS,
    JSONProviderRegistry,
    derive_openai_audio_transcription_providers,
)

OPENAI_KEY_CANARY: Final = "sk-leaked-openai-key"


def _declares_audio(endpoints: Sequence[str]) -> bool:
    return any(endpoint in OPENAI_AUDIO_ENDPOINTS for endpoint in endpoints)


def _chat_only_json_provider() -> str:
    """A JSON provider that is reachable as OpenAI-compatible yet declares no audio endpoint"""
    return next(
        slug
        for slug, endpoints in JSONProviderRegistry.declared_endpoints().items()
        if slug in openai_compatible_providers and not _declares_audio(endpoints)
    )


def _audio_file() -> io.BytesIO:
    file: Final = io.BytesIO(b"riff-bytes")
    file.name = "audio.wav"
    return file


def test_json_provider_is_audio_capable_only_when_it_declares_audio():
    for slug, endpoints in JSONProviderRegistry.declared_endpoints().items():
        assert (slug in OPENAI_AUDIO_TRANSCRIPTION_PROVIDERS) is _declares_audio(endpoints), slug


def test_python_openai_compatible_providers_keep_the_audio_transport():
    for provider in openai_compatible_providers:
        if not JSONProviderRegistry.exists(provider):
            assert provider in OPENAI_AUDIO_TRANSCRIPTION_PROVIDERS, provider


@pytest.mark.parametrize(
    ["declared", "expected"],
    [
        (["/v1/audio/transcriptions"], True),
        (["/v1/audio/speech"], True),
        (["/v1/chat/completions", "/v1/audio/speech"], True),
        (["/v1/chat/completions", "/v1/responses"], False),
        ([], False),
    ],
    ids=["transcriptions", "speech", "audio_plus_chat", "chat_only", "nothing_declared"],
)
def test_derivation_grants_membership_on_declared_audio_endpoints_only(declared: list[str], expected: bool):
    derived: Final = derive_openai_audio_transcription_providers(
        compatible_providers=["groq", "chat-only"],
        declared_endpoints={"chat-only": ["/v1/chat/completions"], "voiced": declared},
    )

    assert ("voiced" in derived) is expected
    assert "groq" in derived
    assert "chat-only" not in derived
    assert "openai" in derived


def test_transcription_sends_no_openai_key_to_json_provider_without_audio_endpoint(monkeypatch):
    slug: Final = _chat_only_json_provider()
    monkeypatch.delenv(JSONProviderRegistry.get(slug).api_key_env, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY_CANARY)

    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.post(f"{JSONProviderRegistry.get(slug).base_url}/audio/transcriptions").respond(
            200, json={"text": "should never be reached"}
        )

        with pytest.raises(ValueError, match="Unmapped provider"):
            litellm.transcription(model=f"{slug}/whisper-1", file=_audio_file())

        assert route.called is False
        assert len(upstream.calls) == 0


def test_speech_sends_no_openai_key_to_json_provider_without_audio_endpoint(monkeypatch):
    slug: Final = _chat_only_json_provider()
    monkeypatch.delenv(JSONProviderRegistry.get(slug).api_key_env, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY_CANARY)

    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.post(f"{JSONProviderRegistry.get(slug).base_url}/audio/speech").respond(
            200, content=b"should never be reached"
        )

        with pytest.raises(Exception, match=f"Unable to map the custom llm provider={slug}"):
            litellm.speech(model=f"{slug}/tts-1", input="hello", voice="alloy")

        assert route.called is False
        assert len(upstream.calls) == 0


def test_transcription_still_routes_a_python_provider_with_its_own_credential(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-provider-key")
    _, _, _, api_base = get_llm_provider(
        model="groq/whisper-large-v3-turbo", custom_llm_provider=None, api_base=None, api_key=None
    )

    with respx.mock() as upstream:
        route: Final = upstream.post(f"{api_base}/audio/transcriptions").respond(
            200, json={"text": "hello world", "language": "english"}
        )

        response: Final = litellm.transcription(model="groq/whisper-large-v3-turbo", file=_audio_file())

        assert response.text == "hello world"
        assert route.calls[0].request.headers["authorization"] == "Bearer gsk-provider-key"
