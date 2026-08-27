import base64
import json

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.flowspeech.text_to_speech.transformation import (
    FlowSpeechException,
    FlowSpeechTextToSpeechConfig,
)


def test_maps_voice_and_instructions_to_flowspeech_request():
    config = FlowSpeechTextToSpeechConfig()
    voice, params = config.map_openai_params(
        model="flowspeech-tts",
        optional_params={"instructions": "Speak with calm confidence"},
        voice={"voiceName": "Aoede"},
    )

    request = config.transform_text_to_speech_request(
        model="flowspeech-tts",
        input="Hello from FlowSpeech",
        voice=voice,
        optional_params=params,
        litellm_params={},
        headers={},
    )

    assert request["dict_body"] == {
        "text": "Hello from FlowSpeech",
        "originalText": "Hello from FlowSpeech",
        "speakers": [{"voiceName": "Aoede"}],
        "prompt": "Speak with calm confidence",
    }


def test_uses_default_voice_and_ignores_unsupported_openai_params():
    config = FlowSpeechTextToSpeechConfig()
    voice, params = config.map_openai_params(
        model="flowspeech-tts",
        optional_params={"response_format": "mp3", "speed": 1.5},
    )

    assert voice == "Kore"
    assert params == {}


def test_validates_bearer_api_key():
    config = FlowSpeechTextToSpeechConfig()

    headers = config.validate_environment({}, "flowspeech-tts", api_key="test-key")

    assert headers == {
        "Authorization": "Bearer test-key",
        "Content-Type": "application/json",
    }


def test_requires_api_key(monkeypatch):
    config = FlowSpeechTextToSpeechConfig()
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.delenv("FLOWSPEECH_API_KEY", raising=False)

    with pytest.raises(ValueError, match="FlowSpeech API key is required"):
        config.validate_environment({}, "flowspeech-tts")


def test_decodes_flowspeech_audio_response():
    config = FlowSpeechTextToSpeechConfig()
    audio = b"test audio bytes"
    response = httpx.Response(
        200,
        headers={"content-type": "application/json", "x-request-id": "request-1"},
        json={
            "code": 0,
            "data": {
                "mimeType": "audio/mpeg",
                "audioBase64": base64.b64encode(audio).decode(),
            },
        },
        request=httpx.Request("POST", "https://flowspeech.io/api/ai/text-to-speech"),
    )

    result = config.transform_text_to_speech_response("flowspeech-tts", response, logging_obj=None)

    assert result.content == audio
    assert result.response.headers["content-type"] == "audio/mpeg"
    assert result.response.headers["x-request-id"] == "request-1"


def test_rejects_api_error_response():
    config = FlowSpeechTextToSpeechConfig()
    response = httpx.Response(
        400,
        json={"code": 4001, "message": "Quota exceeded", "data": None},
        request=httpx.Request("POST", "https://flowspeech.io/api/ai/text-to-speech"),
    )

    with pytest.raises(FlowSpeechException, match="Quota exceeded"):
        config.transform_text_to_speech_response("flowspeech-tts", response, logging_obj=None)


def test_builds_default_and_custom_urls():
    config = FlowSpeechTextToSpeechConfig()

    assert config.get_complete_url("flowspeech-tts", None, {}) == "https://flowspeech.io/api/ai/text-to-speech"
    assert (
        config.get_complete_url(
            "flowspeech-tts",
            "https://example.com/api/ai/text-to-speech",
            {},
        )
        == "https://example.com/api/ai/text-to-speech"
    )


def test_registers_flowspeech_provider_and_config():
    from litellm.utils import ProviderConfigManager

    model, provider, _, _ = litellm.get_llm_provider("flowspeech/flowspeech-tts")
    config = ProviderConfigManager.get_provider_text_to_speech_config(
        model=model,
        provider=litellm.LlmProviders.FLOWSPEECH,
    )

    assert model == "flowspeech-tts"
    assert provider == "flowspeech"
    assert isinstance(config, FlowSpeechTextToSpeechConfig)


def test_speech_dispatches_to_flowspeech_handler():
    audio = b"generated audio"

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://flowspeech.io/api/ai/text-to-speech"
        assert request.headers["authorization"] == "Bearer test-key"
        assert json.loads(request.content) == {
            "text": "Hello",
            "originalText": "Hello",
            "speakers": [{"voiceName": "Aoede"}],
            "prompt": "Speak warmly",
        }
        return httpx.Response(
            200,
            json={
                "code": 0,
                "data": {
                    "mimeType": "audio/mpeg",
                    "audioBase64": base64.b64encode(audio).decode(),
                },
            },
        )

    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond)))
    result = litellm.speech(
        model="flowspeech/flowspeech-tts",
        input="Hello",
        voice="Aoede",
        instructions="Speak warmly",
        api_key="test-key",
        client=client,
    )

    assert result.content == audio
    assert result.response.headers["content-type"] == "audio/mpeg"
