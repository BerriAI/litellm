import httpx
import pytest

import litellm
from litellm.llms.base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
)
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.xai.audio_transcription.transformation import (
    XAIAudioTranscriptionConfig,
)
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

CONFIG = XAIAudioTranscriptionConfig()

WAV_BYTES = b"RIFF" + b"\x00" * 64


def test_transform_request_serializes_provider_params():
    result = CONFIG.transform_audio_transcription_request(
        model="grok-voice-transcribe-2.0",
        audio_file=WAV_BYTES,
        optional_params={
            "language": "en",
            "diarize": True,
            "keyterm": ["LiteLLM", "Grok"],
        },
        litellm_params={},
    )

    assert isinstance(result, AudioTranscriptionRequestData)
    data = result.data
    assert data["model"] == "grok-voice-transcribe-2.0"
    assert data["language"] == "en"
    assert data["diarize"] == "true"
    assert data["keyterm"] == ["LiteLLM", "Grok"]
    filename, content, content_type = result.files["file"]
    assert content == WAV_BYTES
    assert isinstance(filename, str)
    assert isinstance(content_type, str)


def test_transform_request_flattens_extra_body():
    result = CONFIG.transform_audio_transcription_request(
        model="grok-voice-transcribe-1.0",
        audio_file=WAV_BYTES,
        optional_params={
            "language": "en",
            "extra_body": {"diarize": False, "channels": 2},
        },
        litellm_params={},
    )
    assert result.data["diarize"] == "false"
    assert result.data["channels"] == "2"
    assert "extra_body" not in result.data


@pytest.mark.parametrize(
    "api_base,expected",
    [
        (None, "https://api.x.ai/v1/stt"),
        ("https://api.x.ai/v1", "https://api.x.ai/v1/stt"),
        ("https://api.x.ai/v1/", "https://api.x.ai/v1/stt"),
        ("https://proxy.example/", "https://proxy.example/v1/stt"),
    ],
)
def test_get_complete_url(api_base, expected):
    url = CONFIG.get_complete_url(
        api_base=api_base,
        api_key=None,
        model="grok-voice-transcribe-2.0",
        optional_params={},
        litellm_params={},
    )
    assert url == expected


def test_validate_environment_sets_bearer_header():
    headers = CONFIG.validate_environment(
        headers={},
        model="grok-voice-transcribe-2.0",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="sk-test",
    )
    assert headers["Authorization"] == "Bearer sk-test"
    assert "Content-Type" not in headers


def test_validate_environment_requires_key(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setattr(litellm, "xai_key", None)
    with pytest.raises(ValueError, match="xAI API key is required"):
        CONFIG.validate_environment(
            headers={},
            model="grok-voice-transcribe-2.0",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=None,
        )


def test_transform_response_maps_xai_shape():
    raw = httpx.Response(
        200,
        json={
            "text": "hello world",
            "language": "en",
            "duration": 3.2,
            "words": [
                {"text": "hello", "start": 0.0, "end": 0.5, "speaker": 1},
                {"text": "world", "start": 0.5, "end": 1.0},
            ],
        },
        request=httpx.Request("POST", "https://api.x.ai/v1/stt"),
    )
    response = CONFIG.transform_audio_transcription_response(raw_response=raw)

    assert response.text == "hello world"
    assert response["language"] == "en"
    assert response["duration"] == 3.2
    assert response["task"] == "transcribe"
    assert response["words"] == [
        {"word": "hello", "start": 0.0, "end": 0.5, "speaker": 1},
        {"word": "world", "start": 0.5, "end": 1.0},
    ]
    assert response._hidden_params["audio_transcription_duration"] == 3.2


def test_transcription_routes_to_xai_stt(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setattr(litellm, "xai_key", None)
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return httpx.Response(
            200,
            json={"text": "transcribed text", "language": "en", "duration": 1.5},
            request=request,
        )

    http_handler = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(handler)))
    response = litellm.transcription(
        model="xai/grok-voice-transcribe-2.0",
        file=("sample.wav", WAV_BYTES, "audio/wav"),
        api_key="sk-test",
        diarize=True,
        keyterm=["LiteLLM"],
        client=http_handler,
    )

    request = captured["request"]
    assert str(request.url) == "https://api.x.ai/v1/stt"
    assert request.headers["Authorization"] == "Bearer sk-test"
    body = request.content.decode("utf-8", errors="replace")
    assert 'name="model"' in body and "grok-voice-transcribe-2.0" in body
    assert 'name="diarize"' in body and "true" in body
    assert 'name="keyterm"' in body and "LiteLLM" in body
    assert 'name="file"' in body
    assert response.text == "transcribed text"


def test_provider_config_manager_returns_xai_config():
    config = ProviderConfigManager.get_provider_audio_transcription_config(
        model="grok-voice-transcribe-2.0",
        provider=LlmProviders.XAI,
    )
    assert isinstance(config, XAIAudioTranscriptionConfig)
