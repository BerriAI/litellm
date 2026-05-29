import io
import json
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

import litellm
from litellm.llms.azure.audio_transcription.transformation import (
    AzureSpeechAudioTranscriptionConfig,
    AzureSpeechAudioTranscriptionException,
)
from litellm.llms.base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
    BaseAudioTranscriptionConfig,
)
from litellm.types.utils import TranscriptionResponse
from litellm.utils import ProviderConfigManager


def test_azure_speech_audio_transcription_config_installed():
    config = ProviderConfigManager.get_provider_audio_transcription_config(
        model="speech/azure-stt",
        provider=litellm.LlmProviders.AZURE,
    )

    assert isinstance(config, BaseAudioTranscriptionConfig)
    assert isinstance(config, AzureSpeechAudioTranscriptionConfig)


def test_azure_speech_audio_transcription_builds_stt_url_from_cognitive_endpoint():
    config = AzureSpeechAudioTranscriptionConfig()

    url = config.get_complete_url(
        api_base="https://eastus.api.cognitive.microsoft.com/",
        api_key="test-key",
        model="speech/azure-stt",
        optional_params={"language": "fr-FR", "response_format": "verbose_json"},
        litellm_params={},
    )

    assert (
        url
        == "https://eastus.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?language=fr-FR&format=detailed"
    )


def test_azure_speech_audio_transcription_accepts_stt_endpoint_base():
    config = AzureSpeechAudioTranscriptionConfig()

    url = config.get_complete_url(
        api_base="https://westus.stt.speech.microsoft.com",
        api_key="test-key",
        model="speech/azure-stt",
        optional_params={},
        litellm_params={},
    )

    assert (
        url
        == "https://westus.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?language=en-US&format=simple"
    )


def test_azure_speech_audio_transcription_uses_dedicated_api_base_env(monkeypatch):
    config = AzureSpeechAudioTranscriptionConfig()

    monkeypatch.setattr(
        "litellm.llms.azure.audio_transcription.transformation.get_secret_str",
        lambda key: (
            "https://centralus.api.cognitive.microsoft.com"
            if key == "AZURE_SPEECH_API_BASE"
            else None
        ),
    )

    url = config.get_complete_url(
        api_base=None,
        api_key="test-key",
        model="speech/azure-stt",
        optional_params={},
        litellm_params={},
    )

    assert (
        url
        == "https://centralus.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?language=en-US&format=simple"
    )


def test_azure_speech_audio_transcription_rejects_azure_openai_endpoint():
    config = AzureSpeechAudioTranscriptionConfig()

    with pytest.raises(
        AzureSpeechAudioTranscriptionException,
        match="not an Azure OpenAI endpoint",
    ):
        config.get_complete_url(
            api_base="https://example.openai.azure.com",
            api_key="test-key",
            model="speech/azure-stt",
            optional_params={},
            litellm_params={},
        )


def test_azure_speech_audio_transcription_validate_environment():
    config = AzureSpeechAudioTranscriptionConfig()

    headers = config.validate_environment(
        headers={},
        model="speech/azure-stt",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="test-key",
    )

    assert headers["Ocp-Apim-Subscription-Key"] == "test-key"
    assert headers["Content-Type"] == "audio/wav"
    assert headers["Accept"] == "application/json"


def test_azure_speech_audio_transcription_uses_dedicated_api_key_env(monkeypatch):
    config = AzureSpeechAudioTranscriptionConfig()

    monkeypatch.setattr(
        "litellm.llms.azure.audio_transcription.transformation.get_secret_str",
        lambda key: "speech-key" if key == "AZURE_SPEECH_API_KEY" else None,
    )

    headers = config.validate_environment(
        headers={},
        model="speech/azure-stt",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key=None,
    )

    assert headers["Ocp-Apim-Subscription-Key"] == "speech-key"


def test_azure_speech_audio_transcription_request_transform():
    config = AzureSpeechAudioTranscriptionConfig()
    audio = io.BytesIO(b"RIFF....WAVE")

    request_data = config.transform_audio_transcription_request(
        model="speech/azure-stt",
        audio_file=audio,
        optional_params={},
        litellm_params={},
    )

    assert isinstance(request_data, AudioTranscriptionRequestData)
    assert request_data.data == b"RIFF....WAVE"
    assert request_data.files is None
    assert request_data.content_type == "audio/wav"


@pytest.mark.parametrize(
    "payload,expected_text",
    [
        ({"DisplayText": "hello world"}, "hello world"),
        (
            {
                "RecognitionStatus": "Success",
                "NBest": [{"Display": "best text", "Confidence": 0.91}],
            },
            "best text",
        ),
    ],
)
def test_azure_speech_audio_transcription_response_transform(payload, expected_text):
    config = AzureSpeechAudioTranscriptionConfig()
    response = httpx.Response(200, json=payload)

    result = config.transform_audio_transcription_response(response)

    assert isinstance(result, TranscriptionResponse)
    assert result.text == expected_text
    assert result._hidden_params == payload


def test_azure_speech_audio_transcription_response_raises_on_failed_status():
    config = AzureSpeechAudioTranscriptionConfig()
    response = httpx.Response(
        200,
        json={
            "RecognitionStatus": "NoMatch",
            "Offset": 0,
            "Duration": 0,
        },
    )

    with pytest.raises(
        AzureSpeechAudioTranscriptionException,
        match="RecognitionStatus=NoMatch",
    ):
        config.transform_audio_transcription_response(response)


def test_azure_speech_transcription_routes_through_provider_config(monkeypatch):
    expected = TranscriptionResponse(text="hello")
    audio_handler = MagicMock(return_value=expected)

    monkeypatch.setattr(
        litellm.main.base_llm_http_handler,
        "audio_transcriptions",
        audio_handler,
    )

    response = litellm.transcription(
        model="azure/speech/azure-stt",
        file=io.BytesIO(b"RIFF....WAVE"),
        api_base="https://eastus.api.cognitive.microsoft.com",
        api_key="test-key",
        language="en-US",
    )

    assert response is expected
    audio_handler.assert_called_once()
    assert isinstance(
        audio_handler.call_args.kwargs["provider_config"],
        AzureSpeechAudioTranscriptionConfig,
    )
    assert audio_handler.call_args.kwargs["custom_llm_provider"] == "azure"


def test_azure_speech_stt_has_non_zero_input_pricing():
    pricing_path = Path(__file__).parents[4] / "model_prices_and_context_window.json"
    pricing = json.loads(pricing_path.read_text())

    assert pricing["azure/speech/azure-stt"]["input_cost_per_second"] > 0
    assert (
        pricing["azure/speech/azure-stt"]["audio_transcription_config"]
        == "azure_speech"
    )
