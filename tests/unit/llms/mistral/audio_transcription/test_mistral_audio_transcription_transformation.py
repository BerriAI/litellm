import os
from unittest.mock import MagicMock

import httpx
import pytest

import litellm

from litellm.llms.base_llm.audio_transcription.transformation import (
    BaseAudioTranscriptionConfig,
)
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.mistral.audio_transcription.transformation import (
    MistralAudioTranscriptionConfig,
)
from litellm.types.utils import TranscriptionResponse
from litellm.utils import ProviderConfigManager


def test_mistral_audio_transcription_config_installed():
    """Ensure Mistral audio transcription config is registered with ProviderConfigManager."""
    config = ProviderConfigManager.get_provider_audio_transcription_config(
        model="mistral/voxtral-mini-latest",
        provider=litellm.LlmProviders.MISTRAL,
    )
    assert config is not None
    assert isinstance(config, BaseAudioTranscriptionConfig)
    assert isinstance(config, MistralAudioTranscriptionConfig)


def test_mistral_audio_transcription_get_complete_url():
    config = MistralAudioTranscriptionConfig()
    url = config.get_complete_url(
        api_base=None,
        api_key="fake-key",
        model="voxtral-mini-latest",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://api.mistral.ai/v1/audio/transcriptions"


def test_mistral_audio_transcription_get_complete_url_custom_base():
    config = MistralAudioTranscriptionConfig()
    url = config.get_complete_url(
        api_base="https://custom.api.example.com/v1/",
        api_key="fake-key",
        model="voxtral-mini-latest",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://custom.api.example.com/v1/audio/transcriptions"


def test_mistral_audio_transcription_validate_environment():
    config = MistralAudioTranscriptionConfig()
    headers = config.validate_environment(
        headers={},
        model="voxtral-mini-latest",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="test-key-123",
    )
    assert headers["Authorization"] == "Bearer test-key-123"
    assert headers["accept"] == "application/json"


def test_mistral_audio_transcription_supported_params():
    config = MistralAudioTranscriptionConfig()
    params = config.get_supported_openai_params("voxtral-mini-latest")
    assert "language" in params
    assert "temperature" in params
    assert "response_format" in params
    assert "timestamp_granularities" in params


def test_mistral_audio_transcription_request_transform():
    config = MistralAudioTranscriptionConfig()

    wav_path = os.path.join(
        os.path.dirname(__file__),
        "../../../../..",
        "tests",
        "llm_translation",
        "gettysburg.wav",
    )
    audio_file = open(wav_path, "rb")

    result = config.transform_audio_transcription_request(
        model="voxtral-mini-latest",
        audio_file=audio_file,
        optional_params={"language": "en", "temperature": 0.0},
        litellm_params={},
    )

    audio_file.close()

    assert isinstance(result.data, dict)
    assert result.data["model"] == "voxtral-mini-latest"
    assert result.data["language"] == "en"
    assert result.data["temperature"] == 0.0
    assert result.files is not None
    assert "file" in result.files


def test_mistral_audio_transcription_request_with_diarize():
    """Test that Mistral-specific params like diarize are passed through."""
    config = MistralAudioTranscriptionConfig()

    wav_path = os.path.join(
        os.path.dirname(__file__),
        "../../../../..",
        "tests",
        "llm_translation",
        "gettysburg.wav",
    )
    audio_file = open(wav_path, "rb")

    result = config.transform_audio_transcription_request(
        model="voxtral-mini-latest",
        audio_file=audio_file,
        optional_params={"diarize": True},
        litellm_params={},
    )

    audio_file.close()

    assert isinstance(result.data, dict)
    assert result.data["diarize"] == "true"


def test_mistral_audio_transcription_response_transform():
    config = MistralAudioTranscriptionConfig()

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.json.return_value = {"text": "Four score and seven years ago..."}

    response = config.transform_audio_transcription_response(mock_response)

    assert isinstance(response, TranscriptionResponse)
    assert response.text == "Four score and seven years ago..."


def test_mistral_audio_transcription_response_transform_diarized():
    """Test that diarized responses preserve segments and language."""
    config = MistralAudioTranscriptionConfig()

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.json.return_value = {
        "model": "voxtral-mini-latest",
        "text": "Hello, how are you? I am fine.",
        "language": None,
        "segments": [
            {
                "text": "Hello, how are you?",
                "start": 0.3,
                "end": 2.1,
                "speaker_id": "speaker_1",
                "type": "transcription_segment",
            },
            {
                "text": "I am fine.",
                "start": 2.5,
                "end": 3.8,
                "speaker_id": "speaker_2",
                "type": "transcription_segment",
            },
        ],
        "usage": {
            "prompt_audio_seconds": 4,
            "prompt_tokens": 5,
            "total_tokens": 50,
            "completion_tokens": 20,
        },
    }

    response = config.transform_audio_transcription_response(mock_response)

    assert isinstance(response, TranscriptionResponse)
    assert response.text == "Hello, how are you? I am fine."
    assert response["segments"] is not None
    assert len(response["segments"]) == 2
    assert response["segments"][0]["speaker_id"] == "speaker_1"
    assert response["segments"][1]["speaker_id"] == "speaker_2"
    assert response["language"] is None


def test_mistral_audio_transcription_response_transform_empty():
    config = MistralAudioTranscriptionConfig()

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.json.return_value = {}

    response = config.transform_audio_transcription_response(mock_response)

    assert isinstance(response, TranscriptionResponse)
    assert response.text == ""


def test_mistral_audio_transcription_response_uses_prompt_audio_seconds_for_cost():
    """Mistral bills on usage.prompt_audio_seconds, so that is the duration cost reads"""
    config = MistralAudioTranscriptionConfig()

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.json.return_value = {
        "text": "hello",
        "usage": {"prompt_audio_seconds": 600, "total_tokens": 10},
    }

    response = config.transform_audio_transcription_response(mock_response)

    assert response.hidden_params["audio_transcription_duration"] == 600.0
    assert response.hidden_params["usage"] == {"prompt_audio_seconds": 600, "total_tokens": 10}
    assert getattr(response, "duration", None) is None


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_audio_seconds": None},
        {"prompt_audio_seconds": "not-a-number"},
        "not-an-object",
    ],
)
def test_mistral_audio_transcription_response_without_usable_usage_has_no_duration(usage):
    config = MistralAudioTranscriptionConfig()

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.json.return_value = {"text": "hello", "usage": usage}

    response = config.transform_audio_transcription_response(mock_response)

    assert "audio_transcription_duration" not in response.hidden_params


def test_mistral_transcription_cost_uses_provider_billed_seconds():
    """
    End to end through litellm.transcription: the seconds Mistral reports win over the
    local file measurement (about 17s for this wav), so spend matches what Mistral bills
    """
    billed_seconds = 30
    wav_path = os.path.join(
        os.path.dirname(__file__),
        "../../../../..",
        "tests",
        "llm_translation",
        "gettysburg.wav",
    )
    with open(wav_path, "rb") as f:
        audio_bytes = f.read()

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "voxtral-mini-2602",
                "text": "a tone",
                "language": None,
                "segments": [],
                "usage": {"prompt_audio_seconds": billed_seconds, "total_tokens": 3},
            },
            request=request,
        )

    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond)))

    response = litellm.transcription(
        model="mistral/voxtral-mini-2602",
        file=("gettysburg.wav", audio_bytes, "audio/wav"),
        api_key="fake-key",
        client=client,
    )

    assert response.hidden_params["audio_transcription_duration"] == float(billed_seconds)

    cost_per_second = litellm.get_model_info("mistral/voxtral-mini-2602")["input_cost_per_second"]
    assert cost_per_second
    cost = litellm.completion_cost(
        completion_response=response,
        model="mistral/voxtral-mini-2602",
        call_type="transcription",
    )
    assert cost == pytest.approx(billed_seconds * cost_per_second)
