import os
from typing import Dict
from unittest.mock import MagicMock

import httpx
import litellm
import pytest

from litellm.llms.base_llm.audio_transcription.transformation import (
    BaseAudioTranscriptionConfig,
)
from litellm.llms.mistral.audio_transcription.transformation import (
    MistralAudioTranscriptionConfig,
)
from litellm.types.utils import TranscriptionResponse
from litellm.utils import ProviderConfigManager
from tests.llm_translation.base_audio_transcription_unit_tests import (
    BaseLLMAudioTranscriptionTest,
)


@pytest.mark.skipif(
    not os.getenv("MISTRAL_API_KEY"),
    reason="MISTRAL_API_KEY not set, skipping Mistral audio transcription tests",
)
class TestMistralAudioTranscription(BaseLLMAudioTranscriptionTest):
    def get_base_audio_transcription_call_args(self) -> Dict:
        return {
            "model": "mistral/voxtral-mini-latest",
        }

    def get_custom_llm_provider(self) -> litellm.LlmProviders:
        return litellm.LlmProviders.MISTRAL

    def test_audio_transcription_async(self):  # type: ignore[override]
        pytest.skip(
            "Async audio transcription test for Mistral is skipped in this suite; "
            "async test plugins (e.g. pytest-asyncio/anyio) are not configured here."
        )


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
