"""
Tests for Mistral Audio Transcription (Voxtral Mini) transformation config.

Validates:
- URL construction
- Auth header generation
- Request transformation (multipart form data with OpenAI + Mistral-specific params)
- Response transformation (text, duration, segments, words, diarization)
- Supported OpenAI params
- Provider config registration
- Cost calculation with input_cost_per_second pricing
"""

import io
import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.llms.base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
    BaseAudioTranscriptionConfig,
)
from litellm.llms.mistral.audio_transcription.transformation import (
    MistralAudioTranscriptionConfig,
    MistralAudioTranscriptionError,
)
from litellm.types.utils import TranscriptionResponse
from litellm.utils import ProviderConfigManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    return MistralAudioTranscriptionConfig()


@pytest.fixture
def fake_audio_bytes():
    return b"fake-audio-data"


# ---------------------------------------------------------------------------
# URL construction
# ---------------------------------------------------------------------------


class TestGetCompleteUrl:
    def test_default_url(self, config):
        """Default URL points to Mistral's audio transcriptions endpoint."""
        url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model="voxtral-mini-latest",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://api.mistral.ai/v1/audio/transcriptions"

    def test_custom_api_base(self, config):
        """Custom api_base replaces the default host."""
        url = config.get_complete_url(
            api_base="https://custom.mistral.example.com/v1",
            api_key=None,
            model="voxtral-mini-2602",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://custom.mistral.example.com/v1/audio/transcriptions"

    def test_api_base_trailing_slash_stripped(self, config):
        """Trailing slashes on api_base are stripped before appending path."""
        url = config.get_complete_url(
            api_base="https://api.mistral.ai/v1/",
            api_key=None,
            model="voxtral-mini-latest",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://api.mistral.ai/v1/audio/transcriptions"

    def test_api_base_from_env(self, config, monkeypatch):
        """Falls back to MISTRAL_API_BASE env var when api_base is None."""
        monkeypatch.setenv("MISTRAL_API_BASE", "https://env-base.example.com/v1")
        url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model="voxtral-mini-latest",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://env-base.example.com/v1/audio/transcriptions"


# ---------------------------------------------------------------------------
# Auth / validate_environment
# ---------------------------------------------------------------------------


class TestValidateEnvironment:
    def test_returns_bearer_header(self, config):
        headers = config.validate_environment(
            headers={},
            model="voxtral-mini-latest",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="test-key-123",
        )
        assert headers["Authorization"] == "Bearer test-key-123"

    def test_raises_without_api_key(self, config, monkeypatch):
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Missing Mistral API Key"):
            config.validate_environment(
                headers={},
                model="voxtral-mini-latest",
                messages=[],
                optional_params={},
                litellm_params={},
                api_key=None,
            )

    def test_falls_back_to_env_var(self, config, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "env-key-456")
        headers = config.validate_environment(
            headers={},
            model="voxtral-mini-latest",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=None,
        )
        assert headers["Authorization"] == "Bearer env-key-456"


# ---------------------------------------------------------------------------
# Request transformation
# ---------------------------------------------------------------------------


class TestTransformRequest:
    def test_basic_request(self, config, fake_audio_bytes):
        """Minimal request contains model and file."""
        result = config.transform_audio_transcription_request(
            model="voxtral-mini-latest",
            audio_file=fake_audio_bytes,
            optional_params={},
            litellm_params={},
        )
        assert isinstance(result, AudioTranscriptionRequestData)
        assert result.data["model"] == "voxtral-mini-latest"
        assert result.files is not None
        assert "file" in result.files
        # file tuple: (filename, content)
        assert result.files["file"][1] == fake_audio_bytes

    def test_language_param(self, config, fake_audio_bytes):
        result = config.transform_audio_transcription_request(
            model="voxtral-mini-latest",
            audio_file=fake_audio_bytes,
            optional_params={"language": "fr"},
            litellm_params={},
        )
        assert result.data["language"] == "fr"

    def test_response_format_param(self, config, fake_audio_bytes):
        result = config.transform_audio_transcription_request(
            model="voxtral-mini-latest",
            audio_file=fake_audio_bytes,
            optional_params={"response_format": "verbose_json"},
            litellm_params={},
        )
        assert result.data["response_format"] == "verbose_json"

    def test_timestamp_granularities_list(self, config, fake_audio_bytes):
        """timestamp_granularities as list produces repeated form keys."""
        result = config.transform_audio_transcription_request(
            model="voxtral-mini-latest",
            audio_file=fake_audio_bytes,
            optional_params={"timestamp_granularities": ["segment", "word"]},
            litellm_params={},
        )
        assert result.data["timestamp_granularities[]"] == ["segment", "word"]

    def test_timestamp_granularities_string(self, config, fake_audio_bytes):
        """timestamp_granularities as single string."""
        result = config.transform_audio_transcription_request(
            model="voxtral-mini-latest",
            audio_file=fake_audio_bytes,
            optional_params={"timestamp_granularities": "word"},
            litellm_params={},
        )
        assert result.data["timestamp_granularities[]"] == "word"

    def test_diarize_param(self, config, fake_audio_bytes):
        """Mistral-specific diarize param is passed through."""
        result = config.transform_audio_transcription_request(
            model="voxtral-mini-latest",
            audio_file=fake_audio_bytes,
            optional_params={"diarize": True},
            litellm_params={},
        )
        assert result.data["diarize"] == "true"

    def test_diarize_false(self, config, fake_audio_bytes):
        result = config.transform_audio_transcription_request(
            model="voxtral-mini-latest",
            audio_file=fake_audio_bytes,
            optional_params={"diarize": False},
            litellm_params={},
        )
        assert result.data["diarize"] == "false"

    def test_context_bias_param(self, config, fake_audio_bytes):
        """Mistral-specific context_bias param is passed through."""
        result = config.transform_audio_transcription_request(
            model="voxtral-mini-latest",
            audio_file=fake_audio_bytes,
            optional_params={"context_bias": "LiteLLM,Voxtral,API"},
            litellm_params={},
        )
        assert result.data["context_bias"] == "LiteLLM,Voxtral,API"

    def test_all_params_combined(self, config, fake_audio_bytes):
        """All OpenAI + Mistral params together."""
        result = config.transform_audio_transcription_request(
            model="voxtral-mini-2602",
            audio_file=fake_audio_bytes,
            optional_params={
                "language": "en",
                "response_format": "verbose_json",
                "timestamp_granularities": ["word"],
                "diarize": True,
                "context_bias": "AI,ML",
            },
            litellm_params={},
        )
        data = result.data
        assert data["model"] == "voxtral-mini-2602"
        assert data["language"] == "en"
        assert data["response_format"] == "verbose_json"
        assert data["timestamp_granularities[]"] == ["word"]
        assert data["diarize"] == "true"
        assert data["context_bias"] == "AI,ML"

    def test_file_tuple_input(self, config):
        """File provided as (filename, content) tuple."""
        audio_tuple = ("meeting.mp3", b"mp3-data")
        result = config.transform_audio_transcription_request(
            model="voxtral-mini-latest",
            audio_file=audio_tuple,
            optional_params={},
            litellm_params={},
        )
        assert result.files["file"][0] == "meeting.mp3"
        assert result.files["file"][1] == b"mp3-data"

    def test_io_bytes_input(self, config):
        """File provided as BytesIO object."""
        bio = io.BytesIO(b"wav-data")
        result = config.transform_audio_transcription_request(
            model="voxtral-mini-latest",
            audio_file=bio,
            optional_params={},
            litellm_params={},
        )
        assert result.files["file"][1] == b"wav-data"


# ---------------------------------------------------------------------------
# Response transformation
# ---------------------------------------------------------------------------


class TestTransformResponse:
    def test_basic_response(self, config):
        """Basic text-only response."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": "Hello world"}

        result = config.transform_audio_transcription_response(mock_resp)

        assert isinstance(result, TranscriptionResponse)
        assert result.text == "Hello world"
        assert result["task"] == "transcribe"

    def test_response_with_duration_and_language(self, config):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "text": "Bonjour le monde",
            "language": "fr",
            "duration": 3.14,
        }

        result = config.transform_audio_transcription_response(mock_resp)

        assert result.text == "Bonjour le monde"
        assert result["language"] == "fr"
        assert result["duration"] == 3.14

    def test_response_with_segments(self, config):
        segments = [
            {"id": 0, "start": 0.0, "end": 2.5, "text": "Hello"},
            {"id": 1, "start": 2.5, "end": 5.0, "text": "world"},
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "text": "Hello world",
            "segments": segments,
        }

        result = config.transform_audio_transcription_response(mock_resp)

        assert result["segments"] == segments

    def test_response_with_words(self, config):
        words = [
            {"word": "Hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.6, "end": 1.0},
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "text": "Hello world",
            "words": words,
        }

        result = config.transform_audio_transcription_response(mock_resp)

        assert result["words"] == words

    def test_response_with_diarization(self, config):
        """Diarized response includes speaker labels in segments."""
        segments = [
            {"id": 0, "start": 0.0, "end": 2.0, "text": "Hi there", "speaker": 0},
            {"id": 1, "start": 2.5, "end": 4.0, "text": "Hello", "speaker": 1},
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "text": "Hi there Hello",
            "duration": 4.0,
            "segments": segments,
        }

        result = config.transform_audio_transcription_response(mock_resp)

        assert result["segments"][0]["speaker"] == 0
        assert result["segments"][1]["speaker"] == 1

    def test_response_preserves_hidden_params(self, config):
        """Full API response is stored in _hidden_params."""
        full_response = {
            "text": "test",
            "duration": 1.0,
            "language": "en",
            "some_future_field": "extra_data",
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = full_response

        result = config.transform_audio_transcription_response(mock_resp)

        assert result._hidden_params == full_response

    def test_response_bad_json_raises(self, config):
        """Non-JSON response raises ValueError."""
        mock_resp = MagicMock()
        mock_resp.json.side_effect = ValueError("Bad JSON")
        mock_resp.text = "not json at all"

        with pytest.raises(ValueError, match="Error parsing Mistral transcription"):
            config.transform_audio_transcription_response(mock_resp)

    def test_verbose_json_response(self, config):
        """Full verbose_json response with all fields."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "text": "The quick brown fox",
            "language": "en",
            "duration": 12.5,
            "segments": [
                {"id": 0, "start": 0.0, "end": 3.0, "text": "The quick brown fox"}
            ],
            "words": [
                {"word": "The", "start": 0.0, "end": 0.3},
                {"word": "quick", "start": 0.4, "end": 0.7},
                {"word": "brown", "start": 0.8, "end": 1.1},
                {"word": "fox", "start": 1.2, "end": 1.5},
            ],
        }

        result = config.transform_audio_transcription_response(mock_resp)

        assert result.text == "The quick brown fox"
        assert result["duration"] == 12.5
        assert result["language"] == "en"
        assert len(result["segments"]) == 1
        assert len(result["words"]) == 4


# ---------------------------------------------------------------------------
# Error class
# ---------------------------------------------------------------------------


class TestErrorClass:
    def test_get_error_class(self, config):
        error = config.get_error_class(
            error_message="Rate limit exceeded",
            status_code=429,
            headers={"retry-after": "30"},
        )
        assert isinstance(error, MistralAudioTranscriptionError)
        assert error.status_code == 429
        assert error.message == "Rate limit exceeded"


# ---------------------------------------------------------------------------
# Supported params & config registration
# ---------------------------------------------------------------------------


class TestConfigRegistration:
    def test_supported_openai_params(self, config):
        params = config.get_supported_openai_params(model="voxtral-mini-latest")
        assert "language" in params
        assert "response_format" in params
        assert "timestamp_granularities" in params
        # Should NOT include chat-only params
        assert "max_completion_tokens" not in params

    def test_provider_config_registered(self):
        """MistralAudioTranscriptionConfig is registered with ProviderConfigManager."""
        config = ProviderConfigManager.get_provider_audio_transcription_config(
            model="mistral/voxtral-mini-latest",
            provider=litellm.LlmProviders.MISTRAL,
        )
        assert config is not None
        assert isinstance(config, BaseAudioTranscriptionConfig)
        assert isinstance(config, MistralAudioTranscriptionConfig)


# ---------------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------------


class TestCostCalculation:
    def test_transcription_cost_uses_input_cost_per_second(self):
        """Mistral voxtral uses input_cost_per_second pricing ($0.003/min = $0.00005/s)."""
        from litellm import completion_cost

        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        response = TranscriptionResponse(text="transcribed audio content")
        response["duration"] = 120.0  # 2 minutes

        cost = completion_cost(
            completion_response=response,
            model="mistral/voxtral-mini-2602",
            custom_llm_provider="mistral",
            call_type="atranscription",
        )

        # 120 seconds * $0.00005/second = $0.006
        expected_cost = 120.0 * 5e-05
        assert pytest.approx(cost, rel=1e-6) == expected_cost

    def test_transcription_cost_voxtral_mini_latest(self):
        """Same pricing test for the 'latest' alias."""
        from litellm import completion_cost

        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        response = TranscriptionResponse(text="some words")
        response["duration"] = 60.0  # 1 minute

        cost = completion_cost(
            completion_response=response,
            model="mistral/voxtral-mini-latest",
            custom_llm_provider="mistral",
            call_type="transcription",
        )

        # 60 seconds * $0.00005/second = $0.003
        expected_cost = 60.0 * 5e-05
        assert pytest.approx(cost, rel=1e-6) == expected_cost

    def test_model_info_contains_correct_pricing(self):
        """get_model_info returns the expected pricing fields for voxtral."""
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        info = litellm.get_model_info(
            model="voxtral-mini-2602", custom_llm_provider="mistral"
        )
        assert info["input_cost_per_second"] == 5e-05
        assert info["mode"] == "audio_transcription"
        assert info["litellm_provider"] == "mistral"
        # output_cost_per_second should be None (not 0.0) so it doesn't
        # short-circuit the cost calculator
        assert info.get("output_cost_per_second") is None
