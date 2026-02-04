"""
Sarvam Audio Transcription Transformation Tests

Tests for Sarvam AI Speech-to-Text integration.
Following TDD - these tests are written BEFORE implementation.
"""

import io
import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
)
from litellm.llms.sarvam.audio_transcription.transformation import (
    SarvamAudioTranscriptionConfig,
)
from litellm.types.utils import TranscriptionResponse


@pytest.fixture
def handler():
    return SarvamAudioTranscriptionConfig()


@pytest.fixture
def test_bytes():
    return b"audio_data", b"audio_data"


@pytest.fixture
def test_io_bytes(test_bytes):
    return io.BytesIO(test_bytes[0]), test_bytes[1]


class TestGetCompleteUrl:
    """Tests for URL generation"""

    def test_get_complete_url_basic(self, handler):
        """Test basic URL generation without optional parameters"""
        url = handler.get_complete_url(
            api_base=None,
            api_key=None,
            model="saarika:v2.5",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://api.sarvam.ai/speech-to-text"

    def test_get_complete_url_with_custom_api_base(self, handler):
        """Test URL generation with custom API base"""
        url = handler.get_complete_url(
            api_base="https://custom.sarvam.ai",
            api_key=None,
            model="saarika:v2.5",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://custom.sarvam.ai/speech-to-text"


class TestValidateEnvironment:
    """Tests for environment validation and headers"""

    def test_validate_environment_with_api_key(self, handler):
        """Test that API key is set in headers correctly"""
        headers = handler.validate_environment(
            headers={},
            model="saarika:v2.5",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="test-api-key",
        )
        assert headers["api-subscription-key"] == "test-api-key"

    def test_validate_environment_missing_key_raises(self, handler):
        """Test that missing API key raises ValueError"""
        with pytest.raises(ValueError, match="SARVAM_API_KEY"):
            handler.validate_environment(
                headers={},
                model="saarika:v2.5",
                messages=[],
                optional_params={},
                litellm_params={},
                api_key=None,
            )


class TestTransformRequest:
    """Tests for request transformation"""

    def test_transform_audio_transcription_request_bytes(self, handler, test_bytes):
        """Test request transformation with raw bytes"""
        audio_file, expected_content = test_bytes
        result = handler.transform_audio_transcription_request(
            model="saarika:v2.5",
            audio_file=audio_file,
            optional_params={},
            litellm_params={},
        )

        assert isinstance(result, AudioTranscriptionRequestData)
        assert result.files is not None
        assert "file" in result.files

    def test_transform_audio_transcription_request_io(self, handler, test_io_bytes):
        """Test request transformation with BytesIO"""
        audio_file, expected_content = test_io_bytes
        result = handler.transform_audio_transcription_request(
            model="saarika:v2.5",
            audio_file=audio_file,
            optional_params={},
            litellm_params={},
        )

        assert isinstance(result, AudioTranscriptionRequestData)
        assert result.files is not None

    def test_transform_request_includes_model(self, handler, test_bytes):
        """Test that model is included in request data"""
        audio_file, _ = test_bytes
        result = handler.transform_audio_transcription_request(
            model="saarika:v2.5",
            audio_file=audio_file,
            optional_params={},
            litellm_params={},
        )

        # Model should be in form data
        assert isinstance(result.data, dict)
        assert result.data.get("model") == "saarika:v2.5"

    def test_transform_request_includes_language(self, handler, test_bytes):
        """Test that language code is included when provided"""
        audio_file, _ = test_bytes
        result = handler.transform_audio_transcription_request(
            model="saarika:v2.5",
            audio_file=audio_file,
            optional_params={"language": "hi-IN"},
            litellm_params={},
        )

        assert isinstance(result.data, dict)
        assert result.data.get("language_code") == "hi-IN"


class TestTransformResponse:
    """Tests for response transformation"""

    def test_transform_response_basic(self, handler):
        """Test basic response transformation"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "transcript": "Hello, this is a test.",
            "language_code": "en-IN",
        }

        result = handler.transform_audio_transcription_response(mock_response)

        assert isinstance(result, TranscriptionResponse)
        assert result.text == "Hello, this is a test."
        assert result["language"] == "en-IN"

    def test_transform_response_with_hindi(self, handler):
        """Test response with Hindi text"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "transcript": "नमस्ते, यह एक परीक्षण है।",
            "language_code": "hi-IN",
        }

        result = handler.transform_audio_transcription_response(mock_response)

        assert isinstance(result, TranscriptionResponse)
        assert result.text == "नमस्ते, यह एक परीक्षण है।"
        assert result["language"] == "hi-IN"


class TestSupportedParams:
    """Tests for supported parameters"""

    def test_get_supported_openai_params(self, handler):
        """Test that language is a supported OpenAI parameter"""
        params = handler.get_supported_openai_params("saarika:v2.5")
        assert "language" in params


class TestSaarasV3Modes:
    """Tests for saaras:v3 mode parameter handling"""

    def test_mode_transcribe_included_in_request(self, handler, test_bytes):
        """Test that transcribe mode is included in request"""
        audio_file, _ = test_bytes
        result = handler.transform_audio_transcription_request(
            model="saaras:v3",
            audio_file=audio_file,
            optional_params={"mode": "transcribe"},
            litellm_params={},
        )

        assert isinstance(result.data, dict)
        assert result.data.get("mode") == "transcribe"
        assert result.data.get("model") == "saaras:v3"

    def test_mode_translate_included_in_request(self, handler, test_bytes):
        """Test that translate mode is included in request"""
        audio_file, _ = test_bytes
        result = handler.transform_audio_transcription_request(
            model="saaras:v3",
            audio_file=audio_file,
            optional_params={"mode": "translate"},
            litellm_params={},
        )

        assert isinstance(result.data, dict)
        assert result.data.get("mode") == "translate"

    def test_mode_verbatim_included_in_request(self, handler, test_bytes):
        """Test that verbatim mode is included in request"""
        audio_file, _ = test_bytes
        result = handler.transform_audio_transcription_request(
            model="saaras:v3",
            audio_file=audio_file,
            optional_params={"mode": "verbatim"},
            litellm_params={},
        )

        assert isinstance(result.data, dict)
        assert result.data.get("mode") == "verbatim"

    def test_mode_translit_included_in_request(self, handler, test_bytes):
        """Test that translit mode is included in request"""
        audio_file, _ = test_bytes
        result = handler.transform_audio_transcription_request(
            model="saaras:v3",
            audio_file=audio_file,
            optional_params={"mode": "translit"},
            litellm_params={},
        )

        assert isinstance(result.data, dict)
        assert result.data.get("mode") == "translit"

    def test_mode_codemix_included_in_request(self, handler, test_bytes):
        """Test that codemix mode is included in request"""
        audio_file, _ = test_bytes
        result = handler.transform_audio_transcription_request(
            model="saaras:v3",
            audio_file=audio_file,
            optional_params={"mode": "codemix"},
            litellm_params={},
        )

        assert isinstance(result.data, dict)
        assert result.data.get("mode") == "codemix"

    def test_mode_not_included_when_not_provided(self, handler, test_bytes):
        """Test that mode is not included when not provided"""
        audio_file, _ = test_bytes
        result = handler.transform_audio_transcription_request(
            model="saaras:v3",
            audio_file=audio_file,
            optional_params={},
            litellm_params={},
        )

        assert isinstance(result.data, dict)
        assert "mode" not in result.data

    def test_mode_with_language_code(self, handler, test_bytes):
        """Test that mode works together with language_code"""
        audio_file, _ = test_bytes
        result = handler.transform_audio_transcription_request(
            model="saaras:v3",
            audio_file=audio_file,
            optional_params={"mode": "translate", "language": "hi-IN"},
            litellm_params={},
        )

        assert isinstance(result.data, dict)
        assert result.data.get("mode") == "translate"
        assert result.data.get("language_code") == "hi-IN"
