"""
Unit tests for GDM Transcription configuration.

GDM (https://ai.gdm.se) provides an OpenAI-compatible API.
"""

import io
import os
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.llms.gdm.transcription.transformation import (
    GDMException,
    GDMTranscriptionConfig,
)
from litellm.types.utils import TranscriptionResponse


class TestGDMTranscriptionConfig:
    """Test class for GDM Transcription functionality"""

    @pytest.fixture
    def config(self):
        """Get GDM transcription config"""
        return GDMTranscriptionConfig()

    @pytest.fixture
    def test_audio_bytes(self):
        """Mock audio file bytes"""
        return b"fake_audio_data_for_testing"

    @pytest.fixture
    def test_audio_file(self):
        """Mock audio file object"""
        return io.BytesIO(b"fake_audio_data_for_testing")

    def test_get_supported_openai_params(self, config):
        """
        Test that get_supported_openai_params returns expected params
        """
        supported_params = config.get_supported_openai_params(model="whisper-1")

        assert "language" in supported_params
        assert "prompt" in supported_params
        assert "response_format" in supported_params
        assert "timestamp_granularities" in supported_params
        assert "temperature" in supported_params

    def test_map_openai_params_language(self, config):
        """
        Test that language parameter is correctly mapped
        """
        non_default_params = {"language": "en"}

        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model="whisper-1",
            drop_params=False
        )

        assert result.get("language") == "en"

    def test_map_openai_params_temperature(self, config):
        """
        Test that temperature parameter is correctly mapped
        """
        non_default_params = {"temperature": 0.5}

        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model="whisper-1",
            drop_params=False
        )

        assert result.get("temperature") == 0.5

    def test_map_openai_params_multiple(self, config):
        """
        Test that multiple parameters are correctly mapped
        """
        non_default_params = {
            "language": "sv",
            "prompt": "This is a prompt",
            "response_format": "json",
            "temperature": 0.3
        }

        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model="whisper-1",
            drop_params=False
        )

        assert result.get("language") == "sv"
        assert result.get("prompt") == "This is a prompt"
        assert result.get("response_format") == "json"
        assert result.get("temperature") == 0.3

    def test_map_openai_params_unsupported_dropped(self, config):
        """
        Test that unsupported parameters are not mapped
        """
        non_default_params = {
            "language": "en",
            "unsupported_param": "should_be_dropped"
        }

        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model="whisper-1",
            drop_params=False
        )

        assert result.get("language") == "en"
        assert "unsupported_param" not in result

    def test_get_complete_url_default(self, config):
        """
        Test that get_complete_url constructs the correct endpoint URL
        """
        url = config.get_complete_url(
            api_base=None,
            api_key="test-key",
            model="whisper-1",
            optional_params={},
            litellm_params={},
            stream=False
        )

        assert url == "https://ai.gdm.se/api/v1/audio/transcriptions"

    def test_get_complete_url_with_custom_base(self, config):
        """
        Test that get_complete_url works with custom api_base
        """
        url = config.get_complete_url(
            api_base="https://custom.gdm.se/api/v1",
            api_key="test-key",
            model="whisper-1",
            optional_params={},
            litellm_params={},
            stream=False
        )

        assert url == "https://custom.gdm.se/api/v1/audio/transcriptions"

    def test_get_complete_url_strips_trailing_slash(self, config):
        """
        Test that get_complete_url properly handles trailing slashes
        """
        url = config.get_complete_url(
            api_base="https://custom.gdm.se/api/v1/",
            api_key="test-key",
            model="whisper-1",
            optional_params={},
            litellm_params={},
            stream=False
        )

        assert url == "https://custom.gdm.se/api/v1/audio/transcriptions"

    def test_get_error_class(self, config):
        """
        Test that get_error_class returns GDMException
        """
        error = config.get_error_class(
            error_message="Test error",
            status_code=400,
            headers={"Content-Type": "application/json"}
        )

        assert isinstance(error, GDMException)
        assert error.status_code == 400
        assert error.message == "Test error"

    def test_validate_environment_with_api_key(self, config):
        """
        Test that validate_environment returns proper headers with API key
        """
        headers = config.validate_environment(
            headers={},
            model="whisper-1",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="test-api-key",
            api_base=None,
        )

        assert headers["Authorization"] == "Bearer test-api-key"
        assert headers["accept"] == "application/json"

    def test_validate_environment_with_env_key(self, config, monkeypatch):
        """
        Test that validate_environment reads API key from environment
        """
        monkeypatch.setenv("GDM_API_KEY", "env-api-key")

        headers = config.validate_environment(
            headers={},
            model="whisper-1",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=None,
            api_base=None,
        )

        assert headers["Authorization"] == "Bearer env-api-key"

    def test_validate_environment_custom_headers_override(self, config):
        """
        Test that custom headers are merged with defaults
        """
        custom_headers = {"X-Custom-Header": "custom-value"}

        headers = config.validate_environment(
            headers=custom_headers,
            model="whisper-1",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="test-api-key",
            api_base=None,
        )

        assert headers["Authorization"] == "Bearer test-api-key"
        assert headers["X-Custom-Header"] == "custom-value"

    def test_transform_audio_transcription_request(self, config, test_audio_bytes):
        """
        Test that transform_audio_transcription_request creates proper request data
        """
        with patch("litellm.litellm_core_utils.audio_utils.utils.process_audio_file") as mock_process:
            mock_processed = MagicMock()
            mock_processed.filename = "test.wav"
            mock_processed.file_content = test_audio_bytes
            mock_processed.content_type = "audio/wav"
            mock_process.return_value = mock_processed

            request_data = config.transform_audio_transcription_request(
                model="whisper-1",
                audio_file=test_audio_bytes,
                optional_params={"language": "en", "temperature": 0.5},
                litellm_params={}
            )

            assert request_data.data["model"] == "whisper-1"
            assert request_data.data["language"] == "en"
            assert request_data.data["temperature"] == 0.5
            assert "file" in request_data.files

    def test_transform_audio_transcription_request_minimal(self, config, test_audio_bytes):
        """
        Test that transform_audio_transcription_request works with minimal params
        """
        with patch("litellm.litellm_core_utils.audio_utils.utils.process_audio_file") as mock_process:
            mock_processed = MagicMock()
            mock_processed.filename = "test.wav"
            mock_processed.file_content = test_audio_bytes
            mock_processed.content_type = "audio/wav"
            mock_process.return_value = mock_processed

            request_data = config.transform_audio_transcription_request(
                model="whisper-1",
                audio_file=test_audio_bytes,
                optional_params={},
                litellm_params={}
            )

            assert request_data.data["model"] == "whisper-1"
            assert "file" in request_data.files

    def test_transform_audio_transcription_response_success(self, config):
        """
        Test that transform_audio_transcription_response handles successful response
        """
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {"text": "Hello, world!"}
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}

        result = config.transform_audio_transcription_response(mock_response)

        assert isinstance(result, TranscriptionResponse)
        assert result.text == "Hello, world!"

    def test_transform_audio_transcription_response_with_transcript_field(self, config):
        """
        Test that transform_audio_transcription_response handles 'transcript' field
        """
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {"transcript": "Hello, world!"}
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}

        result = config.transform_audio_transcription_response(mock_response)

        assert isinstance(result, TranscriptionResponse)
        assert result.text == "Hello, world!"

    def test_transform_audio_transcription_response_empty_text(self, config):
        """
        Test that transform_audio_transcription_response handles empty response
        """
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {}
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}

        result = config.transform_audio_transcription_response(mock_response)

        assert isinstance(result, TranscriptionResponse)
        assert result.text == ""

    def test_transform_audio_transcription_response_invalid_json(self, config):
        """
        Test that transform_audio_transcription_response handles invalid JSON
        """
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.side_effect = Exception("Invalid JSON")
        mock_response.text = "Invalid response body"
        mock_response.status_code = 500
        mock_response.headers = {"Content-Type": "text/plain"}

        with pytest.raises(GDMException) as exc_info:
            config.transform_audio_transcription_response(mock_response)

        assert exc_info.value.status_code == 500
        assert exc_info.value.message == "Invalid response body"


class TestGDMException:
    """Test class for GDMException"""

    def test_gdm_exception_creation(self):
        """
        Test that GDMException can be created with required parameters
        """
        exception = GDMException(
            status_code=400,
            message="Bad Request"
        )

        assert exception.status_code == 400
        assert exception.message == "Bad Request"

    def test_gdm_exception_with_headers(self):
        """
        Test that GDMException can be created with headers
        """
        headers = {"Content-Type": "application/json", "X-Request-Id": "12345"}
        exception = GDMException(
            status_code=500,
            message="Internal Server Error",
            headers=headers
        )

        assert exception.status_code == 500
        assert exception.message == "Internal Server Error"
        assert exception.headers == headers

    def test_gdm_exception_with_httpx_headers(self):
        """
        Test that GDMException can be created with httpx.Headers
        """
        headers = httpx.Headers({"Content-Type": "application/json"})
        exception = GDMException(
            status_code=404,
            message="Not Found",
            headers=headers
        )

        assert exception.status_code == 404
        assert exception.message == "Not Found"
