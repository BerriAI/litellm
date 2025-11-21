"""Unit tests for BurnCloud audio transcription transformation."""

import unittest
from typing import Dict, List, Union
from unittest.mock import patch, Mock

import httpx
from httpx import Headers

from litellm.llms.burncloud.audio_transcription.transformation import (
    BurnCloudAudioTranscriptionConfig,
)
from litellm.llms.base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
)
from litellm.llms.burncloud.common_utils import BurnCloudError
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import FileTypes, TranscriptionResponse


class TestBurnCloudAudioTranscriptionConfig(unittest.TestCase):
    """Test suite for BurnCloudAudioTranscriptionConfig class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = BurnCloudAudioTranscriptionConfig()
        self.test_model = "whisper-1"
        self.test_api_key = "test-api-key"
        self.test_api_base = "https://api.burncloud.ai"
        self.test_audio_file: FileTypes = ("test.wav", b"fake audio content")

    def test_get_complete_url_with_valid_base(self) -> None:
        """Test get_complete_url with valid API base."""
        result = self.config.get_complete_url(
            api_base=self.test_api_base,
            api_key=self.test_api_key,
            model=self.test_model,
            optional_params={},
            litellm_params={}
        )

        expected = f"{self.test_api_base}/v1/audio/transcriptions"
        self.assertEqual(result, expected)

    def test_get_complete_url_with_v1_base(self) -> None:
        """Test get_complete_url when API base ends with /v1."""
        api_base_with_v1 = f"{self.test_api_base}/v1"

        result = self.config.get_complete_url(
            api_base=api_base_with_v1,
            api_key=self.test_api_key,
            model=self.test_model,
            optional_params={},
            litellm_params={}
        )

        expected = f"{api_base_with_v1}/audio/transcriptions"
        self.assertEqual(result, expected)

    def test_get_complete_url_without_api_base(self) -> None:
        """Test get_complete_url when API base is None."""
        with patch("litellm.llms.burncloud.audio_transcription.transformation.get_secret_str") as mock_get_secret:
            mock_get_secret.return_value = self.test_api_base

            result = self.config.get_complete_url(
                api_base=None,
                api_key=self.test_api_key,
                model=self.test_model,
                optional_params={},
                litellm_params={}
            )

            expected = f"{self.test_api_base}/v1/audio/transcriptions"
            self.assertEqual(result, expected)

    def test_get_supported_openai_params(self) -> None:
        """Test that all supported OpenAI parameters are returned."""
        result = self.config.get_supported_openai_params(self.test_model)

        expected = [
            "language",
            "prompt",
            "response_format",
            "temperature",
            "timestamp_granularities",
        ]
        self.assertEqual(result, expected)

    def test_map_openai_params_with_supported_params(self) -> None:
        """Test mapping of supported OpenAI parameters."""
        non_default_params = {
            "language": "en",
            "temperature": 0.5,
            "prompt": "test prompt"
        }
        optional_params: Dict = {}
        drop_params = False

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.test_model,
            drop_params=drop_params
        )

        self.assertEqual(result, non_default_params)

    def test_map_openai_params_with_unsupported_params(self) -> None:
        """Test that unsupported parameters are not included."""
        non_default_params = {
            "unsupported_param": "value",
            "temperature": 0.7
        }
        optional_params: Dict = {}
        drop_params = False

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.test_model,
            drop_params=drop_params
        )

        expected = {"temperature": 0.7}
        self.assertEqual(result, expected)

    def test_validate_environment_with_api_key(self) -> None:
        """Test environment validation with API key provided."""
        headers: Dict = {}
        messages: List[AllMessageValues] = []
        optional_params: Dict = {}
        litellm_params: Dict = {}

        with patch("litellm.llms.burncloud.audio_transcription.transformation.get_secret_str") as mock_get_secret:
            mock_get_secret.return_value = self.test_api_key

            result = self.config.validate_environment(
                headers=headers,
                model=self.test_model,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                api_key=None,
                api_base=None
            )

            expected = {
                "Authorization": f"Bearer {self.test_api_key}",
            }
            self.assertEqual(result, expected)

    def test_validate_environment_with_existing_auth_header(self) -> None:
        """Test environment validation with existing Authorization header."""
        custom_auth = "Custom token"
        headers = {"Authorization": custom_auth}
        messages: List[AllMessageValues] = []
        optional_params: Dict = {}
        litellm_params: Dict = {}

        result = self.config.validate_environment(
            headers=headers,
            model=self.test_model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=self.test_api_key,
            api_base=None
        )

        expected = {"Authorization": custom_auth}
        self.assertEqual(result, expected)

    def test_transform_audio_transcription_request(self) -> None:
        """Test transformation of audio transcription request."""
        optional_params = {"language": "en", "temperature": 0.5}
        litellm_params: Dict = {}

        result = self.config.transform_audio_transcription_request(
            model=self.test_model,
            audio_file=self.test_audio_file,
            optional_params=optional_params,
            litellm_params=litellm_params
        )

        self.assertIsInstance(result, AudioTranscriptionRequestData)
        self.assertEqual(result.data["model"], self.test_model)
        self.assertEqual(result.data["file"], self.test_audio_file)
        self.assertEqual(result.data["language"], "en")
        self.assertEqual(result.data["temperature"], 0.5)

    def test_get_error_class(self) -> None:
        """Test creation of error class."""
        error_message = "Test error"
        status_code = 400
        headers: Union[dict, Headers] = {"Content-Type": "application/json"}

        result = self.config.get_error_class(error_message, status_code, headers)

        self.assertIsInstance(result, BurnCloudError)
        self.assertEqual(result.status_code, status_code)
        self.assertEqual(result.message, error_message)
        self.assertEqual(result.headers, headers)

    def test_transform_audio_transcription_response_success(self) -> None:
        """Test successful transformation of audio transcription response."""
        mock_response = Mock(spec=httpx.Response)
        response_data = {"text": "Transcribed text"}
        mock_response.json.return_value = response_data
        mock_response.text = str(response_data)

        result = self.config.transform_audio_transcription_response(mock_response)

        self.assertIsInstance(result, TranscriptionResponse)
        self.assertEqual(result.text, "Transcribed text")

    def test_transform_audio_transcription_response_invalid_format(self) -> None:
        """Test transformation with invalid response format."""
        mock_response = Mock(spec=httpx.Response)
        response_data = {"invalid_field": "value"}
        mock_response.json.return_value = response_data
        mock_response.text = str(response_data)

        with self.assertRaises(ValueError) as context:
            self.config.transform_audio_transcription_response(mock_response)

        self.assertIn("Invalid response format", str(context.exception))

    def test_transform_audio_transcription_response_json_error(self) -> None:
        """Test transformation when JSON parsing fails."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.side_effect = Exception("JSON decode error")
        mock_response.text = "Invalid JSON"

        with self.assertRaises(ValueError) as context:
            self.config.transform_audio_transcription_response(mock_response)

        self.assertIn("Error transforming response to json", str(context.exception))
