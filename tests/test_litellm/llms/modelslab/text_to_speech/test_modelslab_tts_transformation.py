"""
Tests for ModelsLab TTS transformation. All mocked — no real network calls.
"""
from unittest.mock import MagicMock, Mock, patch

import httpx
import pytest

from litellm.llms.modelslab.text_to_speech.transformation import (
    ModelsLabTextToSpeechConfig,
    VOICE_MAPPINGS,
)


class TestModelsLabTTSTransformation:

    def setup_method(self):
        self.config = ModelsLabTextToSpeechConfig()
        self.config._api_key = "test-api-key"
        self.mock_logging = Mock()

    # -------------------------------------------------------------------------
    # validate_environment
    # -------------------------------------------------------------------------

    def test_validate_environment_no_auth_header(self):
        """Key-in-body auth: only Content-Type, no Authorization header."""
        with patch(
            "litellm.llms.modelslab.text_to_speech.transformation.get_secret_str",
            return_value="test-key",
        ):
            headers = self.config.validate_environment(headers={}, model="default")
        assert headers["Content-Type"] == "application/json"
        assert "Authorization" not in headers

    def test_validate_environment_raises_without_key(self):
        with patch(
            "litellm.llms.modelslab.text_to_speech.transformation.get_secret_str",
            return_value=None,
        ):
            with pytest.raises(ValueError, match="MODELSLAB_API_KEY"):
                self.config.validate_environment(headers={}, model="default")

    # -------------------------------------------------------------------------
    # map_openai_params / voice mapping
    # -------------------------------------------------------------------------

    def test_map_openai_params_voice_mapping(self):
        """OpenAI voice names map to ModelsLab voice_id integers."""
        _, params = self.config.map_openai_params(
            model="default", optional_params={}, voice="nova"
        )
        assert params["voice_id"] == VOICE_MAPPINGS["nova"]

    def test_map_openai_params_speed(self):
        _, params = self.config.map_openai_params(
            model="default", optional_params={"speed": "1.5"}, voice="alloy"
        )
        assert params["speed"] == 1.5

    def test_get_supported_openai_params(self):
        params = self.config.get_supported_openai_params("default")
        assert "voice" in params
        assert "response_format" in params
        assert "speed" in params

    # -------------------------------------------------------------------------
    # transform_text_to_speech_request
    # -------------------------------------------------------------------------

    def test_transform_request_key_in_body(self):
        """API key must be in body, not headers."""
        result = self.config.transform_text_to_speech_request(
            model="default",
            input="Hello world",
            voice="alloy",
            optional_params={"language": "english", "voice_id": 1, "speed": 1.0},
            litellm_params={},
            headers={},
        )
        body = result["dict_body"]
        assert body["key"] == "test-api-key"
        assert body["prompt"] == "Hello world"
        assert "language" in body

    def test_transform_request_voice_default(self):
        """Unknown voice defaults to voice_id 1."""
        result = self.config.transform_text_to_speech_request(
            model="default",
            input="Test",
            voice="unknown_voice",
            optional_params={"language": "english", "speed": 1.0},
            litellm_params={},
            headers={},
        )
        body = result["dict_body"]
        assert body["voice_id"] == 1  # default

    # -------------------------------------------------------------------------
    # transform_text_to_speech_response
    # -------------------------------------------------------------------------

    def test_transform_response_success_downloads_audio(self):
        """Success response fetches audio URL and returns HttpxBinaryResponseContent."""
        mock_resp = Mock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.headers = {}
        mock_resp.json.return_value = {
            "status": "success",
            "output": "https://cdn.modelslab.com/output/audio.mp3",
        }

        mock_audio_resp = Mock(spec=httpx.Response)
        mock_audio_resp.status_code = 200
        mock_audio_resp.content = b"fake-audio-bytes"

        with patch.object(self.config, "_poll_tts_sync") as mock_poll, \
             patch(
                 "litellm.llms.modelslab.text_to_speech.transformation._get_httpx_client"
             ) as mock_client:
            mock_client.return_value.get.return_value = mock_audio_resp
            result = self.config.transform_text_to_speech_response(
                model="default",
                raw_response=mock_resp,
                logging_obj=self.mock_logging,
            )
            mock_poll.assert_not_called()  # no polling needed for success

        assert result is not None

    def test_transform_response_processing_polls(self):
        """Processing response triggers polling then downloads audio."""
        mock_resp = Mock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.headers = {}
        mock_resp.json.return_value = {
            "status": "processing",
            "request_id": "req_abc",
            "eta": 5,
        }

        poll_result = {
            "status": "success",
            "output": "https://cdn.modelslab.com/output/audio2.mp3",
        }

        mock_audio_resp = Mock(spec=httpx.Response)
        mock_audio_resp.status_code = 200
        mock_audio_resp.content = b"fake-audio-bytes-2"

        with patch.object(
            self.config, "_poll_tts_sync", return_value=poll_result
        ) as mock_poll, patch(
            "litellm.llms.modelslab.text_to_speech.transformation._get_httpx_client"
        ) as mock_client:
            mock_client.return_value.get.return_value = mock_audio_resp
            result = self.config.transform_text_to_speech_response(
                model="default",
                raw_response=mock_resp,
                logging_obj=self.mock_logging,
            )
            mock_poll.assert_called_once_with("req_abc")

        assert result is not None
