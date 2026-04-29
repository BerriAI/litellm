"""
Tests for CAMB AI TTS integration
"""

import httpx
import pytest
from unittest.mock import MagicMock, patch

from litellm.llms.camb_ai.text_to_speech.transformation import (
    CambAITextToSpeechConfig,
)
from litellm.types.utils import LlmProviders


class TestCambAITextToSpeechConfig:
    def setup_method(self):
        self.config = CambAITextToSpeechConfig()

    def test_provider_enum_exists(self):
        assert LlmProviders.CAMB_AI == "camb_ai"

    def test_get_supported_openai_params(self):
        params = self.config.get_supported_openai_params(model="mars-flash")
        assert "voice" in params
        assert "response_format" in params
        assert "language" in params
        assert "speed" not in params

    def test_map_openai_params_voice_string(self):
        voice, params = self.config.map_openai_params(
            model="mars-flash",
            optional_params={},
            voice="123",
        )
        assert voice == "123"
        assert params["voice_id"] == 123

    def test_map_openai_params_voice_dict(self):
        voice, params = self.config.map_openai_params(
            model="mars-flash",
            optional_params={},
            voice={"voice_id": "456"},
        )
        assert voice == "456"
        assert params["voice_id"] == 456

    def test_map_openai_params_response_format(self):
        voice, params = self.config.map_openai_params(
            model="mars-flash",
            optional_params={"response_format": "mp3"},
            voice="123",
        )
        assert params["output_configuration"] == {"format": "mp3"}

    def test_map_openai_params_speed_dropped(self):
        voice, params = self.config.map_openai_params(
            model="mars-flash",
            optional_params={"speed": 1.5},
            voice="123",
        )
        assert "speed" not in params

    def test_get_complete_url_default(self):
        url = self.config.get_complete_url(
            model="mars-flash",
            api_base=None,
            litellm_params={},
        )
        assert url == "https://client.camb.ai/apis/tts-stream"

    def test_get_complete_url_custom_base(self):
        url = self.config.get_complete_url(
            model="mars-flash",
            api_base="https://custom.camb.ai/v2",
            litellm_params={},
        )
        assert url == "https://custom.camb.ai/v2/tts-stream"

    def test_validate_environment_with_key(self):
        headers = self.config.validate_environment(
            headers={},
            model="mars-flash",
            api_key="test-key-123",
        )
        assert headers["x-api-key"] == "test-key-123"
        assert headers["Content-Type"] == "application/json"

    @patch.dict("os.environ", {}, clear=False)
    def test_validate_environment_missing_key(self, monkeypatch):
        monkeypatch.delenv("CAMB_API_KEY", raising=False)
        monkeypatch.delenv("CAMB_AI_API_KEY", raising=False)
        import litellm
        original_api_key = litellm.api_key
        litellm.api_key = None
        try:
            with pytest.raises(ValueError, match="CAMB AI API key is required"):
                self.config.validate_environment(
                    headers={},
                    model="mars-flash",
                    api_key=None,
                )
        finally:
            litellm.api_key = original_api_key

    def test_transform_text_to_speech_request(self):
        result = self.config.transform_text_to_speech_request(
            model="mars-flash",
            input="Hello world",
            voice="123",
            optional_params={"voice_id": 123},
            litellm_params={"language": "en-us"},
            headers={},
        )
        body = result["dict_body"]
        assert body["text"] == "Hello world"
        assert body["language"] == "en-us"
        assert body["speech_model"] == "mars-flash"
        assert body["voice_id"] == 123

    def test_transform_text_to_speech_request_default_language(self):
        result = self.config.transform_text_to_speech_request(
            model="mars-pro",
            input="Test",
            voice="456",
            optional_params={},
            litellm_params={},
            headers={},
        )
        body = result["dict_body"]
        assert body["language"] == "en-us"

    def test_transform_text_to_speech_request_response_format(self):
        result = self.config.transform_text_to_speech_request(
            model="mars-flash",
            input="Test",
            voice="123",
            optional_params={"response_format": "wav"},
            litellm_params={},
            headers={},
        )
        body = result["dict_body"]
        assert body["output_configuration"] == {"format": "wav"}

    def test_transform_text_to_speech_request_speed_dropped(self):
        result = self.config.transform_text_to_speech_request(
            model="mars-flash",
            input="Test",
            voice="123",
            optional_params={"speed": 1.5},
            litellm_params={},
            headers={},
        )
        body = result["dict_body"]
        assert "speed" not in body

    def test_transform_text_to_speech_response(self):
        mock_response = MagicMock(spec=httpx.Response)
        mock_logging = MagicMock()
        result = self.config.transform_text_to_speech_response(
            model="mars-flash",
            raw_response=mock_response,
            logging_obj=mock_logging,
        )
        assert result is not None

    def test_get_error_class(self):
        from litellm.llms.camb_ai.common_utils import CambAIException

        error = self.config.get_error_class(
            error_message="test error",
            status_code=400,
            headers={},
        )
        assert isinstance(error, CambAIException)
        assert error.status_code == 400
        assert error.message == "test error"
