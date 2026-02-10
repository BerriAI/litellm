"""
Sarvam Text-to-Speech Transformation Tests

Tests for Sarvam AI TTS integration.
"""

import base64
import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.sarvam.text_to_speech.transformation import (
    SarvamTextToSpeechConfig,
)


@pytest.fixture
def handler():
    return SarvamTextToSpeechConfig()


class TestSpeakerNames:
    """Tests for Sarvam native speaker names"""

    def test_speaker_anushka(self, handler):
        """Test Anushka speaker passes through"""
        voice, params = handler.map_openai_params(
            model="bulbul:v2",
            optional_params={},
            voice="Anushka",
        )
        assert voice == "Anushka"

    def test_speaker_manisha(self, handler):
        """Test Manisha speaker passes through"""
        voice, params = handler.map_openai_params(
            model="bulbul:v2",
            optional_params={},
            voice="Manisha",
        )
        assert voice == "Manisha"

    def test_speaker_abhilash(self, handler):
        """Test Abhilash (male) speaker passes through"""
        voice, params = handler.map_openai_params(
            model="bulbul:v2",
            optional_params={},
            voice="Abhilash",
        )
        assert voice == "Abhilash"

    def test_speaker_default_when_none(self, handler):
        """Test default speaker when none provided"""
        voice, params = handler.map_openai_params(
            model="bulbul:v2",
            optional_params={},
            voice=None,
        )
        assert voice == "anushka"  # Default for v2

    def test_speaker_from_dict(self, handler):
        """Test speaker extraction from dict"""
        voice, params = handler.map_openai_params(
            model="bulbul:v2",
            optional_params={},
            voice={"name": "Vidya"},
        )
        assert voice == "Vidya"


class TestSpeedToPaceMapping:
    """Tests for speed parameter to pace conversion"""

    def test_speed_converts_to_pace(self, handler):
        """Test speed parameter is converted to pace"""
        voice, params = handler.map_openai_params(
            model="bulbul:v2",
            optional_params={"speed": 1.5},
            voice="Anushka",
        )
        assert params["pace"] == 1.5

    def test_speed_not_in_params_when_absent(self, handler):
        """Test pace not added when speed absent"""
        voice, params = handler.map_openai_params(
            model="bulbul:v2",
            optional_params={},
            voice="Anushka",
        )
        assert "pace" not in params


class TestSupportedParams:
    """Tests for supported OpenAI parameters"""

    def test_get_supported_openai_params(self, handler):
        """Test supported parameters"""
        params = handler.get_supported_openai_params("bulbul:v2")
        assert "voice" in params
        assert "response_format" in params
        assert "speed" in params


class TestGetCompleteUrl:
    """Tests for URL generation"""

    def test_get_complete_url_default(self, handler):
        """Test default URL generation"""
        url = handler.get_complete_url(
            model="bulbul:v2",
            api_base=None,
            litellm_params={},
        )
        assert url == "https://api.sarvam.ai/text-to-speech"

    def test_get_complete_url_custom_base(self, handler):
        """Test custom API base"""
        url = handler.get_complete_url(
            model="bulbul:v2",
            api_base="https://custom.sarvam.ai",
            litellm_params={},
        )
        assert url == "https://custom.sarvam.ai/text-to-speech"


class TestValidateEnvironment:
    """Tests for environment validation"""

    def test_validate_environment_with_api_key(self, handler):
        """Test API key is set in headers"""
        headers = handler.validate_environment(
            headers={},
            model="bulbul:v2",
            api_key="test-api-key",
        )
        assert headers["api-subscription-key"] == "test-api-key"

    def test_validate_environment_missing_key_raises(self, handler):
        """Test missing API key raises error"""
        with pytest.raises(ValueError, match="SARVAM_API_KEY"):
            handler.validate_environment(
                headers={},
                model="bulbul:v2",
                api_key=None,
            )


class TestTransformRequest:
    """Tests for request transformation"""

    def test_transform_request_basic(self, handler):
        """Test basic request transformation"""
        result = handler.transform_text_to_speech_request(
            model="bulbul:v2",
            input="Hello world",
            voice="Anushka",
            optional_params={},
            litellm_params={},
            headers={},
        )

        assert "dict_body" in result
        body = result["dict_body"]
        assert body["text"] == "Hello world"
        assert body["model"] == "bulbul:v2"
        assert body["speaker"] == "anushka"

    def test_transform_request_with_language(self, handler):
        """Test request with language code"""
        result = handler.transform_text_to_speech_request(
            model="bulbul:v2",
            input="नमस्ते",
            voice="Manisha",
            optional_params={"target_language_code": "hi-IN"},
            litellm_params={},
            headers={},
        )

        body = result["dict_body"]
        assert body["target_language_code"] == "hi-IN"
        assert body["speaker"] == "manisha"

    def test_transform_request_with_pace(self, handler):
        """Test request with pace parameter"""
        result = handler.transform_text_to_speech_request(
            model="bulbul:v2",
            input="Hello",
            voice="Abhilash",
            optional_params={"pace": 1.2},
            litellm_params={},
            headers={},
        )

        body = result["dict_body"]
        assert body["pace"] == 1.2
        assert body["speaker"] == "abhilash"


class TestTransformResponse:
    """Tests for response transformation"""

    def test_transform_response_decodes_base64(self, handler):
        """Test response decodes base64 audio"""
        mock_audio = b"fake_audio_data"
        mock_audio_base64 = base64.b64encode(mock_audio).decode("utf-8")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "audios": [mock_audio_base64],
        }

        result = handler.transform_text_to_speech_response(
            model="bulbul:v2",
            raw_response=mock_response,
            logging_obj=MagicMock(),
        )

        assert result is not None
        assert result.content == mock_audio

    def test_transform_response_empty_audios_raises(self, handler):
        """Test error when no audio in response"""
        mock_response = MagicMock()
        mock_response.json.return_value = {"audios": []}

        with pytest.raises(ValueError, match="no audio data"):
            handler.transform_text_to_speech_response(
                model="bulbul:v2",
                raw_response=mock_response,
                logging_obj=MagicMock(),
            )