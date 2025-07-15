import os
import sys
import pytest
from unittest.mock import patch, MagicMock
import httpx

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm.types.utils import HttpxBinaryResponseContent


class TestElevenLabsTextToSpeech:
    def test_elevenlabs_tts_request_transformation(self):
        """
        Test that the ElevenLabs TTS request is correctly transformed
        """
        from litellm.llms.elevenlabs.text_to_speech.transformation import (
            ElevenLabsTextToSpeechConfig,
        )

        config = ElevenLabsTextToSpeechConfig()
        
        # Test basic request transformation
        optional_params = {
            "voice_id": "21m00Tcm4TlvDq8ikWAM",
            "response_format": "mp3",
            "speed": 1.0,
        }
        
        request_data = config.transform_audio_speech_request(
            model="elevenlabs/eleven_multilingual_v2",
            input="Hello, this is a test of ElevenLabs TTS",
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )
        
        assert request_data["text"] == "Hello, this is a test of ElevenLabs TTS"
        assert request_data["model_id"] == "eleven_multilingual_v2"
        assert optional_params["_voice_id"] == "21m00Tcm4TlvDq8ikWAM"
        
    def test_elevenlabs_url_construction(self):
        """
        Test that the ElevenLabs URL is correctly constructed
        """
        from litellm.llms.elevenlabs.text_to_speech.transformation import (
            ElevenLabsTextToSpeechConfig,
        )

        config = ElevenLabsTextToSpeechConfig()
        
        optional_params = {
            "_voice_id": "21m00Tcm4TlvDq8ikWAM",
            "output_format": "mp3_44100_128",
        }
        
        url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model="elevenlabs/eleven_multilingual_v2",
            optional_params=optional_params,
            litellm_params={},
        )
        
        expected_url = "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM?output_format=mp3_44100_128"
        assert url == expected_url

    def test_elevenlabs_openai_param_mapping(self):
        """
        Test that OpenAI parameters are correctly mapped to ElevenLabs format
        """
        from litellm.llms.elevenlabs.text_to_speech.transformation import (
            ElevenLabsTextToSpeechConfig,
        )

        config = ElevenLabsTextToSpeechConfig()
        
        non_default_params = {
            "voice": "alloy",
            "response_format": "wav",
            "speed": 1.2,
        }
        
        optional_params = {}
        
        mapped_params = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="elevenlabs/eleven_multilingual_v2",
            drop_params=False,
        )
        
        assert mapped_params["voice_id"] == "alloy"
        assert mapped_params["output_format"] == "pcm_16000"  # wav maps to pcm_16000
        assert mapped_params["speed"] == 1.2

    @patch('litellm.llms.custom_httpx.http_handler._get_httpx_client')
    def test_elevenlabs_speech_call(self, mock_get_client):
        """
        Test that ElevenLabs speech call works end-to-end
        """
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake audio data"
        mock_response.headers = {"Content-Type": "audio/mpeg"}
        
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        # Mock environment
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test-key"}):
            response = litellm.speech(
                model="elevenlabs/eleven_multilingual_v2",
                input="Hello, this is a test",
                voice="21m00Tcm4TlvDq8ikWAM",
                response_format="mp3",
            )
            
            assert isinstance(response, HttpxBinaryResponseContent)
            
        # Verify the request was made correctly
        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args.kwargs
        
        assert "text-to-speech/21m00Tcm4TlvDq8ikWAM" in call_kwargs["url"]
        assert call_kwargs["json"]["text"] == "Hello, this is a test"
        assert call_kwargs["json"]["model_id"] == "eleven_multilingual_v2"
        assert call_kwargs["headers"]["xi-api-key"] == "test-key"

    def test_elevenlabs_supported_params(self):
        """
        Test that supported OpenAI parameters are correctly identified
        """
        from litellm.llms.elevenlabs.text_to_speech.transformation import (
            ElevenLabsTextToSpeechConfig,
        )

        config = ElevenLabsTextToSpeechConfig()
        supported_params = config.get_supported_openai_params("elevenlabs/eleven_multilingual_v2")
        
        expected_params = ["voice", "response_format", "speed"]
        assert supported_params == expected_params