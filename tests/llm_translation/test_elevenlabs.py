import os
import sys

from typing import Any, Dict

import pytest
from unittest.mock import patch, MagicMock
import httpx

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from base_audio_transcription_unit_tests import BaseLLMAudioTranscriptionTest

os.environ.setdefault("ELEVENLABS_API_KEY", "test-elevenlabs-key")


class TestElevenLabsAudioTranscription(BaseLLMAudioTranscriptionTest):
    def get_base_audio_transcription_call_args(self) -> dict:
        return {
            "model": "elevenlabs/scribe_v1",
        }

    def get_custom_llm_provider(self) -> litellm.LlmProviders:
        return litellm.LlmProviders.ELEVENLABS

    def test_elevenlabs_diarize_parameter_passthrough(self):
        """
        Test that provider-specific parameters like diarize=True get passed through 
        to the ElevenLabs request form data.
        """
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"text": "Four score and seven years ago", "language_code": "en"}'
        mock_response.json.return_value = {
            "text": "Four score and seven years ago",
            "language_code": "en",
            "words": [
                {"type": "word", "text": "Four", "start": 0.0, "end": 0.5},
                {"type": "word", "text": "score", "start": 0.5, "end": 1.0}
            ]
        }
        
        # Create a mock audio file
        audio_content = b"fake audio data"
        
        captured_request_data = {}
        
        def mock_post(*args, **kwargs):
            # Capture the request data for verification
            captured_request_data.update({
                'url': kwargs.get('url'),
                'data': kwargs.get('data'),
                'files': kwargs.get('files'),
                'headers': kwargs.get('headers'),
                'json': kwargs.get('json')
            })
            return mock_response
        
        # Mock the HTTPHandler.post method which is what actually makes the request
        from litellm.llms.custom_httpx.http_handler import HTTPHandler
        
        with patch.object(HTTPHandler, 'post', side_effect=mock_post):
            try:
                result = litellm.transcription(
                    model="elevenlabs/scribe_v1",
                    file=audio_content,
                    diarize=True,  # This should be passed through to the form data
                    language="en",  # This should be mapped to language_code
                    temperature=0.5,  # This should also be passed through
                    custom_param="test_value"  # This should also be passed through
                )
                
                # Verify the request was made with correct form data
                assert 'speech-to-text' in captured_request_data['url']
                
                # Check that form data contains the expected parameters
                form_data = captured_request_data['data']
                assert form_data is not None, "Form data should not be None"
                
                print(f"✅ Captured form data: {form_data}")
                
                # Check basic required parameters
                assert 'model_id' in form_data, "model_id should be in form data"
                assert form_data['model_id'] == 'scribe_v1', f"Expected model_id 'scribe_v1', got {form_data['model_id']}"
                
                # Check that diarize parameter is passed through
                assert 'diarize' in form_data, f"diarize should be in form data. Got: {list(form_data.keys())}"
                assert form_data['diarize'] == 'True', f"Expected diarize='True', got {form_data['diarize']}"
                
                # Check that OpenAI language parameter is mapped correctly
                assert 'language_code' in form_data, "language_code should be in form data"
                assert form_data['language_code'] == 'en', f"Expected language_code='en', got {form_data['language_code']}"
                
                # Check that temperature is passed through
                assert 'temperature' in form_data, "temperature should be in form data"
                assert form_data['temperature'] == '0.5', f"Expected temperature='0.5', got {form_data['temperature']}"
                
                # Check that custom parameters are passed through
                assert 'custom_param' in form_data, "custom_param should be in form data"
                assert form_data['custom_param'] == 'test_value', f"Expected custom_param='test_value', got {form_data['custom_param']}"
                
                # Check that files are included
                files = captured_request_data['files']
                assert files is not None, "Files should not be None"
                assert 'file' in files, "file should be in files"
                
                print("✅ All parameter passthrough tests passed!")
                
            except Exception as e:
                print(f"❌ Test failed: {e}")
                print(f"Captured request data: {captured_request_data}")
                raise 


class TestElevenLabsTextToSpeechTransformation:
    @pytest.fixture(scope="class")
    def config(self):
        from litellm.llms.elevenlabs.text_to_speech.transformation import (
            ElevenLabsTextToSpeechConfig,
        )

        return ElevenLabsTextToSpeechConfig()

    def test_map_openai_params_maps_voice_and_speed(self, config):
        kwargs: Dict[str, Any] = {}
        mapped_voice, mapped_params = config.map_openai_params(
            model="eleven_multilingual_v2",
            optional_params={
                "response_format": "mp3",
                "speed": 1.25,
                "model_id": "eleven_multilingual_v2",
            },
            voice="alloy",
            kwargs=kwargs,
        )

        assert mapped_voice == config.VOICE_MAPPINGS["alloy"]
        assert mapped_params["voice_settings"]["speed"] == pytest.approx(1.25)
        assert (
            kwargs[config.ELEVENLABS_QUERY_PARAMS_KEY]["output_format"]
            == "mp3_44100_128"
        )

    def test_transform_request_and_url(self, config):
        kwargs: Dict[str, Any] = {}
        voice_id, optional_params = config.map_openai_params(
            model="eleven_multilingual_v2",
            optional_params={
                "response_format": "pcm",
                "model_id": "eleven_multilingual_v2",
                "pronunciation_dictionary_locators": [
                    {"pronunciation_dictionary_id": "dict_1"}
                ],
            },
            voice="alloy",
            kwargs=kwargs,
        )

        litellm_params: Dict[str, Any] = {
            config.ELEVENLABS_VOICE_ID_KEY: voice_id,
            config.ELEVENLABS_QUERY_PARAMS_KEY: kwargs[
                config.ELEVENLABS_QUERY_PARAMS_KEY
            ],
        }

        headers = config.validate_environment(
            headers={}, model="eleven_multilingual_v2", api_key="test-key"
        )

        request_data = config.transform_text_to_speech_request(
            model="eleven_multilingual_v2",
            input="Hello world",
            voice=voice_id,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        assert request_data["dict_body"]["text"] == "Hello world"
        assert request_data["dict_body"]["model_id"] == "eleven_multilingual_v2"
        assert request_data["dict_body"]["pronunciation_dictionary_locators"] == [
            {"pronunciation_dictionary_id": "dict_1"}
        ]

        url = config.get_complete_url(
            model="eleven_multilingual_v2",
            api_base=None,
            litellm_params=litellm_params,
        )

        assert voice_id in url
        assert "output_format=pcm_44100" in url