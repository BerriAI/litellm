import json
import os
import sys
from unittest.mock import MagicMock, Mock, patch

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.llms.vertex_ai.text_to_speech.transformation import (
    VertexAITextToSpeechConfig,
)


class TestVertexAITextToSpeechConfig:
    """Tests for VertexAITextToSpeechConfig transformation"""

    def test_get_complete_url(self):
        """Test that get_complete_url returns the correct Google Cloud TTS API URL"""
        config = VertexAITextToSpeechConfig()

        url = config.get_complete_url(
            model="vertex_ai/chirp",
            api_base=None,
            litellm_params={},
        )

        assert url == "https://texttospeech.googleapis.com/v1/text:synthesize"

    def test_get_complete_url_with_custom_api_base(self):
        """Test that get_complete_url uses custom api_base when provided"""
        config = VertexAITextToSpeechConfig()

        custom_url = "https://custom-tts-endpoint.example.com/v1/synthesize"
        url = config.get_complete_url(
            model="vertex_ai/chirp",
            api_base=custom_url,
            litellm_params={},
        )

        assert url == custom_url

    @patch.object(VertexAITextToSpeechConfig, "_ensure_access_token")
    @patch.object(VertexAITextToSpeechConfig, "_get_token_and_url")
    def test_transform_text_to_speech_request_body(
        self, mock_get_token, mock_ensure_token
    ):
        """Test that transform_text_to_speech_request generates correct request body"""
        # Mock authentication
        mock_ensure_token.return_value = ("mock-token", "test-project")
        mock_get_token.return_value = ("mock-token", "mock-url")

        config = VertexAITextToSpeechConfig()

        # Test with voice dict in litellm_params (as set by dispatch)
        result = config.transform_text_to_speech_request(
            model="vertex_ai/chirp",
            input="Hello, this is a test",
            voice=None,
            optional_params={
                "vertex_voice_dict": {
                    "languageCode": "en-US",
                    "name": "en-US-Chirp3-HD-Charon",
                }
            },
            litellm_params={
                "vertex_credentials": None,
                "vertex_project": "test-project",
                "vertex_location": "us-central1",
            },
            headers={},
        )

        # Verify request body structure
        assert "dict_body" in result
        request_body = result["dict_body"]

        assert "input" in request_body
        assert request_body["input"] == {"text": "Hello, this is a test"}

        assert "voice" in request_body
        assert request_body["voice"]["languageCode"] == "en-US"
        assert request_body["voice"]["name"] == "en-US-Chirp3-HD-Charon"

        assert "audioConfig" in request_body

        # Verify headers contain auth
        assert "headers" in result
        assert "Authorization" in result["headers"]

    def test_voice_mapping_openai_to_vertex(self):
        """Test that OpenAI voice names are correctly mapped to Vertex AI voices"""
        config = VertexAITextToSpeechConfig()

        # Test the _map_voice_to_vertex_format helper
        voice_str, voice_dict = config._map_voice_to_vertex_format("alloy")

        assert voice_str == "alloy"
        assert voice_dict is not None
        assert voice_dict["name"] == "en-US-Studio-O"
        assert voice_dict["languageCode"] == "en-US"

    def test_voice_mapping_vertex_voice_passthrough(self):
        """Test that Vertex AI voice names are passed through directly"""
        config = VertexAITextToSpeechConfig()

        # Test with a Chirp3 HD voice
        voice_str, voice_dict = config._map_voice_to_vertex_format(
            "en-US-Chirp3-HD-Charon"
        )

        assert voice_str == "en-US-Chirp3-HD-Charon"
        assert voice_dict is not None
        assert voice_dict["name"] == "en-US-Chirp3-HD-Charon"
        assert voice_dict["languageCode"] == "en-US"

    def test_voice_mapping_dict_passthrough(self):
        """Test that voice dict is passed through unchanged"""
        config = VertexAITextToSpeechConfig()

        voice_input = {
            "languageCode": "de-DE",
            "name": "de-DE-Chirp3-HD-Charon",
        }
        voice_str, voice_dict = config._map_voice_to_vertex_format(voice_input)

        assert voice_str is None
        assert voice_dict == voice_input


@patch("litellm.llms.custom_httpx.llm_http_handler.HTTPHandler.post")
@patch.object(VertexAITextToSpeechConfig, "_ensure_access_token")
@patch.object(VertexAITextToSpeechConfig, "_get_token_and_url")
def test_litellm_speech_vertex_ai_chirp(mock_get_token, mock_ensure_token, mock_post):
    """
    Test that litellm.speech(model="vertex_ai/chirp") sends the correct URL and request body
    """
    # Mock authentication
    mock_ensure_token.return_value = ("mock-token", "test-project")
    mock_get_token.return_value = ("mock-token", "mock-url")

    # Mock HTTP response
    mock_response = Mock(spec=httpx.Response)
    mock_response.content = b'{"audioContent": "SGVsbG8gV29ybGQ="}'  # base64 encoded "Hello World"
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = {"audioContent": "SGVsbG8gV29ybGQ="}
    mock_post.return_value = mock_response

    litellm.speech(
        model="vertex_ai/chirp",
        input="Hello, this is a test",
        voice="en-US-Chirp3-HD-Charon",
        vertex_project="test-project",
        vertex_location="us-central1",
    )

    # Verify the HTTP call was made
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args.kwargs

    # Verify the URL is the Google Cloud TTS API
    assert call_kwargs["url"] == "https://texttospeech.googleapis.com/v1/text:synthesize"

    # Verify request body structure
    assert "data" in call_kwargs
    request_body = json.loads(call_kwargs["data"])

    # Verify input
    assert "input" in request_body
    assert request_body["input"] == {"text": "Hello, this is a test"}

    # Verify voice
    assert "voice" in request_body
    assert request_body["voice"]["name"] == "en-US-Chirp3-HD-Charon"
    assert request_body["voice"]["languageCode"] == "en-US"

    # Verify audioConfig
    assert "audioConfig" in request_body

    # Verify headers contain authorization
    assert "headers" in call_kwargs
    assert "Authorization" in call_kwargs["headers"]
    assert call_kwargs["headers"]["Authorization"] == "Bearer mock-token"


