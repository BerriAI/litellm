import os
import sys
from unittest.mock import MagicMock, Mock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))  # Adds the parent directory to the system path

import litellm
from litellm.llms.vertex_ai.text_to_speech.transformation import (
    VertexAILyriaTextToSpeechConfig,
    VertexAITextToSpeechConfig,
)
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


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
    def test_transform_text_to_speech_request_body(self, mock_get_token, mock_ensure_token):
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
        voice_str, voice_dict = config._map_voice_to_vertex_format("en-US-Chirp3-HD-Charon")

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


class TestVertexAILyriaTextToSpeechConfig:
    @pytest.mark.parametrize(
        "model",
        ["lyria-002", "vertex_ai/lyria-3-clip-preview", "lyria-3-pro-preview"],
    )
    def test_provider_config_manager_selects_lyria_config(self, model):
        config = ProviderConfigManager.get_provider_text_to_speech_config(
            model=model,
            provider=LlmProviders.VERTEX_AI,
        )

        assert isinstance(config, VertexAILyriaTextToSpeechConfig)

    @pytest.mark.parametrize(
        ("model", "vertex_ai_audio_api", "supported_audio_formats", "expected_url"),
        [
            (
                "future-lyria-predict",
                "lyria_predict",
                ["wav"],
                "https://us-central1-aiplatform.googleapis.com/v1/projects/music-project/locations/"
                "us-central1/publishers/google/models/future-lyria-predict:predict",
            ),
            (
                "future-music-interactions",
                "lyria_interactions",
                ["mp3", "wav"],
                "https://aiplatform.googleapis.com/v1beta1/projects/music-project/locations/global/interactions",
            ),
        ],
    )
    def test_dispatches_from_model_metadata(
        self,
        monkeypatch,
        model,
        vertex_ai_audio_api,
        supported_audio_formats,
        expected_url,
    ):
        monkeypatch.setitem(
            litellm.model_cost,
            f"vertex_ai/{model}",
            {
                "vertex_ai_audio_api": vertex_ai_audio_api,
                "supported_audio_formats": supported_audio_formats,
            },
        )

        config = ProviderConfigManager.get_provider_text_to_speech_config(
            model=model,
            provider=LlmProviders.VERTEX_AI,
        )

        assert isinstance(config, VertexAILyriaTextToSpeechConfig)
        assert (
            config.get_complete_url(
                model=model,
                api_base=None,
                litellm_params={
                    "vertex_project": "music-project",
                    "vertex_location": "us-central1",
                },
            )
            == expected_url
        )

    def test_vertex_chirp_does_not_select_lyria_config(self):
        config = ProviderConfigManager.get_provider_text_to_speech_config(
            model="chirp",
            provider=LlmProviders.VERTEX_AI,
        )

        assert isinstance(config, VertexAITextToSpeechConfig)
        assert not isinstance(config, VertexAILyriaTextToSpeechConfig)

    def test_get_complete_url_for_lyria_2(self):
        config = VertexAILyriaTextToSpeechConfig()

        url = config.get_complete_url(
            model="lyria-002",
            api_base=None,
            litellm_params={
                "vertex_project": "music-project",
                "vertex_location": "europe-west4",
            },
        )

        assert url == (
            "https://europe-west4-aiplatform.googleapis.com/v1/projects/music-project/"
            "locations/europe-west4/publishers/google/models/lyria-002:predict"
        )

    def test_get_complete_url_for_lyria_3(self):
        config = VertexAILyriaTextToSpeechConfig()

        url = config.get_complete_url(
            model="lyria-3-pro-preview",
            api_base=None,
            litellm_params={"vertex_project": "music-project"},
        )

        assert url == ("https://aiplatform.googleapis.com/v1beta1/projects/music-project/locations/global/interactions")

    @pytest.mark.parametrize(
        ("model", "response_format", "expected_body"),
        [
            (
                "lyria-002",
                "wav",
                {
                    "instances": [{"prompt": "A bright synth track"}],
                    "parameters": {"sample_count": 1},
                },
            ),
            (
                "lyria-3-clip-preview",
                "mp3",
                {
                    "model": "lyria-3-clip-preview",
                    "input": "A bright synth track",
                },
            ),
            (
                "lyria-3-pro-preview",
                "wav",
                {
                    "model": "lyria-3-pro-preview",
                    "input": "A bright synth track",
                    "response_format": {
                        "type": "audio",
                        "mime_type": "audio/wav",
                    },
                },
            ),
        ],
    )
    @patch.object(VertexAILyriaTextToSpeechConfig, "_ensure_access_token")
    def test_transform_request(
        self,
        mock_ensure_token,
        model,
        response_format,
        expected_body,
    ):
        mock_ensure_token.return_value = ("mock-token", "music-project")
        config = VertexAILyriaTextToSpeechConfig()

        request = config.transform_text_to_speech_request(
            model=model,
            input="A bright synth track",
            voice="alloy",
            optional_params={"response_format": response_format},
            litellm_params={"vertex_project": "music-project"},
            headers={},
        )

        assert request["dict_body"] == expected_body
        assert request["headers"]["Authorization"] == "Bearer mock-token"
        assert request["headers"]["x-goog-user-project"] == "music-project"

    @pytest.mark.parametrize(
        ("model", "response_json", "expected_audio", "expected_mime_type"),
        [
            (
                "lyria-002",
                {
                    "predictions": [
                        {
                            "bytesBase64Encoded": "bHlyaWEtMi1hdWRpbw==",
                        }
                    ]
                },
                b"lyria-2-audio",
                "audio/wav",
            ),
            (
                "lyria-3-pro-preview",
                {
                    "steps": [
                        {
                            "type": "model_output",
                            "content": [
                                {"type": "text", "text": "Generated lyrics"},
                                {
                                    "type": "audio",
                                    "data": "bHlyaWEtMy1hdWRpbw==",
                                    "mime_type": "audio/mpeg",
                                },
                            ],
                        }
                    ]
                },
                b"lyria-3-audio",
                "audio/mpeg",
            ),
            (
                "lyria-3-clip-preview",
                {
                    "outputs": [
                        {"type": "text", "text": "Generated lyrics"},
                        {
                            "type": "audio",
                            "data": "bHlyaWEtMy1hdWRpbw==",
                            "mime_type": "audio/mpeg",
                        },
                    ]
                },
                b"lyria-3-audio",
                "audio/mpeg",
            ),
        ],
    )
    def test_transform_response(
        self,
        model,
        response_json,
        expected_audio,
        expected_mime_type,
    ):
        config = VertexAILyriaTextToSpeechConfig()
        raw_response = httpx.Response(200, json=response_json)

        response = config.transform_text_to_speech_response(
            model=model,
            raw_response=raw_response,
            logging_obj=MagicMock(),
        )

        assert response.content == expected_audio
        assert response._hidden_params["audio_mime_type"] == expected_mime_type

    @pytest.mark.parametrize(
        ("model", "response_format"),
        [
            ("lyria-002", "mp3"),
            ("lyria-3-clip-preview", "wav"),
            ("lyria-3-pro-preview", "opus"),
        ],
    )
    def test_rejects_unsupported_response_format(self, model, response_format):
        config = VertexAILyriaTextToSpeechConfig()

        with pytest.raises(litellm.UnsupportedParamsError):
            config.map_openai_params(
                model=model,
                optional_params={"response_format": response_format},
            )

    @pytest.mark.parametrize("param", ["speed", "instructions"])
    def test_rejects_unsupported_openai_params(self, param):
        config = VertexAILyriaTextToSpeechConfig()

        with pytest.raises(litellm.UnsupportedParamsError):
            config.map_openai_params(
                model="lyria-3-pro-preview",
                optional_params={param: "unsupported"},
            )

    @pytest.mark.parametrize(
        ("model", "response_format", "response_json", "expected_url", "expected_body"),
        [
            (
                "lyria-002",
                "wav",
                {
                    "predictions": [
                        {
                            "audioContent": "bHlyaWEtMi1hdWRpbw==",
                            "mimeType": "audio/wav",
                        }
                    ]
                },
                "https://us-central1-aiplatform.googleapis.com/v1/projects/music-project/locations/us-central1/publishers/google/models/lyria-002:predict",
                {
                    "instances": [{"prompt": "A bright synth track"}],
                    "parameters": {"sample_count": 1},
                },
            ),
            (
                "lyria-3-pro-preview",
                "mp3",
                {
                    "steps": [
                        {
                            "type": "model_output",
                            "content": [
                                {
                                    "type": "audio",
                                    "data": "bHlyaWEtMy1hdWRpbw==",
                                    "mime_type": "audio/mpeg",
                                }
                            ],
                        }
                    ]
                },
                "https://aiplatform.googleapis.com/v1beta1/projects/music-project/locations/global/interactions",
                {
                    "model": "lyria-3-pro-preview",
                    "input": "A bright synth track",
                },
            ),
        ],
    )
    def test_litellm_speech_dispatches_to_lyria_api(
        self,
        model,
        response_format,
        response_json,
        expected_url,
        expected_body,
    ):
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = response_json
        with (
            patch.object(
                VertexAILyriaTextToSpeechConfig,
                "_ensure_access_token",
                return_value=("mock-token", "music-project"),
            ),
            patch(
                "litellm.llms.custom_httpx.llm_http_handler.HTTPHandler.post",
                return_value=mock_response,
            ) as mock_post,
        ):
            response = litellm.speech(
                model=f"vertex_ai/{model}",
                input="A bright synth track",
                voice="alloy",
                response_format=response_format,
                vertex_project="music-project",
                vertex_location="us-central1",
            )

        assert response.content in {b"lyria-2-audio", b"lyria-3-audio"}
        mock_post.assert_called_once()
        assert mock_post.call_args.kwargs["url"] == expected_url
        assert mock_post.call_args.kwargs["json"] == expected_body


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
    assert "json" in call_kwargs
    request_body = call_kwargs["json"]

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
