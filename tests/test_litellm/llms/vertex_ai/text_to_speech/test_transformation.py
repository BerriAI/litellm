import base64
from unittest.mock import MagicMock, Mock, patch

import httpx
import pytest

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

    @patch.object(  # test-quality-ok: isolates provider credentials while testing request serialization
        VertexAITextToSpeechConfig, "_ensure_access_token"
    )
    @patch.object(  # test-quality-ok: fixes the provider URL at the authentication boundary
        VertexAITextToSpeechConfig, "_get_token_and_url"
    )
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

    def test_gemini_tts_multi_speaker_voice_mapping(self):
        config = VertexAITextToSpeechConfig()

        voice = {
            "multi_speaker_voice_config": {
                "speaker_voice_configs": [
                    {
                        "speaker": "Ryan",
                        "voice_config": {
                            "prebuilt_voice_config": {
                                "voice_name": "Umbriel",
                            },
                        },
                    },
                    {
                        "speaker": "Katie",
                        "voice_config": {
                            "prebuilt_voice_config": {
                                "voice_name": "Leda",
                            },
                        },
                    },
                ],
            },
        }

        expected_voice = {
            "languageCode": "en-US",
            "modelName": "gemini-3.1-flash-tts-preview",
            "multiSpeakerVoiceConfig": {
                "speakerVoiceConfigs": [
                    {
                        "speakerAlias": "Ryan",
                        "speakerId": "Umbriel",
                    },
                    {
                        "speakerAlias": "Katie",
                        "speakerId": "Leda",
                    },
                ],
            },
        }

        voice_str, optional_params = config.map_openai_params(
            model="gemini-3.1-flash-tts-preview",
            optional_params={"response_format": "mp3"},
            voice=voice,
        )
        assert voice_str is None
        assert optional_params["audioEncoding"] == "MP3"
        assert optional_params["vertex_voice_dict"] == expected_voice

        voice_str, optional_params = config.map_openai_params(
            model="gemini-3.1-flash-tts-preview",
            optional_params={"response_format": "pcm16"},
            voice=voice,
        )
        assert voice_str is None
        assert optional_params["audioEncoding"] == "LINEAR16"
        assert optional_params["vertex_voice_dict"] == expected_voice

    @pytest.mark.parametrize(
        "voice",
        [
            {"name": "Umbriel", "modelName": "gemini-2.5-flash-tts"},
            {"name": "Umbriel", "model_name": "gemini-2.5-flash-tts"},
            {"modelName": "gemini-2.5-flash-tts", "model_name": "chirp-3"},
            {
                "modelName": "gemini-2.5-flash-tts",
                "model_name": "chirp-3",
                "multi_speaker_voice_config": {
                    "speaker_voice_configs": [
                        {
                            "speaker": "Ryan",
                            "voice_config": {
                                "prebuilt_voice_config": {
                                    "voice_name": "Umbriel",
                                },
                            },
                        },
                    ],
                },
            },
        ],
    )
    def test_gemini_tts_ignores_voice_model_name_override(self, voice):
        config = VertexAITextToSpeechConfig()
        routed_model = "gemini-3.1-flash-tts-preview"

        _, optional_params = config.map_openai_params(
            model=routed_model,
            optional_params={"response_format": "mp3"},
            voice=voice,
        )
        assert optional_params["vertex_voice_dict"]["modelName"] == routed_model

    @patch.object(  # test-quality-ok: isolates provider credentials while testing Gemini request serialization
        VertexAITextToSpeechConfig, "_ensure_access_token"
    )
    @patch.object(  # test-quality-ok: fixes the Gemini provider URL at the authentication boundary
        VertexAITextToSpeechConfig, "_get_token_and_url"
    )
    def test_gemini_tts_mp3_request_body(self, mock_get_token, mock_ensure_token):
        mock_ensure_token.return_value = ("mock-token", "test-project")
        mock_get_token.return_value = ("mock-token", "mock-url")
        config = VertexAITextToSpeechConfig()

        result = config.transform_text_to_speech_request(
            model="gemini-3.1-flash-tts-preview",
            input="Ryan: Hi.\nKatie: Hello.",
            voice=None,
            optional_params={
                "audioEncoding": "MP3",
                "vertex_voice_dict": {
                    "languageCode": "en-US",
                    "modelName": "gemini-3.1-flash-tts-preview",
                    "multiSpeakerVoiceConfig": {
                        "speakerVoiceConfigs": [
                            {
                                "speakerAlias": "Ryan",
                                "speakerId": "Umbriel",
                            },
                            {
                                "speakerAlias": "Katie",
                                "speakerId": "Leda",
                            },
                        ],
                    },
                },
            },
            litellm_params={
                "vertex_credentials": None,
                "vertex_project": "test-project",
                "vertex_location": "global",
            },
            headers={},
        )

        request_body = result["dict_body"]
        assert request_body["input"] == {"text": "Ryan: Hi.\nKatie: Hello."}
        assert request_body["voice"] == {
            "languageCode": "en-US",
            "modelName": "gemini-3.1-flash-tts-preview",
            "multiSpeakerVoiceConfig": {
                "speakerVoiceConfigs": [
                    {
                        "speakerAlias": "Ryan",
                        "speakerId": "Umbriel",
                    },
                    {
                        "speakerAlias": "Katie",
                        "speakerId": "Leda",
                    },
                ],
            },
        }
        assert request_body["audioConfig"]["audioEncoding"] == "MP3"


@pytest.mark.parametrize(
    ("audio", "expected_content_type"),
    [
        (b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00", "audio/wav"),
        (b"\xff\xfb\x90\x64\x00\x00\x00\x00", "audio/mpeg"),
        (b"OggS" + b"\x00" * 24 + b"OpusHead", "audio/opus"),
        (b"fLaC\x00\x00\x00\x22", "audio/flac"),
    ],
)
def test_transform_text_to_speech_response_labels_content_type(audio, expected_content_type):
    raw_response = httpx.Response(
        status_code=200,
        json={"audioContent": base64.b64encode(audio).decode()},
    )

    result = VertexAITextToSpeechConfig().transform_text_to_speech_response(
        model="vertex_ai/chirp",
        raw_response=raw_response,
        logging_obj=MagicMock(),
    )

    assert result.response.headers["content-type"] == expected_content_type
    assert result.response.content == audio


def test_transform_text_to_speech_response_leaves_unknown_bytes_unlabeled():
    raw_pcm = b"\x00\x01\x02\x03\x04\x05\x06\x07"
    raw_response = httpx.Response(
        status_code=200,
        json={"audioContent": base64.b64encode(raw_pcm).decode()},
    )

    result = VertexAITextToSpeechConfig().transform_text_to_speech_response(
        model="vertex_ai/chirp",
        raw_response=raw_response,
        logging_obj=MagicMock(),
    )

    assert "content-type" not in result.response.headers
    assert result.response.content == raw_pcm


@patch(  # test-quality-ok: exercises public speech dispatch up to the outbound HTTP boundary
    "litellm.llms.custom_httpx.llm_http_handler.HTTPHandler.post"
)
@patch.object(  # test-quality-ok: isolates provider credentials in the public API test
    VertexAITextToSpeechConfig, "_ensure_access_token"
)
@patch.object(  # test-quality-ok: fixes the provider URL for deterministic dispatch assertions
    VertexAITextToSpeechConfig, "_get_token_and_url"
)
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


@patch(  # test-quality-ok: exercises public Gemini speech dispatch up to the outbound HTTP boundary
    "litellm.llms.custom_httpx.llm_http_handler.HTTPHandler.post"
)
@patch.object(  # test-quality-ok: isolates provider credentials in the public Gemini API test
    VertexAITextToSpeechConfig, "_ensure_access_token"
)
@patch.object(  # test-quality-ok: fixes the Gemini provider URL for deterministic dispatch assertions
    VertexAITextToSpeechConfig, "_get_token_and_url"
)
def test_litellm_speech_vertex_ai_gemini_tts_mp3_uses_cloud_tts(mock_get_token, mock_ensure_token, mock_post):
    mock_ensure_token.return_value = ("mock-token", "test-project")
    mock_get_token.return_value = ("mock-token", "mock-url")
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = {"audioContent": "SGVsbG8gV29ybGQ="}
    mock_post.return_value = mock_response

    litellm.speech(
        model="vertex_ai/gemini-3.1-flash-tts-preview",
        input="Ryan: Hi.\nKatie: Hello.",
        voice={
            "multi_speaker_voice_config": {
                "speaker_voice_configs": [
                    {
                        "speaker": "Ryan",
                        "voice_config": {
                            "prebuilt_voice_config": {
                                "voice_name": "Umbriel",
                            },
                        },
                    },
                    {
                        "speaker": "Katie",
                        "voice_config": {
                            "prebuilt_voice_config": {
                                "voice_name": "Leda",
                            },
                        },
                    },
                ],
            },
        },
        response_format="mp3",
        vertex_project="test-project",
        vertex_location="global",
    )

    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["url"] == "https://texttospeech.googleapis.com/v1/text:synthesize"
    request_body = call_kwargs["json"]
    assert request_body["audioConfig"]["audioEncoding"] == "MP3"
    assert request_body["voice"] == {
        "languageCode": "en-US",
        "modelName": "gemini-3.1-flash-tts-preview",
        "multiSpeakerVoiceConfig": {
            "speakerVoiceConfigs": [
                {
                    "speakerAlias": "Ryan",
                    "speakerId": "Umbriel",
                },
                {
                    "speakerAlias": "Katie",
                    "speakerId": "Leda",
                },
            ],
        },
    }
