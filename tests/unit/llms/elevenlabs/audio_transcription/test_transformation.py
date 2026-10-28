from collections.abc import Iterator
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.llms.elevenlabs.audio_transcription.transformation import ElevenLabsAudioTranscriptionConfig
from litellm.llms.elevenlabs.text_to_speech.transformation import ElevenLabsTextToSpeechConfig
from litellm.types.utils import TranscriptionResponse
from typing import Any, Dict
from unittest.mock import patch, MagicMock

ELEVENLABS_API_BASE: Final = "https://api.elevenlabs.io"
ELEVENLABS_TRANSCRIPTION_URL: Final = f"{ELEVENLABS_API_BASE}/v1/speech-to-text"
ELEVENLABS_API_KEY: Final = "test-elevenlabs-key"


@pytest.fixture
def elevenlabs_api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", ELEVENLABS_API_KEY)


@pytest.fixture
def _elevenlabs_httpx_transport(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    client_cache: Final = LLMClientCache()
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", client_cache)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "force_ipv4", False)
    monkeypatch.setattr(litellm, "sync_transport", None, raising=False)
    yield
    client_cache.flush_cache()


def _transform(payload: object) -> TranscriptionResponse:
    return ElevenLabsAudioTranscriptionConfig().transform_audio_transcription_response(
        raw_response=httpx.Response(200, json=payload)
    )


def test_transform_audio_transcription_response_keeps_only_spoken_words():
    payload = {
        "language_code": "en",
        "text": "Hello world",
        "words": [
            {"type": "word", "text": "Hello", "start": 0.0, "end": 0.4, "speaker_id": "speaker_0"},
            {"type": "spacing", "text": " ", "start": 0.4, "end": 0.5},
            {"type": "audio_event", "text": "(laughter)", "start": 0.5, "end": 0.9},
            {"type": "word", "text": "world", "start": 0.9, "end": 1.3},
        ],
    }

    response = _transform(payload)

    assert response.text == "Hello world"
    assert response["task"] == "transcribe"
    assert response["language"] == "en"
    assert response["words"] == [
        {"word": "Hello", "start": 0.0, "end": 0.4},
        {"word": "world", "start": 0.9, "end": 1.3},
    ]
    assert response._hidden_params == payload


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ({"type": "word"}, [{"word": "", "start": 0, "end": 0}]),
        ({"type": "word", "text": None, "start": None, "end": None}, [{"word": None, "start": None, "end": None}]),
        ({"type": "word", "text": 7, "start": "0.1", "end": [2]}, [{"word": 7, "start": "0.1", "end": [2]}]),
        ({"text": "untyped"}, []),
        ({"type": None, "text": "untyped"}, []),
        ({}, []),
    ],
)
def test_transform_audio_transcription_response_maps_one_word(
    word: dict[str, object], expected: list[dict[str, object]]
):
    assert _transform({"text": "t", "words": [word]})["words"] == expected


@pytest.mark.parametrize("words", [[], "", {}])
def test_transform_audio_transcription_response_with_empty_words_has_empty_word_list(words: object):
    assert _transform({"text": "t", "words": words})["words"] == []


@pytest.mark.parametrize(
    ("payload", "expected_text", "expected_language"),
    [
        ({}, "", "unknown"),
        ({"text": None, "language_code": None}, None, None),
        ({"text": "bonjour", "language_code": "fr"}, "bonjour", "fr"),
        ({"text": "hola", "language_code": ["es"]}, "hola", ["es"]),
    ],
)
def test_transform_audio_transcription_response_without_words_key_has_no_word_list(
    payload: dict[str, object], expected_text: str | None, expected_language: object
):
    response = _transform(payload)

    assert response.text == expected_text
    assert response["language"] == expected_language
    assert "words" not in response
    assert response._hidden_params == payload


@pytest.mark.parametrize(
    "payload",
    [
        ["not", "an", "object"],
        "plain text",
        7,
        {"text": ["not", "text"]},
        {"text": "t", "words": None},
        {"text": "t", "words": 7},
        {"text": "t", "words": "not a list"},
        {"text": "t", "words": ["not an object"]},
        {"text": "t", "words": [{"type": "word", "text": "ok"}, None]},
    ],
)
def test_transform_audio_transcription_response_wraps_malformed_payloads_with_the_raw_body(payload: object):
    raw_response = httpx.Response(200, json=payload)

    with pytest.raises(ValueError, match="Error transforming ElevenLabs response: ") as exc_info:
        ElevenLabsAudioTranscriptionConfig().transform_audio_transcription_response(raw_response=raw_response)

    assert str(exc_info.value).endswith(f"\nResponse: {raw_response.text}")


class TestElevenLabsAudioTranscription:
    @pytest.mark.usefixtures("elevenlabs_api_key_env")
    def test_elevenlabs_diarize_parameter_passthrough(self):
        """
        Test that provider-specific parameters like diarize=True get passed through
        to the ElevenLabs request form data.
        """
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = (
            '{"text": "Four score and seven years ago", "language_code": "en"}'
        )
        mock_response.json.return_value = {
            "text": "Four score and seven years ago",
            "language_code": "en",
            "words": [
                {"type": "word", "text": "Four", "start": 0.0, "end": 0.5},
                {"type": "word", "text": "score", "start": 0.5, "end": 1.0},
            ],
        }

        # Create a mock audio file
        audio_content = b"fake audio data"

        captured_request_data = {}

        def mock_post(*args, **kwargs):
            # Capture the request data for verification
            captured_request_data.update(
                {
                    "url": kwargs.get("url"),
                    "data": kwargs.get("data"),
                    "files": kwargs.get("files"),
                    "headers": kwargs.get("headers"),
                    "json": kwargs.get("json"),
                }
            )
            return mock_response

        # Mock the HTTPHandler.post method which is what actually makes the request
        from litellm.llms.custom_httpx.http_handler import HTTPHandler

        with patch.object(HTTPHandler, "post", side_effect=mock_post):
            try:
                result = litellm.transcription(
                    model="elevenlabs/scribe_v1",
                    file=audio_content,
                    diarize=True,  # This should be passed through to the form data
                    language="en",  # This should be mapped to language_code
                    temperature=0.5,  # This should also be passed through
                    custom_param="test_value",  # This should also be passed through
                )

                # Verify the request was made with correct form data
                assert "speech-to-text" in captured_request_data["url"]

                # Check that form data contains the expected parameters
                form_data = captured_request_data["data"]
                assert form_data is not None, "Form data should not be None"

                print(f"✅ Captured form data: {form_data}")

                # Check basic required parameters
                assert "model_id" in form_data, "model_id should be in form data"
                assert (
                    form_data["model_id"] == "scribe_v1"
                ), f"Expected model_id 'scribe_v1', got {form_data['model_id']}"

                # Check that diarize parameter is passed through
                assert (
                    "diarize" in form_data
                ), f"diarize should be in form data. Got: {list(form_data.keys())}"
                assert (
                    form_data["diarize"] == "True"
                ), f"Expected diarize='True', got {form_data['diarize']}"

                # Check that OpenAI language parameter is mapped correctly
                assert (
                    "language_code" in form_data
                ), "language_code should be in form data"
                assert (
                    form_data["language_code"] == "en"
                ), f"Expected language_code='en', got {form_data['language_code']}"

                # Check that temperature is passed through
                assert "temperature" in form_data, "temperature should be in form data"
                assert (
                    form_data["temperature"] == "0.5"
                ), f"Expected temperature='0.5', got {form_data['temperature']}"

                # Check that custom parameters are passed through
                assert (
                    "custom_param" in form_data
                ), "custom_param should be in form data"
                assert (
                    form_data["custom_param"] == "test_value"
                ), f"Expected custom_param='test_value', got {form_data['custom_param']}"

                # Check that files are included
                files = captured_request_data["files"]
                assert files is not None, "Files should not be None"
                assert "file" in files, "file should be in files"

                print("✅ All parameter passthrough tests passed!")

            except Exception as e:
                print(f"❌ Test failed: {e}")
                print(f"Captured request data: {captured_request_data}")
                raise


class TestElevenLabsTextToSpeechTransformation:
    @pytest.fixture(scope="class")
    def config(self) -> ElevenLabsTextToSpeechConfig:
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
