import io
from collections.abc import Iterator
from typing import Final

import httpx
import numpy as np
import pytest
import respx
import soundfile

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.llms.elevenlabs.audio_transcription.transformation import ElevenLabsAudioTranscriptionConfig
from litellm.llms.elevenlabs.text_to_speech.transformation import ElevenLabsTextToSpeechConfig
from litellm.types.utils import TranscriptionResponse, TranscriptionUsageDurationObject
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
            {"type": "word", "text": "Hello", "start": 0.0, "end": 0.4},
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


def _wav_bytes(channels: int) -> bytes:
    buffer: Final = io.BytesIO()
    samples: Final = np.zeros((1600, channels) if channels > 1 else 1600, dtype="float32")
    soundfile.write(buffer, samples, 16000, format="WAV")
    return buffer.getvalue()


def _map_params(non_default_params: dict[str, object]) -> dict:
    return ElevenLabsAudioTranscriptionConfig().map_openai_params(
        non_default_params=non_default_params,
        optional_params={},
        model="scribe_v2",
        drop_params=False,
    )


def _form_data(audio: bytes, optional_params: dict[str, object]) -> dict | bytes:
    return (
        ElevenLabsAudioTranscriptionConfig()
        .transform_audio_transcription_request(
            model="scribe_v2",
            audio_file=audio,
            optional_params=optional_params,
            litellm_params={},
        )
        .data
    )


def test_response_format_is_a_supported_openai_param():
    assert "response_format" in ElevenLabsAudioTranscriptionConfig().get_supported_openai_params("scribe_v2")


def test_diarized_json_response_format_maps_to_diarize():
    assert _map_params({"response_format": "diarized_json"}) == {"diarize": True}


def test_other_response_formats_do_not_enable_diarize():
    assert _map_params({"response_format": "json"}) == {}


def test_language_maps_to_language_code():
    assert _map_params({"language": "pt"}) == {"language_code": "pt"}


@pytest.mark.parametrize(
    ("channels", "optional_params", "expected"),
    [
        pytest.param(1, {"diarize": True}, {"diarize": "True"}, id="mono-uses-acoustic-diarize"),
        pytest.param(
            2,
            {"diarize": True},
            {"use_multi_channel": "True", "multichannel_output_style": "combined"},
            id="stereo-switches-to-multichannel",
        ),
        pytest.param(
            1,
            {"diarize": True, "use_multi_channel": True},
            {"use_multi_channel": "True", "multichannel_output_style": "combined"},
            id="explicit-multichannel-wins-on-mono",
        ),
        pytest.param(
            2,
            {"diarize": True, "use_multi_channel": False},
            {"diarize": "True"},
            id="explicit-no-multichannel-keeps-diarize-on-stereo",
        ),
        pytest.param(2, {}, {}, id="no-diarization-requested"),
    ],
)
def test_diarization_mechanism_follows_audio_channel_count(
    channels: int, optional_params: dict[str, object], expected: dict[str, str]
):
    assert _form_data(_wav_bytes(channels), optional_params) == {"model_id": "scribe_v2", **expected}


def test_unreadable_audio_falls_back_to_acoustic_diarize():
    assert _form_data(b"not audio at all", {"diarize": True}) == {"model_id": "scribe_v2", "diarize": "True"}


def test_diarized_response_builds_speaker_segments():
    response: Final = _transform(
        {
            "language_code": "por",
            "text": "Bom dia. Oi?",
            "audio_duration_secs": 70.5,
            "words": [
                {"text": "ev", "start": 1.0, "end": 2.0, "type": "audio_event", "speaker_id": "speaker_0"},
                {"text": "Bom", "start": 3.0, "end": 3.4, "type": "word", "speaker_id": "speaker_1"},
                {"text": "dia", "start": 3.4, "end": 3.8, "type": "word", "speaker_id": "speaker_1"},
                {"text": "Oi", "start": 5.0, "end": 5.3, "type": "word", "speaker_id": "speaker_0"},
            ],
        }
    )

    assert response.text == "Bom dia. Oi?"
    assert response["language"] == "por"
    assert response["segments"] == (
        {
            "id": "segment_0",
            "start": 3.0,
            "end": 3.8,
            "speaker": "speaker_1",
            "text": "Bom dia",
            "type": "transcript.text.segment",
        },
        {
            "id": "segment_1",
            "start": 5.0,
            "end": 5.3,
            "speaker": "speaker_0",
            "text": "Oi",
            "type": "transcript.text.segment",
        },
    )
    assert "words" not in response


def test_diarized_response_reports_fractional_duration_as_typed_usage():
    response: Final = _transform(
        {
            "text": "oi",
            "audio_duration_secs": 70.5,
            "words": [{"text": "oi", "start": 1.0, "end": 1.5, "type": "word", "speaker_id": "speaker_0"}],
        }
    )

    assert response["usage"] == TranscriptionUsageDurationObject(type="duration", seconds=70.5)
    assert response["duration"] == 70.5
    assert response.model_dump()["usage"] == {"type": "duration", "seconds": 70.5}


def test_multichannel_response_uses_the_channel_as_speaker():
    response: Final = _transform(
        {
            "audio_duration_secs": 40.0,
            "transcripts": [
                {
                    "channel_index": 1,
                    "language_code": "por",
                    "text": "lado B",
                    "words": [
                        {"text": "lado", "start": 10.0, "end": 10.5, "type": "word", "speaker_id": "speaker_0"},
                        {"text": "B", "start": 10.5, "end": 10.8, "type": "word", "speaker_id": "speaker_0"},
                    ],
                },
                {
                    "channel_index": 0,
                    "language_code": "por",
                    "text": "lado A",
                    "words": [
                        {"text": "lado", "start": 1.0, "end": 1.5, "type": "word", "speaker_id": "speaker_0"},
                        {"text": "A", "start": 1.5, "end": 1.8, "type": "word", "speaker_id": "speaker_0"},
                    ],
                },
            ],
        }
    )

    assert [(s["speaker"], s["text"], s["start"]) for s in response["segments"]] == [
        ("speaker_0", "lado A", 1.0),
        ("speaker_1", "lado B", 10.0),
    ]
    assert response.text == "lado B lado A"
    assert response["language"] == "por"
    assert response["duration"] == 40.0


def test_multichannel_response_without_channel_index_keeps_speakers_distinct():
    response: Final = _transform(
        {
            "transcripts": [
                {"text": "a", "words": [{"text": "a", "start": 1.0, "end": 1.2, "type": "word"}]},
                {"text": "b", "words": [{"text": "b", "start": 2.0, "end": 2.2, "type": "word"}]},
            ],
        }
    )

    assert [s["speaker"] for s in response["segments"]] == ["speaker_0", "speaker_1"]
    assert response["usage"] is None


@pytest.mark.parametrize(
    "payload",
    [
        {"text": "t", "words": [{"type": "word", "text": None, "speaker_id": "speaker_0"}]},
        {"transcripts": "not a list"},
        {"transcripts": [{"words": [{"type": "unknown", "text": "x"}]}]},
    ],
)
def test_malformed_diarized_payloads_raise_with_the_raw_body(payload: object):
    raw_response: Final = httpx.Response(200, json=payload)

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
