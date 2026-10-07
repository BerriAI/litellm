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

ELEVENLABS_API_BASE: Final = "https://api.elevenlabs.io"
ELEVENLABS_TRANSCRIPTION_URL: Final = f"{ELEVENLABS_API_BASE}/v1/speech-to-text"
ELEVENLABS_API_KEY: Final = "test-elevenlabs-key"


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
    def test_elevenlabs_diarize_parameter_passthrough(
        self,
        respx_mock: respx.MockRouter,
        _elevenlabs_httpx_transport: None,
    ) -> None:
        upstream: Final = respx_mock.post(ELEVENLABS_TRANSCRIPTION_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "text": "Four score and seven years ago",
                    "language_code": "en",
                    "words": [
                        {"type": "word", "text": "Four", "start": 0.0, "end": 0.5},
                        {"type": "word", "text": "score", "start": 0.5, "end": 1.0},
                    ],
                },
            )
        )

        result: Final = litellm.transcription(
            model="elevenlabs/scribe_v1",
            file=b"fake audio data",
            api_key=ELEVENLABS_API_KEY,
            diarize=True,
            language="en",
            temperature=0.5,
            custom_param="test_value",
        )

        assert result.text == "Four score and seven years ago"
        assert upstream.call_count == 1
        body: Final = upstream.calls[0].request.content
        assert b'name="model_id"' in body
        assert b"\r\nscribe_v1\r\n" in body
        assert b'name="diarize"' in body
        assert b"\r\nTrue\r\n" in body
        assert b'name="language_code"' in body
        assert b"\r\nen\r\n" in body
        assert b'name="temperature"' in body
        assert b"\r\n0.5\r\n" in body
        assert b'name="custom_param"' in body
        assert b"\r\ntest_value\r\n" in body
        assert b'name="file"' in body


class TestElevenLabsTextToSpeechTransformation:
    @pytest.fixture(scope="class")
    def config(self) -> ElevenLabsTextToSpeechConfig:
        return ElevenLabsTextToSpeechConfig()

    def test_map_openai_params_maps_voice_and_speed(self, config: ElevenLabsTextToSpeechConfig) -> None:
        kwargs: Final[dict[str, object]] = {}
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
        assert kwargs[config.ELEVENLABS_QUERY_PARAMS_KEY]["output_format"] == "mp3_44100_128"

    def test_transform_request_and_url(self, config: ElevenLabsTextToSpeechConfig) -> None:
        kwargs: Final[dict[str, object]] = {}
        voice_id, optional_params = config.map_openai_params(
            model="eleven_multilingual_v2",
            optional_params={
                "response_format": "pcm",
                "model_id": "eleven_multilingual_v2",
                "pronunciation_dictionary_locators": [{"pronunciation_dictionary_id": "dict_1"}],
            },
            voice="alloy",
            kwargs=kwargs,
        )
        litellm_params: Final = {
            config.ELEVENLABS_VOICE_ID_KEY: voice_id,
            config.ELEVENLABS_QUERY_PARAMS_KEY: kwargs[config.ELEVENLABS_QUERY_PARAMS_KEY],
        }
        headers: Final = config.validate_environment(headers={}, model="eleven_multilingual_v2", api_key="test-key")
        request_data: Final = config.transform_text_to_speech_request(
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
        url: Final = config.get_complete_url(
            model="eleven_multilingual_v2",
            api_base=None,
            litellm_params=litellm_params,
        )
        assert voice_id in url
        assert "output_format=pcm_44100" in url
