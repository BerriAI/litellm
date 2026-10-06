import httpx
import pytest

from litellm.llms.elevenlabs.audio_transcription.transformation import ElevenLabsAudioTranscriptionConfig
from litellm.types.utils import TranscriptionResponse


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
