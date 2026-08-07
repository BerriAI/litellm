import json

import httpx

from litellm.llms.base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
)
from litellm.llms.elevenlabs.audio_transcription.transformation import (
    ElevenLabsAudioTranscriptionConfig,
)


def _response(payload: dict) -> httpx.Response:
    return httpx.Response(
        200,
        content=json.dumps(payload).encode(),
        request=httpx.Request("POST", "https://api.elevenlabs.io/v1/speech-to-text"),
    )


def test_request_serializes_bool_form_values_lowercase():
    """Boolean optional params must reach ElevenLabs as lowercase ``true``/``false``
    (httpx-style), not Python's ``str(True)`` -> ``"True"``; otherwise flags such as
    ``use_multi_channel`` / ``diarize`` are silently ignored."""
    config = ElevenLabsAudioTranscriptionConfig()

    result = config.transform_audio_transcription_request(
        model="scribe_v2",
        audio_file=b"\x00\x01\x02\x03",
        optional_params={"use_multi_channel": True, "diarize": False},
        litellm_params={},
    )

    assert isinstance(result, AudioTranscriptionRequestData)
    assert result.data["model_id"] == "scribe_v2"
    assert result.data["use_multi_channel"] == "true"
    assert result.data["diarize"] == "false"


def test_response_preserves_multichannel_transcripts():
    """A ``use_multi_channel`` response carries a per-channel ``transcripts`` array;
    it must survive on the TranscriptionResponse (along with ``audio_duration_secs``)
    rather than being flattened to a single ``text``."""
    config = ElevenLabsAudioTranscriptionConfig()

    payload = {
        "transcripts": [
            {
                "channel_index": 0,
                "words": [{"text": "hello", "start": 0.0, "end": 0.4, "type": "word"}],
            },
            {
                "channel_index": 1,
                "words": [{"text": "hi", "start": 0.2, "end": 0.5, "type": "word"}],
            },
        ],
        "audio_duration_secs": 1.5,
        "language_code": "en",
    }

    response = config.transform_audio_transcription_response(_response(payload))

    assert [t["channel_index"] for t in response["transcripts"]] == [0, 1]
    assert response["transcripts"][0]["words"][0]["text"] == "hello"
    assert response["audio_duration_secs"] == 1.5


def test_response_single_channel_is_unchanged():
    """Single-channel responses keep the flat ``text`` and gain no ``transcripts`` key."""
    config = ElevenLabsAudioTranscriptionConfig()

    response = config.transform_audio_transcription_response(
        _response({"text": "hello world", "language_code": "en"})
    )

    assert response.text == "hello world"
    assert "transcripts" not in response.model_dump()


def test_response_mono_words_synthesize_raw_single_channel():
    """A mono (single-channel) response carries raw ``words`` but no ``transcripts``.
    We synthesize a one-channel ``transcripts`` entry holding the RAW words verbatim
    (``text``/``type``/spacing preserved), so callers get the same per-word tokens as
    multichannel. The lossy OpenAI-format ``words`` (renamed ``word``, punctuation
    dropped) stays too for OpenAI-compat consumers."""
    config = ElevenLabsAudioTranscriptionConfig()

    payload = {
        "text": "hello world",
        "language_code": "en",
        "audio_duration_secs": 1.5,
        "words": [
            {"text": "hello", "start": 0.0, "end": 0.4, "type": "word"},
            {"text": " ", "start": 0.4, "end": 0.5, "type": "spacing"},
            {"text": "world", "start": 0.5, "end": 0.9, "type": "word"},
        ],
    }

    response = config.transform_audio_transcription_response(_response(payload))

    # Synthesized single channel with the RAW words (spacing + type intact).
    assert [t["channel_index"] for t in response["transcripts"]] == [0]
    raw = response["transcripts"][0]["words"]
    assert [w["text"] for w in raw] == ["hello", " ", "world"]
    assert [w["type"] for w in raw] == ["word", "spacing", "word"]
    # OpenAI-format words unchanged: only real words, renamed key, no spacing.
    assert response["words"] == [
        {"word": "hello", "start": 0.0, "end": 0.4},
        {"word": "world", "start": 0.5, "end": 0.9},
    ]
