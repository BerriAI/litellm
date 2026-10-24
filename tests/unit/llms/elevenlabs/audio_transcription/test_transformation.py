import io
import json
import re
import wave
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


def _transform_raw(content: bytes) -> TranscriptionResponse:
    return ElevenLabsAudioTranscriptionConfig().transform_audio_transcription_response(
        raw_response=httpx.Response(200, content=content)
    )


def _word_with(**fields: object) -> dict[str, object]:
    return {"type": "word", "text": "hi", "start": 0.0, "end": 0.5, **fields}


NUMBER_CASES: Final = (
    ("-0.35", -0.35),
    ("0", 0.0),
    ("0.0", 0.0),
    ("-2", -2.0),
    ("-1e-300", -1e-300),
    ("0.3", 0.3),
    ("1", 1.0),
)
NOT_FINITE_NUMBER_LITERALS: Final = (
    "NaN",
    "Infinity",
    "-Infinity",
    "true",
    "false",
    '"-0.5"',
    "null",
    "1" + "0" * 400,
    "-1" + "0" * 400,
    "[1]",
)


def test_transform_audio_transcription_response_keeps_spoken_words_with_native_metadata():
    payload = {
        "language_code": "en",
        "text": "Hello (laughter) world",
        "words": [
            {"type": "word", "text": "Hello", "start": 0.0, "end": 0.4, "speaker_id": "speaker_0", "logprob": -0.35},
            {"type": "spacing", "text": " ", "start": 0.4, "end": 0.5, "logprob": -0.1},
            {"type": "audio_event", "text": "(laughter)", "start": 0.5, "end": 0.9, "speaker_id": "speaker_1"},
            {"type": "word", "text": "world", "start": 0.9, "end": 1.3},
        ],
    }

    response = _transform(payload)

    assert response.text == "Hello (laughter) world"
    assert response["task"] == "transcribe"
    assert response["language"] == "en"
    assert response["words"] == [
        {"word": "Hello", "start": 0.0, "end": 0.4, "logprob": -0.35, "speaker": "speaker_0"},
        {"word": "world", "start": 0.9, "end": 1.3},
    ]
    assert response["audio_events"] == [{"text": "(laughter)", "start": 0.5, "end": 0.9, "speaker": "speaker_1"}]
    assert response._hidden_params == payload


@pytest.mark.parametrize(("literal", "expected"), NUMBER_CASES)
def test_transform_audio_transcription_response_passes_finite_logprob_through(literal: str, expected: float):
    body: Final = b'{"words": [{"type": "word", "text": "a", "start": 0, "end": 1, "logprob": %s}]}' % literal.encode()

    logprob: Final = _transform_raw(body)["words"][0]["logprob"]

    assert logprob == expected
    assert isinstance(logprob, float)


@pytest.mark.parametrize("literal", NOT_FINITE_NUMBER_LITERALS)
def test_transform_audio_transcription_response_omits_unusable_logprob(literal: str):
    body: Final = b'{"words": [{"type": "word", "text": "a", "start": 0, "end": 1, "logprob": %s}]}' % literal.encode()

    response: Final = _transform_raw(body)

    assert response["words"] == [{"word": "a", "start": 0, "end": 1}]
    assert "NaN" not in json.dumps(response["words"]) and "Infinity" not in json.dumps(response["words"])


def test_transform_audio_transcription_response_omits_absent_logprob_and_keeps_siblings():
    body: Final = (
        b'{"words": [{"type": "word", "text": "a", "logprob": -1.5}, {"type": "word", "text": "b"},'
        b' {"type": "word", "text": "c", "logprob": NaN}]}'
    )

    response: Final = _transform_raw(body)

    assert response["words"][0]["logprob"] == -1.5
    assert all("logprob" not in w for w in response["words"][1:])


@pytest.mark.parametrize(("literal", "expected"), NUMBER_CASES)
def test_transform_audio_transcription_response_passes_finite_language_probability_through(
    literal: str, expected: float
):
    response: Final = _transform_raw(b'{"language_probability": %s}' % literal.encode())

    assert response["language_probability"] == expected
    assert isinstance(response["language_probability"], float)


@pytest.mark.parametrize("literal", NOT_FINITE_NUMBER_LITERALS)
def test_transform_audio_transcription_response_omits_unusable_language_probability(literal: str):
    response: Final = _transform_raw(b'{"text": "t", "language_probability": %s}' % literal.encode())

    assert "language_probability" not in response


def test_transform_audio_transcription_response_omits_absent_language_probability():
    assert "language_probability" not in _transform({"text": "t"})


@pytest.mark.parametrize(
    ("literal", "expected"),
    [("0", 0.0), ("0.0", 0.0), ("11.0", 11.0), ("17", 17.0), ("1e-300", 1e-300), ("3600.5", 3600.5)],
)
def test_transform_audio_transcription_response_exposes_valid_duration(literal: str, expected: float):
    response: Final = _transform_raw(b'{"text": "t", "audio_duration_secs": %s}' % literal.encode())

    assert response["duration"] == expected
    assert isinstance(response["duration"], float)


@pytest.mark.parametrize(
    "literal",
    ["-0.01", "-3", "-1e-300", "-Infinity", "NaN", "Infinity", "true", "false", '"12"', "null", "1" + "0" * 400, "[1]"],
)
def test_transform_audio_transcription_response_omits_invalid_duration(literal: str):
    body: Final = (
        b'{"text": "t", "audio_duration_secs": %s, "words": [{"type": "word", "end": 2.5}]}' % literal.encode()
    )

    assert "duration" not in _transform_raw(body)


def test_transform_audio_transcription_response_omits_absent_duration():
    assert "duration" not in _transform({"text": "t", "words": [_word_with(end=2.5)]})


@pytest.mark.parametrize("speaker", ["speaker_0", "speaker_12", ""])
def test_transform_audio_transcription_response_exposes_string_speaker_on_words_and_events(speaker: str):
    response: Final = _transform(
        {"words": [_word_with(speaker_id=speaker), {"type": "audio_event", "text": "(x)", "speaker_id": speaker}]}
    )

    assert response["words"][0]["speaker"] == speaker
    assert response["audio_events"][0]["speaker"] == speaker


@pytest.mark.parametrize("speaker", [None, 3, 0, True, ["a"], {"id": "a"}, 1.5])
def test_transform_audio_transcription_response_omits_non_string_speaker(speaker: object):
    response: Final = _transform(
        {"words": [_word_with(speaker_id=speaker), {"type": "audio_event", "text": "(x)", "speaker_id": speaker}]}
    )

    assert "speaker" not in response["words"][0]
    assert "speaker" not in response["audio_events"][0]


def test_transform_audio_transcription_response_omits_speaker_when_missing():
    response: Final = _transform({"words": [_word_with(), {"type": "audio_event", "text": "(x)"}]})

    assert "speaker" not in response["words"][0]
    assert "speaker" not in response["audio_events"][0]


def test_transform_audio_transcription_response_keeps_audio_events_in_order_apart_from_words_and_spacing():
    payload = {
        "text": "a [music] b [applause]",
        "words": [
            {"type": "audio_event", "text": "[music]", "start": 0.0, "end": 1.0, "logprob": -0.2},
            {"type": "word", "text": "a", "start": 1.0, "end": 1.2},
            {"type": "spacing", "text": " ", "start": 1.2, "end": 1.3},
            {"type": "word", "text": "b", "start": 1.3, "end": 1.5},
            {"type": "audio_event", "text": "[applause]", "start": 1.5, "end": 2.5, "speaker_id": "speaker_1"},
            {"type": "audio_event", "text": "[cough]", "start": 2.5, "end": 2.6},
        ],
    }

    response: Final = _transform(payload)

    assert response["audio_events"] == [
        {"text": "[music]", "start": 0.0, "end": 1.0},
        {"text": "[applause]", "start": 1.5, "end": 2.5, "speaker": "speaker_1"},
        {"text": "[cough]", "start": 2.5, "end": 2.6},
    ]
    assert [w["word"] for w in response["words"]] == ["a", "b"]
    assert response.text == "a [music] b [applause]"


def test_transform_audio_transcription_response_defaults_missing_audio_event_fields_like_words():
    response: Final = _transform({"words": [{"type": "audio_event"}, {"type": "word"}]})

    assert response["audio_events"] == [{"text": "", "start": 0, "end": 0}]
    assert response["words"] == [{"word": "", "start": 0, "end": 0}]


@pytest.mark.parametrize("words", [[], [{"type": "spacing", "text": " "}], [{"type": "word", "text": "a"}]])
def test_transform_audio_transcription_response_has_empty_audio_events_when_words_present_without_events(
    words: list[dict[str, object]],
):
    assert _transform({"text": "t", "words": words})["audio_events"] == []


@pytest.mark.parametrize("payload", [{}, {"text": "t"}, {"text": "t", "audio_duration_secs": 3.0}])
def test_transform_audio_transcription_response_has_no_audio_events_without_words(payload: dict[str, object]):
    assert "audio_events" not in _transform(payload)


def test_transform_audio_transcription_response_output_is_strict_json_for_hostile_numbers():
    body: Final = (
        b'{"language_probability": NaN, "audio_duration_secs": Infinity, "words": ['
        b'{"type": "word", "text": "a", "logprob": -Infinity, "speaker_id": "s"},'
        b'{"type": "audio_event", "text": "(x)", "start": 0, "end": 1, "logprob": NaN}]}'
    )

    response: Final = _transform_raw(body)

    strict: Final = json.dumps({k: response[k] for k in ("words", "audio_events", "language", "task")}, allow_nan=False)
    assert "language_probability" not in response and "duration" not in response
    assert "null" not in strict


def test_transform_audio_transcription_response_matches_real_response_shape():
    payload = {
        "language_code": "eng",
        "language_probability": 0.9746739864349365,
        "text": "Four score and seven years ago (applause)",
        "words": [
            {"text": "Four", "start": 0.44, "end": 0.7, "type": "word", "speaker_id": "speaker_0", "logprob": -0.00017},
            {"text": " ", "start": 0.7, "end": 0.82, "type": "spacing", "speaker_id": "speaker_0", "logprob": -0.35},
            {"text": "score", "start": 0.82, "end": 1.12, "type": "word", "speaker_id": "speaker_0", "logprob": -0.62},
            {
                "text": "(applause)",
                "start": 1.2,
                "end": 2.0,
                "type": "audio_event",
                "speaker_id": "speaker_0",
                "logprob": -0.9,
            },
        ],
        "transcription_id": "abc",
        "audio_duration_secs": 11.0,
    }

    response: Final = _transform(payload)

    assert response["language"] == "eng"
    assert response["language_probability"] == 0.9746739864349365
    assert response["duration"] == 11.0
    assert response["words"] == [
        {"word": "Four", "start": 0.44, "end": 0.7, "logprob": -0.00017, "speaker": "speaker_0"},
        {"word": "score", "start": 0.82, "end": 1.12, "logprob": -0.62, "speaker": "speaker_0"},
    ]
    assert response["audio_events"] == [{"text": "(applause)", "start": 1.2, "end": 2.0, "speaker": "speaker_0"}]
    assert response._hidden_params == payload


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ({"type": "word"}, [{"word": "", "start": 0.0, "end": 0.0}]),
        ({"type": "word", "text": None, "start": None, "end": None}, [{"word": None, "start": 0.0, "end": 0.0}]),
        ({"type": "word", "text": 7, "start": "0.1", "end": [2]}, [{"word": 7, "start": 0.0, "end": 0.0}]),
        ({"text": "untyped"}, []),
        ({"type": None, "text": "untyped"}, []),
        ({}, []),
    ],
)
def test_transform_audio_transcription_response_maps_one_word(
    word: dict[str, object], expected: list[dict[str, object]]
):
    words: Final = _transform({"text": "t", "words": [word]})["words"]

    assert words == expected
    assert all(isinstance(w["start"], float) and isinstance(w["end"], float) for w in words)


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


def _silent_wav(seconds: int) -> bytes:
    buffer: Final = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"\x00\x00" * 8000 * seconds)
    return buffer.getvalue()


@pytest.mark.parametrize("audio", [b"not a decodable audio file", _silent_wav(1)])
def test_transcription_cost_uses_the_duration_in_the_elevenlabs_response(
    audio: bytes, respx_mock: respx.MockRouter
) -> None:
    api_base: Final = "http://localhost:12346"
    billed_seconds: Final = 7.5
    respx_mock.post(f"{api_base}/v1/speech-to-text").mock(
        return_value=httpx.Response(
            200,
            json={"language_code": "eng", "text": "hi", "words": [], "audio_duration_secs": billed_seconds},
        )
    )
    rate: Final = litellm.get_model_info("elevenlabs/scribe_v2")["input_cost_per_second"]
    assert rate

    response: Final = litellm.transcription(
        model="elevenlabs/scribe_v2", file=("a.wav", audio), api_base=api_base, api_key=ELEVENLABS_API_KEY
    )

    assert response["duration"] == billed_seconds
    assert response._hidden_params["response_cost"] == pytest.approx(billed_seconds * rate)


RESPONSE_FORMATS: Final = ("json", "text", "verbose_json", "srt", "vtt")
SUBTITLE_FORMATS: Final = ("srt", "vtt")
NATIVE_FORMATS: Final = ("json", "text", "verbose_json")
SPOKEN_BODY: Final = {
    "language_code": "eng",
    "language_probability": 0.97,
    "text": "Four score (applause) again",
    "audio_duration_secs": 11.0,
    "words": [
        {"text": "Four", "start": 0.44, "end": 0.7, "type": "word", "speaker_id": "speaker_0", "logprob": -0.1},
        {"text": " ", "start": 0.7, "end": 0.82, "type": "spacing"},
        {"text": "score", "start": 0.82, "end": 1.12, "type": "word", "speaker_id": "speaker_0", "logprob": -0.2},
        {"text": "(applause)", "start": 1.2, "end": 2.0, "type": "audio_event"},
        {"text": "again", "start": 6.0, "end": 6.5, "type": "word", "speaker_id": "speaker_0", "logprob": -0.3},
    ],
}


def _form_values(request: httpx.Request) -> dict[str, bytes]:
    boundary: Final = request.headers["content-type"].split("boundary=")[1].encode()
    parts: Final = request.content.split(b"--" + boundary)
    return {
        match.group(1).decode(): match.group(2)
        for part in parts
        if (match := re.search(rb'name="([^"]+)"\r\n\r\n(.*)\r\n$', part, re.S)) and b"filename=" not in part
    }


def _transcribe(
    respx_mock: respx.MockRouter, body: bytes, **params: object
) -> tuple[TranscriptionResponse, dict[str, bytes]]:
    api_base: Final = "http://localhost:12346"
    route: Final = respx_mock.post(f"{api_base}/v1/speech-to-text").mock(
        return_value=httpx.Response(200, content=body, headers={"content-type": "application/json"})
    )
    response: Final = litellm.transcription(
        model="elevenlabs/scribe_v2",
        file=("a.wav", _silent_wav(1)),
        api_base=api_base,
        api_key=ELEVENLABS_API_KEY,
        **params,
    )
    return response, _form_values(route.calls.last.request)


def _transcribe_spoken_body(
    respx_mock: respx.MockRouter, **params: object
) -> tuple[TranscriptionResponse, dict[str, bytes]]:
    return _transcribe(respx_mock, json.dumps(SPOKEN_BODY).encode(), **params)


@pytest.mark.parametrize("response_format", RESPONSE_FORMATS)
def test_elevenlabs_transcription_accepts_response_format_without_drop_params(
    response_format: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "drop_params", False)

    optional_params: Final = litellm.utils.get_optional_params_transcription(
        model="scribe_v2", custom_llm_provider="elevenlabs", response_format=response_format
    )

    assert optional_params["response_format"] == response_format


@pytest.mark.parametrize("response_format", RESPONSE_FORMATS)
def test_transform_audio_transcription_request_form_is_unchanged_by_response_format(response_format: str):
    config: Final = ElevenLabsAudioTranscriptionConfig()
    audio: Final = ("a.wav", b"fake audio data")
    base_params: Final = {"language_code": "es", "temperature": 0.5, "diarize": True, "tag_audio_events": False}

    without: Final = config.transform_audio_transcription_request(
        model="scribe_v2", audio_file=audio, optional_params=dict(base_params), litellm_params={}
    )
    with_format: Final = config.transform_audio_transcription_request(
        model="scribe_v2",
        audio_file=audio,
        optional_params={**base_params, "response_format": response_format},
        litellm_params={},
    )

    assert with_format.data == without.data
    assert without.data["diarize"] == "True" and without.data["temperature"] == "0.5"
    assert response_format not in with_format.data.values()
    assert "response_format" not in with_format.data


@pytest.mark.parametrize("response_format", SUBTITLE_FORMATS)
def test_elevenlabs_transcription_returns_subtitle_document_built_from_the_spoken_words(
    response_format: str, respx_mock: respx.MockRouter
):
    response, form = _transcribe_spoken_body(respx_mock, response_format=response_format, language="es")

    _, form_without_format = _transcribe_spoken_body(respx_mock, language="es")
    assert form == form_without_format
    assert form["language_code"] == b"es"
    separator: Final = "," if response_format == "srt" else "."
    cues: Final = re.findall(r"^(\d{2}:\d{2}:\d{2}[,.]\d{3}) --> (\d{2}:\d{2}:\d{2}[,.]\d{3})$", response.text, re.M)
    assert [start for start, _ in cues] == [f"00:00:00{separator}440", f"00:00:06{separator}000"]
    assert [end for _, end in cues] == [f"00:00:01{separator}120", f"00:00:06{separator}500"]
    assert "Four score" in response.text and "again" in response.text
    assert (response_format == "vtt") == response.text.startswith("WEBVTT")
    assert response.text != SPOKEN_BODY["text"]


@pytest.mark.parametrize("response_format", NATIVE_FORMATS)
def test_elevenlabs_transcription_keeps_native_text_and_metadata_for_non_subtitle_formats(
    response_format: str, respx_mock: respx.MockRouter
):
    response, form = _transcribe_spoken_body(respx_mock, response_format=response_format, temperature=0.5)

    _, form_without_format = _transcribe_spoken_body(respx_mock, temperature=0.5)
    assert form == form_without_format
    assert form["temperature"] == b"0.5"
    assert response.text == SPOKEN_BODY["text"]
    assert [w["word"] for w in response["words"]] == ["Four", "score", "again"]
    assert response["words"][0]["speaker"] == "speaker_0"
    assert response["audio_events"] == [{"text": "(applause)", "start": 1.2, "end": 2.0}]
    assert response["duration"] == SPOKEN_BODY["audio_duration_secs"]


PASSING_TIMESTAMPS: Final = (("0", 0.0), ("2", 2.0), ("0.44", 0.44), ("6.5", 6.5), ("1e300", 1e300))
UNUSABLE_TIMESTAMPS: Final = ("NaN", "Infinity", "-Infinity", "1e308", "-0.5", "true", "false", '"0.1"', "[2]", "null")


def _item_body(item_type: str, start: str, end: str) -> bytes:
    return b'{"words": [{"type": "%s", "text": "x", "start": %s, "end": %s}]}' % (
        item_type.encode(),
        start.encode(),
        end.encode(),
    )


def _timed_item(response: TranscriptionResponse, item_type: str) -> dict[str, object]:
    return response["words" if item_type == "word" else "audio_events"][0]


@pytest.mark.parametrize("item_type", ["word", "audio_event"])
@pytest.mark.parametrize(("literal", "expected"), PASSING_TIMESTAMPS)
def test_transform_audio_transcription_response_passes_usable_start_and_end_through_as_float(
    item_type: str, literal: str, expected: float
):
    item: Final = _timed_item(_transform_raw(_item_body(item_type, literal, literal)), item_type)

    assert item["start"] == expected and item["end"] == expected
    assert isinstance(item["start"], float) and isinstance(item["end"], float)


@pytest.mark.parametrize("item_type", ["word", "audio_event"])
@pytest.mark.parametrize("literal", UNUSABLE_TIMESTAMPS)
def test_transform_audio_transcription_response_zeroes_unusable_start_and_end(item_type: str, literal: str):
    item: Final = _timed_item(_transform_raw(_item_body(item_type, literal, literal)), item_type)

    assert item["start"] == 0.0 and item["end"] == 0.0
    assert isinstance(item["start"], float) and isinstance(item["end"], float)


@pytest.mark.parametrize("item_type", ["word", "audio_event"])
def test_transform_audio_transcription_response_zeroes_absent_start_and_end_and_keeps_the_other(item_type: str):
    body: Final = b'{"words": [{"type": "%s", "text": "x", "end": 3}]}' % item_type.encode()

    item: Final = _timed_item(_transform_raw(body), item_type)

    assert item["start"] == 0.0 and isinstance(item["start"], float)
    assert item["end"] == 3.0 and isinstance(item["end"], float)


@pytest.mark.parametrize("item_type", ["word", "audio_event"])
@pytest.mark.parametrize("literal", UNUSABLE_TIMESTAMPS)
def test_transform_audio_transcription_response_ends_at_start_when_end_is_unusable(item_type: str, literal: str):
    item: Final = _timed_item(_transform_raw(_item_body(item_type, "1.0", literal)), item_type)

    assert item["start"] == 1.0 and item["end"] == 1.0
    assert isinstance(item["start"], float) and isinstance(item["end"], float)


@pytest.mark.parametrize("item_type", ["word", "audio_event"])
@pytest.mark.parametrize(
    ("start", "end", "expected_end"),
    [("1.0", "0.5", 1.0), ("2", "0", 2.0), ("1.0", "1.0", 1.0), ("0.44", "0.68", 0.68), ("1.0", "6.5", 6.5)],
)
def test_transform_audio_transcription_response_never_ends_before_it_starts(
    item_type: str, start: str, end: str, expected_end: float
):
    item: Final = _timed_item(_transform_raw(_item_body(item_type, start, end)), item_type)

    assert item["start"] == float(start)
    assert item["end"] == expected_end
    assert isinstance(item["end"], float)


@pytest.mark.parametrize("item_type", ["word", "audio_event"])
def test_transform_audio_transcription_response_ends_at_start_when_end_is_absent(item_type: str):
    body: Final = b'{"words": [{"type": "%s", "text": "x", "start": 1.5}]}' % item_type.encode()

    item: Final = _timed_item(_transform_raw(body), item_type)

    assert item["start"] == 1.5 and item["end"] == 1.5


@pytest.mark.parametrize("item_type", ["word", "audio_event"])
def test_transform_audio_transcription_response_keeps_a_usable_end_when_start_is_unusable(item_type: str):
    item: Final = _timed_item(_transform_raw(_item_body(item_type, "Infinity", "2.0")), item_type)

    assert item["start"] == 0.0 and item["end"] == 2.0


def _cue_seconds(clock: str) -> float:
    hours, minutes, seconds, millis = (int(part) for part in re.split(r"[:,.]", clock))
    return hours * 3600 + minutes * 60 + seconds + millis / 1000


def _cue_spans(document: str) -> tuple[tuple[float, float], ...]:
    cues: Final = re.findall(r"(\d{2}:\d{2}:\d{2}[,.]\d{3}) --> (\d{2}:\d{2}:\d{2}[,.]\d{3})", document)
    return tuple((_cue_seconds(start), _cue_seconds(end)) for start, end in cues)


@pytest.mark.parametrize("response_format", SUBTITLE_FORMATS)
@pytest.mark.parametrize("literal", ["Infinity", "NaN", "1e308", "-Infinity"])
def test_elevenlabs_transcription_subtitles_survive_unusable_word_timestamps(
    response_format: str, literal: str, respx_mock: respx.MockRouter
):
    word: Final = '{"type": "word", "text": "%s", "start": %s, "end": %s}'
    body: Final = (
        '{"text": "Four score", "words": [%s, %s]}'
        % (word % ("Four", literal, literal), word % ("score", "1.0", literal))
    ).encode()

    response, _ = _transcribe(respx_mock, body, response_format=response_format)

    separator: Final = "," if response_format == "srt" else "."
    assert response.text != "Four score"
    assert f"00:00:00{separator}000 --> " in response.text
    assert "Four" in response.text and "score" in response.text
    assert (response_format == "vtt") == response.text.startswith("WEBVTT")
    spans: Final = _cue_spans(response.text)
    assert spans
    assert all(start <= end for start, end in spans)


@pytest.mark.parametrize("response_format", SUBTITLE_FORMATS)
def test_elevenlabs_transcription_subtitles_never_end_before_they_start_when_the_end_is_earlier(
    response_format: str, respx_mock: respx.MockRouter
):
    body: Final = b'{"text": "score", "words": [{"type": "word", "text": "score", "start": 1.0, "end": 0.5}]}'

    response, _ = _transcribe(respx_mock, body, response_format=response_format)

    separator: Final = "," if response_format == "srt" else "."
    spans: Final = _cue_spans(response.text)
    assert f"00:00:01{separator}000 --> 00:00:01{separator}000" in response.text
    assert spans
    assert all(start <= end for start, end in spans)


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
