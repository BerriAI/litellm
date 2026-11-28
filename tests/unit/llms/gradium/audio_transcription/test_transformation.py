import io
import json
import wave
from collections.abc import Iterator, Mapping, Sequence
from typing import Final
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
import respx
from pydantic import TypeAdapter

import litellm
from litellm.llms.gradium.common_utils import GradiumException
from litellm.types.utils import FileTypes, TranscriptionResponse

_API_BASE: Final = "https://eu.api.gradium.ai"
_API_KEY: Final = "gd_test_key"
_ASR_URL: Final = f"{_API_BASE}/api/speech/asr"
_JSON_CONFIG: Final = TypeAdapter(dict[str, object])


def _wav(seconds: int) -> bytes:
    buffer: Final = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24000)
        writer.writeframes(b"\0\0" * 24000 * seconds)
    return buffer.getvalue()


_WAV: Final = _wav(3)
_HELLO_WORLD: Final = (
    {"type": "step", "inactivity_prob": 0.1},
    {"type": "text", "text": "Hello", "start_s": 0.5, "stream_id": 0},
    {"type": "end_text", "stop_s": 0.9, "stream_id": 0},
    {"type": "text", "text": " world", "start_s": 1.0, "stream_id": 0},
    {"type": "end_text", "stop_s": 1.4, "stream_id": 0},
    {"type": "end_of_stream"},
)


def _ndjson(messages: Sequence[Mapping[str, object]]) -> str:
    return "\n".join(json.dumps(message) for message in messages)


@pytest.fixture
def gradium_asr(monkeypatch: pytest.MonkeyPatch, local_model_cost_map: None) -> Iterator[respx.Route]:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    with respx.mock:
        yield respx.post(url__startswith=_ASR_URL).mock(return_value=httpx.Response(200, text=_ndjson(_HELLO_WORLD)))


def _transcribe(
    file: FileTypes = ("speech.wav", _WAV), model: str = "gradium/default", **kwargs: object
) -> TranscriptionResponse:
    response: Final = litellm.transcription(model=model, file=file, api_base=_API_BASE, api_key=_API_KEY, **kwargs)
    assert isinstance(response, TranscriptionResponse)
    return response


def _sent_query(route: respx.Route) -> Mapping[str, str]:
    query: Final = parse_qs(urlsplit(str(route.calls[-1].request.url)).query)
    return {key: values[0] for key, values in query.items()}


def _sent_json_config(route: respx.Route) -> dict[str, object]:
    return _JSON_CONFIG.validate_json(_sent_query(route)["json_config"])


def _per_second_rate(model: str) -> float:
    rate: Final = litellm.get_model_info(model=model, custom_llm_provider="gradium").get("input_cost_per_second")
    assert rate is not None
    return rate


def test_sends_the_raw_audio_and_returns_the_joined_transcript_with_timed_segments(gradium_asr: respx.Route):
    """Gradium takes the audio bytes as the body and answers ndjson; callers get one transcript plus timings."""
    response: Final = _transcribe()

    request: Final = gradium_asr.calls[-1].request
    assert request.content == _WAV
    assert request.headers["content-type"] == "audio/wav"
    assert request.headers["x-api-key"] == _API_KEY
    assert _sent_query(gradium_asr)["model"] == "default"
    assert response.text == "Hello world"
    assert response.get("segments") == (
        {"id": 0, "start": 0.5, "end": 0.9, "text": "Hello"},
        {"id": 1, "start": 1.0, "end": 1.4, "text": "world"},
    )


def test_matches_each_segment_end_to_its_own_stream(gradium_asr: respx.Route):
    """An end_text closes the latest text of the same stream_id, so interleaved streams keep their own timings."""
    gradium_asr.mock(
        return_value=httpx.Response(
            200,
            text=_ndjson(
                (
                    {"type": "text", "text": "left", "start_s": 0.0, "stream_id": 0},
                    {"type": "text", "text": "right", "start_s": 0.2, "stream_id": 1},
                    {"type": "end_text", "stop_s": 0.8, "stream_id": 1},
                    {"type": "end_text", "stop_s": 0.6, "stream_id": 0},
                    {"type": "end_of_stream"},
                )
            ),
        )
    )

    response: Final = _transcribe()

    assert response.get("segments") == (
        {"id": 0, "start": 0.0, "end": 0.6, "text": "left"},
        {"id": 1, "start": 0.2, "end": 0.8, "text": "right"},
    )


def test_asks_gradium_to_detect_the_language_when_the_caller_sets_none(gradium_asr: respx.Route):
    """Gradium rejects requests without a language, so an OpenAI-style call without one must still succeed."""
    _transcribe()

    assert _sent_json_config(gradium_asr) == {"language": "any"}


def test_maps_openai_language_and_temperature_into_json_config(gradium_asr: respx.Route):
    """OpenAI's language and temperature reach Gradium under the json_config keys it reads."""
    _transcribe(language="fr", temperature=0.2)

    assert _sent_json_config(gradium_asr) == {"language": "fr", "temp": 0.2}


@pytest.mark.parametrize(
    "json_config",
    [{"delay_in_frames": 16, "language": "de"}, '{"delay_in_frames": 16, "language": "de"}'],
    ids=["dict", "form-encoded string"],
)
def test_forwards_the_callers_json_config_with_its_language(gradium_asr: respx.Route, json_config: object):
    """Gradium-only settings ride in json_config, as a dict from the SDK or a JSON string from a proxy form."""
    _transcribe(json_config=json_config)

    assert _sent_json_config(gradium_asr) == {"delay_in_frames": 16, "language": "de"}


def test_prefers_the_openai_language_over_the_one_in_json_config(gradium_asr: respx.Route):
    """The explicit OpenAI param is the more specific request, so it wins a conflict."""
    _transcribe(language="en", json_config={"language": "de"})

    assert _sent_json_config(gradium_asr)["language"] == "en"


def test_rejects_a_json_config_that_is_not_an_object(gradium_asr: respx.Route):
    """A malformed json_config fails as a client error instead of reaching Gradium."""
    with pytest.raises(litellm.BadRequestError, match="json_config must be a JSON object"):
        _transcribe(json_config="[1, 2]")

    assert not gradium_asr.called


@pytest.mark.parametrize(
    ("filename", "audio", "content_type"),
    [
        ("speech.bin", _WAV, "audio/wav"),
        ("speech.wav", b"OggS" + b"\0" * 32, "audio/ogg"),
        ("speech.opus", b"\1" * 32, "audio/ogg"),
        ("speech.pcm", b"\1" * 32, "audio/pcm"),
    ],
)
def test_labels_the_audio_with_the_content_type_gradium_expects(
    gradium_asr: respx.Route, filename: str, audio: bytes, content_type: str
):
    """The container is read from the bytes first and the filename second, since uploads are often misnamed."""
    _transcribe(file=(filename, audio))

    assert gradium_asr.calls[-1].request.headers["content-type"] == content_type


@pytest.mark.parametrize(
    ("filename", "audio"),
    [("speech.mp3", b"ID3" + b"\0" * 32), ("speech.flac", b"fLaC" + b"\0" * 32), ("speech", b"\1" * 32)],
)
def test_rejects_audio_gradium_cannot_decode_before_calling_it(gradium_asr: respx.Route, filename: str, audio: bytes):
    """Gradium answers an opaque 500 for unknown containers, so the caller gets a 400 that names the fix."""
    with pytest.raises(litellm.BadRequestError, match="input_format"):
        _transcribe(file=(filename, audio))

    assert not gradium_asr.called


def test_bills_raw_pcm_by_its_length_at_the_declared_sample_rate(gradium_asr: respx.Route):
    """Headerless audio has no duration to read, so it is derived from the byte count and input_format."""
    two_seconds_at_16khz: Final = b"\0\0" * 16000 * 2

    response: Final = _transcribe(file=("speech.raw", two_seconds_at_16khz), input_format="pcm_16000")

    assert _sent_query(gradium_asr)["input_format"] == "pcm_16000"
    assert gradium_asr.calls[-1].request.headers["content-type"] == "audio/pcm"
    assert response.usage is not None and response.usage.seconds == pytest.approx(2.0)
    assert response._hidden_params["response_cost"] == pytest.approx(2.0 * _per_second_rate("default"))


def test_bills_wav_audio_per_second_of_the_file(gradium_asr: respx.Route):
    """gradium/default is shared with TTS, so its per-second STT rate must still apply to transcriptions."""
    response: Final = _transcribe()

    assert response._hidden_params["response_cost"] == pytest.approx(3.0 * _per_second_rate("default"))


def test_rejects_an_input_format_gradium_does_not_know(gradium_asr: respx.Route):
    """Gradium answers an opaque 500 for unknown input_format values, so the caller gets a 400 instead."""
    with pytest.raises(litellm.BadRequestError, match="input_format must be one of"):
        _transcribe(input_format="mp3")

    assert not gradium_asr.called


def test_raises_the_error_gradium_sends_mid_stream(gradium_asr: respx.Route):
    """Gradium reports failures after the 200 status as an ndjson error line, which must not pass as a transcript."""
    gradium_asr.mock(
        return_value=httpx.Response(
            200,
            text=_ndjson(
                (
                    {"type": "text", "text": "partial", "start_s": 0.0},
                    {"type": "error", "message": "model overloaded"},
                )
            ),
        )
    )

    with pytest.raises(GradiumException, match="model overloaded"):
        _transcribe()


def test_keeps_the_status_and_detail_of_errors_gradium_sends_before_the_stream(gradium_asr: respx.Route):
    """A rejected request keeps Gradium's status, so a bad key surfaces as 401 rather than a server error."""
    gradium_asr.mock(return_value=httpx.Response(401, json={"detail": "Invalid or expired API key"}))

    with pytest.raises(GradiumException, match="Invalid or expired API key") as raised:
        _transcribe()

    assert raised.value.status_code == 401


@pytest.mark.parametrize(
    "messages",
    [(), ({"type": "ready"}, {"type": "step", "inactivity_prob": 0.0})],
    ids=["empty body", "cut after ready"],
)
def test_raises_when_gradium_stops_before_end_of_stream(
    gradium_asr: respx.Route, messages: Sequence[Mapping[str, object]]
):
    """Gradium answers an unsupported language or misread audio with a 200 that never finishes; that is no transcript."""
    gradium_asr.mock(return_value=httpx.Response(200, text=_ndjson(messages)))

    with pytest.raises(GradiumException, match="before end_of_stream") as raised:
        _transcribe()

    assert raised.value.status_code == 502


@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\x85"])
@pytest.mark.parametrize("line_end", ["\n", "\r\n"])
def test_splits_the_stream_only_on_ndjson_line_ends(gradium_asr: respx.Route, separator: str, line_end: str):
    """Unicode line separators are valid inside a JSON string, so a transcript holding one must not break parsing."""
    spoken: Final = f"Hello{separator}world"
    gradium_asr.mock(
        return_value=httpx.Response(
            200,
            text=line_end.join(
                json.dumps(message, ensure_ascii=False)
                for message in (
                    {"type": "text", "text": spoken, "start_s": 0.0},
                    {"type": "end_text", "stop_s": 0.8},
                    {"type": "end_of_stream"},
                )
            ),
        )
    )

    response: Final = _transcribe()

    assert response.get("segments") == ({"id": 0, "start": 0.0, "end": 0.8, "text": spoken},)


def test_raises_on_a_body_that_is_not_ndjson(gradium_asr: respx.Route):
    """A 200 that does not carry ndjson messages is an upstream fault, not an empty transcript."""
    gradium_asr.mock(return_value=httpx.Response(200, text="<html>gateway</html>"))

    with pytest.raises(GradiumException, match="malformed transcription stream") as raised:
        _transcribe()

    assert raised.value.status_code == 502


@pytest.mark.parametrize("response_format", ["srt", "vtt"])
def test_rejects_subtitle_formats_unless_drop_params_is_set(gradium_asr: respx.Route, response_format: str):
    """Gradium has no subtitle output, so srt/vtt must fail loudly or be dropped on request."""
    with pytest.raises(litellm.UnsupportedParamsError, match="response_format"):
        _transcribe(response_format=response_format)

    response: Final = _transcribe(response_format=response_format, drop_params=True)

    assert response.text == "Hello world"


def test_requires_an_api_key(gradium_asr: respx.Route):
    """Gradium authenticates every call with x-api-key, so a missing key fails before the request."""
    with pytest.raises(litellm.AuthenticationError, match="Gradium API key is required"):
        litellm.transcription(model="gradium/default", file=("speech.wav", _WAV), api_base=_API_BASE)

    assert not gradium_asr.called


def test_routes_the_translating_model_by_name(gradium_asr: respx.Route):
    """stt-translate is a separate Gradium model, priced per second like the default one."""
    response: Final = _transcribe(model="gradium/stt-translate", json_config={"target_language": "en"})

    assert _sent_query(gradium_asr)["model"] == "stt-translate"
    assert _sent_json_config(gradium_asr) == {"target_language": "en", "language": "any"}
    assert response._hidden_params["response_cost"] == pytest.approx(3.0 * _per_second_rate("stt-translate"))
