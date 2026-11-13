import io
import json
from collections.abc import Iterator, Mapping
from email.parser import BytesParser
from email.policy import HTTP
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.llms.zoom_ai.audio_transcription.transformation import (
    ZoomAIAudioTranscriptionConfig,
    zoom_ai_recognition_config,
)
from litellm.llms.zoom_ai.common_utils import ZOOM_AI_TRACKING_ID_PREFIX, ZoomAIException

ZOOM_AI_FAST_URL: Final = "https://api.zoom.us/v2/aiservices/scribe/transcribe"
ZOOM_AI_API_KEY: Final = "test-zoom-ai-key"
AUDIO: Final = ("speech.wav", b"RIFF0000WAVEfmt ", "audio/wav")
SCRIBE_RESPONSE: Final = {
    "request_id": "req-1",
    "duration_sec": 5.775,
    "model": "zoom-scribe-en",
    "result": {"text_display": "Hi there, what is the capital of France?", "segments": []},
    "usage": {"input_units": 5.775, "output_units": 0, "unit_type": "seconds"},
}


@pytest.fixture
def zoom_ai_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("ZOOM_AI_API_KEY", ZOOM_AI_API_KEY)
    monkeypatch.delenv("ZOOM_AI_API_BASE", raising=False)
    client_cache: Final = LLMClientCache()
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", client_cache)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "force_ipv4", False)
    monkeypatch.setattr(litellm, "sync_transport", None, raising=False)
    yield
    client_cache.flush_cache()


def _form_fields(request: httpx.Request) -> Mapping[str, bytes]:
    message: Final = BytesParser(policy=HTTP).parsebytes(
        f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode() + request.content
    )
    return {
        part.get_param("name", header="content-disposition"): part.get_payload(decode=True)
        for part in message.iter_parts()
    }


def _sent_config(request: httpx.Request) -> Mapping[str, object]:
    return json.loads(_form_fields(request)["config"])


def _rate(model: str) -> float:
    return litellm.get_model_info(f"zoom_ai/{model}")["input_cost_per_second"]


def test_transcription_sends_the_audio_and_config_to_scribe_fast(
    zoom_ai_env: None, respx_mock: respx.MockRouter
) -> None:
    route: Final = respx_mock.post(ZOOM_AI_FAST_URL).respond(json=SCRIBE_RESPONSE)

    response: Final = litellm.transcription(model="zoom_ai/zoom-scribe-en", file=AUDIO, language="en", diarization=True)

    request: Final = route.calls.last.request
    assert request.headers["authorization"] == f"Bearer {ZOOM_AI_API_KEY}"
    assert request.headers["x-zm-trackingid"].startswith(ZOOM_AI_TRACKING_ID_PREFIX)
    assert _form_fields(request)["file"] == AUDIO[1]
    assert _sent_config(request) == {"model": "zoom-scribe-en", "language": "en-US", "diarization": True}
    assert response.text == SCRIBE_RESPONSE["result"]["text_display"]
    assert response["duration"] == SCRIBE_RESPONSE["duration_sec"]


@pytest.mark.asyncio
async def test_atranscription_matches_the_sync_request(zoom_ai_env: None, respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(ZOOM_AI_FAST_URL).respond(json=SCRIBE_RESPONSE)

    response: Final = await litellm.atranscription(model="zoom_ai/zoom-scribe-en-pro", file=AUDIO)

    assert _sent_config(route.calls.last.request) == {"model": "zoom-scribe-en-pro", "language": "en-US"}
    assert response.text == SCRIBE_RESPONSE["result"]["text_display"]


def test_each_request_gets_its_own_tracking_id(zoom_ai_env: None, respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(ZOOM_AI_FAST_URL).respond(json=SCRIBE_RESPONSE)

    litellm.transcription(model="zoom_ai/zoom-scribe-en", file=AUDIO)
    litellm.transcription(model="zoom_ai/zoom-scribe-en", file=AUDIO)

    first, second = (call.request.headers["x-zm-trackingid"] for call in route.calls)
    assert first != second


def test_api_base_override_keeps_the_scribe_route(zoom_ai_env: None, respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post("https://zoom.example.test/v2/aiservices/scribe/transcribe").respond(
        json=SCRIBE_RESPONSE
    )

    litellm.transcription(
        model="zoom_ai/zoom-scribe-en", file=AUDIO, api_base="https://zoom.example.test/v2/aiservices/"
    )

    assert route.called


def test_spend_is_the_billed_seconds_times_the_model_rate(
    zoom_ai_env: None, local_model_cost_map: None, respx_mock: respx.MockRouter
) -> None:
    respx_mock.post(ZOOM_AI_FAST_URL).respond(
        json={**SCRIBE_RESPONSE, "model": "zoom-scribe-multilingual", "duration_sec": 6.0}
    )

    response: Final = litellm.transcription(model="zoom_ai/zoom-scribe-multilingual", file=AUDIO)

    cost: Final = litellm.completion_cost(
        completion_response=response, model="zoom_ai/zoom-scribe-multilingual", call_type="transcription"
    )
    assert cost == pytest.approx(SCRIBE_RESPONSE["usage"]["input_units"] * _rate("zoom-scribe-multilingual"))


@pytest.mark.parametrize("served_model", ["zoom-scribe-en", "zoom-scribe-fr", "zoom-scribe-multilingual", "whisper"])
def test_auto_model_lets_scribe_choose_and_is_billed_at_the_model_that_ran(
    zoom_ai_env: None, local_model_cost_map: None, respx_mock: respx.MockRouter, served_model: str
) -> None:
    route: Final = respx_mock.post(ZOOM_AI_FAST_URL).respond(json={**SCRIBE_RESPONSE, "model": served_model})

    response: Final = litellm.transcription(model="zoom_ai/auto", file=AUDIO, language="ko")

    assert "model" not in _sent_config(route.calls.last.request)
    cost: Final = litellm.completion_cost(completion_response=response, model="zoom_ai/auto", call_type="transcription")
    assert cost == pytest.approx(SCRIBE_RESPONSE["usage"]["input_units"] * _rate(served_model))


@pytest.mark.parametrize(
    ("status_code", "body", "error_type", "message"),
    [
        (401, {"code": 124, "message": "Invalid Access token"}, litellm.AuthenticationError, "Invalid Access token"),
        (
            400,
            {"code": 400, "message": "Model 'zoom-scribe-fr' does not support language 'en'."},
            litellm.BadRequestError,
            "does not support language",
        ),
    ],
)
@pytest.mark.asyncio
async def test_scribe_refusals_map_to_litellm_errors_with_zooms_reason(
    zoom_ai_env: None,
    respx_mock: respx.MockRouter,
    status_code: int,
    body: Mapping[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    respx_mock.post(ZOOM_AI_FAST_URL).respond(status_code=status_code, json=body)

    with pytest.raises(error_type, match=message):
        await litellm.atranscription(model="zoom_ai/zoom-scribe-en", file=AUDIO)


@pytest.mark.parametrize(
    "body", ["<html><body>private-content-marker</body></html>", "private-content-marker from a proxy"]
)
def test_an_error_body_without_zooms_json_reason_is_not_shown_to_the_caller(
    zoom_ai_env: None, respx_mock: respx.MockRouter, body: str
) -> None:
    respx_mock.post(ZOOM_AI_FAST_URL).respond(status_code=502, text=body)

    with pytest.raises(ZoomAIException) as error:
        litellm.transcription(model="zoom_ai/zoom-scribe-en", file=AUDIO, max_retries=0)

    assert error.value.status_code == 502
    assert "private-content-marker" not in str(error.value)


def test_a_file_object_can_be_transcribed_again_after_a_call(zoom_ai_env: None, respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(ZOOM_AI_FAST_URL).respond(json=SCRIBE_RESPONSE)
    audio: Final = io.BytesIO(AUDIO[1])

    litellm.transcription(model="zoom_ai/zoom-scribe-en", file=audio)
    litellm.transcription(model="zoom_ai/zoom-scribe-en", file=audio)

    assert [_form_fields(call.request)["file"] for call in route.calls] == [AUDIO[1], AUDIO[1]]


@pytest.mark.asyncio
async def test_missing_api_key_is_an_authentication_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ZOOM_AI_API_KEY", raising=False)

    with pytest.raises(litellm.AuthenticationError, match="ZOOM_AI_API_KEY"):
        await litellm.atranscription(model="zoom_ai/zoom-scribe-en", file=AUDIO)


def test_unsupported_response_format_is_rejected_before_sending(
    zoom_ai_env: None, respx_mock: respx.MockRouter
) -> None:
    route: Final = respx_mock.post(ZOOM_AI_FAST_URL).respond(json=SCRIBE_RESPONSE)

    with pytest.raises(litellm.UnsupportedParamsError, match="response_format"):
        litellm.transcription(model="zoom_ai/zoom-scribe-en", file=AUDIO, response_format="text")

    assert not route.called


@pytest.mark.parametrize(
    ("model", "params", "expected"),
    [
        ("zoom-scribe-en", {"language": "en"}, {"model": "zoom-scribe-en", "language": "en-US"}),
        ("zoom-scribe-pt", {"language": "pt"}, {"model": "zoom-scribe-pt", "language": "pt-BR"}),
        ("zoom-scribe-ar", {"language": "AR"}, {"model": "zoom-scribe-ar", "language": "ar-SA"}),
        ("zoom-scribe-pt", {"language": "pt-PT"}, {"model": "zoom-scribe-pt", "language": "pt-PT"}),
        ("zoom-scribe-xx", {"language": "xx"}, {"model": "zoom-scribe-xx", "language": "xx"}),
        ("zoom-scribe-fr", {}, {"model": "zoom-scribe-fr", "language": "fr-FR"}),
        ("zoom-scribe-de-pro", {}, {"model": "zoom-scribe-de-pro", "language": "de-DE"}),
        ("zoom-scribe-en-pro-v2.1", {}, {"model": "zoom-scribe-en-pro-v2.1", "language": "en-US"}),
        ("zoom-scribe-fr", {"language": "en"}, {"model": "zoom-scribe-fr", "language": "en-US"}),
        (
            "zoom-scribe-multilingual",
            {},
            {"model": "zoom-scribe-multilingual", "language_detection": {"mode": "continuous"}},
        ),
        (
            "zoom-scribe-multilingual",
            {"language": "es"},
            {"model": "zoom-scribe-multilingual", "language": "es-ES", "language_detection": {"mode": "continuous"}},
        ),
        (
            "zoom-scribe-multilingual",
            {"language_detection": "once"},
            {"model": "zoom-scribe-multilingual", "language_detection": {"mode": "once"}},
        ),
        ("a-future-model", {}, {"model": "a-future-model"}),
        ("auto", {"language": "ko"}, {"language": "ko-KR"}),
        ("auto", {}, {}),
        ("whisper", {"language": "es"}, {"model": "whisper", "language": "es-ES"}),
        (
            "zoom-scribe-multilingual",
            {"language_detection": {"mode": "continuous", "candidate_languages": ["en-US", "es-ES"]}},
            {
                "model": "zoom-scribe-multilingual",
                "language_detection": {"mode": "continuous", "candidate_languages": ["en-US", "es-ES"]},
            },
        ),
        (
            "zoom-scribe-en",
            {"response_format": "diarized_json", "diarization": "false"},
            {"model": "zoom-scribe-en", "language": "en-US", "diarization": False},
        ),
        (
            "zoom-scribe-en",
            {"response_format": "diarized_json"},
            {"model": "zoom-scribe-en", "language": "en-US", "diarization": True},
        ),
    ],
)
def test_recognition_config_language_and_model(
    model: str, params: Mapping[str, object], expected: Mapping[str, object]
) -> None:
    assert zoom_ai_recognition_config(model, params) == expected


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({"diarization": "true", "word_time_offsets": "false"}, {"diarization": True, "word_time_offsets": False}),
        (
            {"channel_separation": True, "profanity_filter": False},
            {"channel_separation": True, "profanity_filter": False},
        ),
        ({"vocabulary": ["Zoom", " Scribe ", ""]}, {"vocabulary": {"phrases": ["Zoom", "Scribe"]}}),
        ({"vocabulary": "Zoom"}, {"vocabulary": {"phrases": ["Zoom"]}}),
        (
            {"segmentation_mode": "none", "vocabulary_id": "voc-1"},
            {"segmentation_mode": "none", "vocabulary_id": "voc-1"},
        ),
        (
            {"language_detection": '{"mode": "once", "candidate_languages": ["en-US", "fr-FR"]}'},
            {"language_detection": {"mode": "once", "candidate_languages": ["en-US", "fr-FR"]}},
        ),
        ({"not_a_scribe_option": "x", "diarization": None}, {}),
    ],
)
def test_provider_options_are_sent_with_scribes_types(
    params: Mapping[str, object], expected: Mapping[str, object]
) -> None:
    config: Final = zoom_ai_recognition_config("zoom-scribe-en", params)

    assert {key: value for key, value in config.items() if key not in ("model", "language")} == expected


def test_an_option_scribe_cannot_parse_is_a_bad_request() -> None:
    with pytest.raises(ZoomAIException, match="diarization") as error:
        zoom_ai_recognition_config("zoom-scribe-en", {"diarization": "sometimes"})

    assert error.value.status_code == 400


def test_diarized_json_returns_scribes_speaker_segments(zoom_ai_env: None, respx_mock: respx.MockRouter) -> None:
    segments: Final = [
        {"speaker": "Speaker 1", "start": 0.0, "end": 3.9, "text": "I'd like to return a mug."},
        {"speaker": "Speaker 2", "start": 4.1, "end": 6.0, "text": "Sure, which order?"},
    ]
    route: Final = respx_mock.post(ZOOM_AI_FAST_URL).respond(
        json={
            **SCRIBE_RESPONSE,
            "result": {"text_display": "I'd like to return a mug. Sure, which order?", "segments": segments},
        }
    )

    response: Final = litellm.transcription(model="zoom_ai/zoom-scribe-en", file=AUDIO, response_format="diarized_json")

    assert _sent_config(route.calls.last.request)["diarization"] is True
    assert [(s["speaker"], s["start"], s["end"], s["text"]) for s in response["segments"]] == [
        (s["speaker"], s["start"], s["end"], s["text"]) for s in segments
    ]


WORDED_SEGMENTS: Final = [
    {
        "id": "utt_0000",
        "start": 0.0,
        "end": 1.2,
        "channel": 0,
        "speaker": "",
        "text": "Four score",
        "words": [{"word": "Four", "start": 0.0, "end": 0.6}, {"word": "score", "start": 0.6, "end": 1.2}],
    },
    {
        "id": "utt_0001",
        "start": 1.5,
        "end": 2.4,
        "channel": 1,
        "speaker": "",
        "text": "and seven",
        "words": [{"word": "and", "start": 1.5, "end": 1.8}, {"word": "seven", "start": 1.8, "end": 2.4}],
    },
]


def _worded_response() -> Mapping[str, object]:
    return {**SCRIBE_RESPONSE, "result": {"text_display": "Four score and seven", "segments": WORDED_SEGMENTS}}


def test_verbose_json_with_word_timestamps_returns_segments_and_words(
    zoom_ai_env: None, respx_mock: respx.MockRouter
) -> None:
    route: Final = respx_mock.post(ZOOM_AI_FAST_URL).respond(json=_worded_response())

    response: Final = litellm.transcription(
        model="zoom_ai/zoom-scribe-en",
        file=AUDIO,
        response_format="verbose_json",
        timestamp_granularities=["word", "segment"],
    )

    assert _sent_config(route.calls.last.request)["word_time_offsets"] is True
    assert [(s["start"], s["end"], s["text"], s["channel"]) for s in response["segments"]] == [
        (s["start"], s["end"], s["text"], s["channel"]) for s in WORDED_SEGMENTS
    ]
    assert "speaker" not in response["segments"][0]
    assert response["words"] == [word for segment in WORDED_SEGMENTS for word in segment["words"]]


@pytest.mark.parametrize("response_format", ["srt", "vtt"])
def test_subtitles_are_built_from_scribes_word_timings(
    zoom_ai_env: None, respx_mock: respx.MockRouter, response_format: str
) -> None:
    route: Final = respx_mock.post(ZOOM_AI_FAST_URL).respond(json=_worded_response())

    response: Final = litellm.transcription(model="zoom_ai/zoom-scribe-en", file=AUDIO, response_format=response_format)

    assert _sent_config(route.calls.last.request)["word_time_offsets"] is True
    assert "Four score and seven" in response.text
    assert "-->" in response.text


def test_text_polishing_is_requested_in_outputs_and_returned_beside_the_transcript(
    zoom_ai_env: None, respx_mock: respx.MockRouter
) -> None:
    route: Final = respx_mock.post(ZOOM_AI_FAST_URL).respond(
        json={**SCRIBE_RESPONSE, "outputs": {"text_polishing": {"text": "Hi there. What is the capital of France?"}}}
    )

    response: Final = litellm.transcription(model="zoom_ai/zoom-scribe-en", file=AUDIO, text_polishing="true")

    assert json.loads(_form_fields(route.calls.last.request)["outputs"]) == {"text_polishing": {"enabled": True}}
    assert response.text == SCRIBE_RESPONSE["result"]["text_display"]
    assert response["polished_text"] == "Hi there. What is the capital of France?"


def test_a_failed_polish_is_not_returned_as_text() -> None:
    response: Final = ZoomAIAudioTranscriptionConfig().transform_audio_transcription_response(
        httpx.Response(
            200, json={**SCRIBE_RESPONSE, "outputs": {"text_polishing": {"text": "[polishing_failed: timeout]"}}}
        )
    )

    assert "polished_text" not in response.model_dump()


def test_segments_carry_the_language_scribe_detected() -> None:
    response: Final = ZoomAIAudioTranscriptionConfig().transform_audio_transcription_response(
        httpx.Response(
            200,
            json={
                **SCRIBE_RESPONSE,
                "result": {"text_display": "Hola", "language": "es", "segments": [{"text": "Hola", "language": "es"}]},
            },
        )
    )

    assert response["language"] == "es"
    assert response["segments"][0]["language"] == "es"


def test_text_polishing_off_sends_no_outputs_field(zoom_ai_env: None, respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(ZOOM_AI_FAST_URL).respond(json=SCRIBE_RESPONSE)

    litellm.transcription(model="zoom_ai/zoom-scribe-en", file=AUDIO, text_polishing="false")

    assert "outputs" not in _form_fields(route.calls.last.request)


def test_drop_params_drops_an_unsupported_response_format_and_still_transcribes(
    zoom_ai_env: None, respx_mock: respx.MockRouter
) -> None:
    route: Final = respx_mock.post(ZOOM_AI_FAST_URL).respond(json=SCRIBE_RESPONSE)

    response: Final = litellm.transcription(
        model="zoom_ai/zoom-scribe-en", file=AUDIO, response_format="text", drop_params=True
    )

    assert route.called
    assert response.text == SCRIBE_RESPONSE["result"]["text_display"]


def test_spend_falls_back_to_the_audio_duration_when_scribe_reports_no_usage(
    local_model_cost_map: None,
) -> None:
    response: Final = ZoomAIAudioTranscriptionConfig().transform_audio_transcription_response(
        httpx.Response(200, json={key: value for key, value in SCRIBE_RESPONSE.items() if key != "usage"})
    )

    cost: Final = litellm.completion_cost(
        completion_response=response, model="zoom_ai/zoom-scribe-en", call_type="transcription"
    )
    assert cost == pytest.approx(SCRIBE_RESPONSE["duration_sec"] * _rate("zoom-scribe-en"))


def test_the_upload_names_only_the_file_not_its_directory(zoom_ai_env: None, respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(ZOOM_AI_FAST_URL).respond(json=SCRIBE_RESPONSE)

    litellm.transcription(model="zoom_ai/zoom-scribe-en", file=("../clips/speech.wav", AUDIO[1], "audio/wav"))

    disposition: Final = route.calls.last.request.content.split(b'name="file"; filename="', 1)[1].split(b'"', 1)[0]
    assert disposition == b"speech.wav"


@pytest.mark.parametrize(
    ("result", "expected_language"),
    [({"text_display": "Hola", "language": "es"}, "es"), ({"text_display": "Hola"}, None)],
)
def test_response_reports_the_detected_language_only_when_scribe_returns_one(
    result: Mapping[str, object], expected_language: str | None
) -> None:
    response: Final = ZoomAIAudioTranscriptionConfig().transform_audio_transcription_response(
        httpx.Response(200, json={**SCRIBE_RESPONSE, "result": result})
    )

    assert response.text == "Hola"
    assert response.get("language") == expected_language


def test_a_body_that_is_not_a_transcription_is_an_error() -> None:
    with pytest.raises(ZoomAIException, match="not a transcription"):
        ZoomAIAudioTranscriptionConfig().transform_audio_transcription_response(
            httpx.Response(200, json={"unexpected": True})
        )
