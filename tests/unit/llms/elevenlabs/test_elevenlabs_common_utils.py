import json
from typing import Final

import pytest

import litellm
from litellm.llms.elevenlabs.common_utils import (
    elevenlabs_tts_characters,
    elevenlabs_tts_client_text,
    elevenlabs_tts_is_priced,
    elevenlabs_tts_model,
    elevenlabs_tts_recorded_upstream_frame,
    elevenlabs_tts_registry_key,
    elevenlabs_tts_requested_model,
    elevenlabs_tts_websocket_target,
    is_elevenlabs_voice_id,
)
from litellm.types.passthrough_endpoints.pass_through_endpoints import ForwardedClientText, RefusedClientText

pytestmark: Final = pytest.mark.usefixtures("local_model_cost_map")

DEFAULT_MODEL: Final = "eleven_multilingual_v2"
STREAM_INPUT: Final = "wss://api.elevenlabs.io/v1/text-to-speech/voice1/stream-input"
FLASH_URL: Final = f"{STREAM_INPUT}?model_id=eleven_flash_v2_5"


@pytest.mark.parametrize(
    ("api_base", "stream", "query_string", "expected"),
    [
        pytest.param(
            None,
            "stream-input",
            "model_id=eleven_flash_v2_5&output_format=pcm_16000",
            f"{STREAM_INPUT}?model_id=eleven_flash_v2_5&output_format=pcm_16000",
            id="default base",
        ),
        pytest.param(
            None,
            "multi-stream-input",
            "model_id=eleven_flash_v2_5",
            "wss://api.elevenlabs.io/v1/text-to-speech/voice1/multi-stream-input?model_id=eleven_flash_v2_5",
            id="multi stream path",
        ),
        pytest.param(
            None,
            "stream-input",
            "output_format=pcm_16000",
            f"{STREAM_INPUT}?output_format=pcm_16000&model_id={DEFAULT_MODEL}",
            id="model added when missing",
        ),
        pytest.param(
            None,
            "stream-input",
            "model_id=&sync_alignment=true",
            f"{STREAM_INPUT}?sync_alignment=true&model_id={DEFAULT_MODEL}",
            id="empty model replaced",
        ),
        pytest.param(
            None,
            "stream-input",
            "model_id=eleven_flash_v2_5&seed=1&model_id=eleven_multilingual_v2",
            f"{STREAM_INPUT}?model_id=eleven_flash_v2_5&seed=1",
            id="second model dropped",
        ),
        pytest.param(
            None,
            "stream-input",
            "authorization=Bearer%20tok&model_id=eleven_flash_v2_5&single_use_token=sut&inactivity_timeout=60",
            f"{STREAM_INPUT}?model_id=eleven_flash_v2_5&inactivity_timeout=60",
            id="caller credentials dropped",
        ),
        pytest.param(
            "http://localhost:9000/",
            "stream-input",
            "model_id=eleven_turbo_v2_5&pronunciation_dictionary_locators=a&pronunciation_dictionary_locators=b",
            "ws://localhost:9000/v1/text-to-speech/voice1/stream-input"
            "?model_id=eleven_turbo_v2_5&pronunciation_dictionary_locators=a&pronunciation_dictionary_locators=b",
            id="custom http base becomes ws and repeated keys stay",
        ),
        pytest.param(
            "https://api.eu.residency.elevenlabs.io",
            "stream-input",
            "model_id=eleven_flash_v2_5",
            "wss://api.eu.residency.elevenlabs.io/v1/text-to-speech/voice1/stream-input?model_id=eleven_flash_v2_5",
            id="custom https base becomes wss",
        ),
    ],
)
def test_websocket_target(api_base, stream, query_string, expected):
    assert elevenlabs_tts_websocket_target(api_base, "voice1", stream, query_string) == expected


@pytest.mark.parametrize(
    ("query_string", "expected"),
    [
        ("model_id=eleven_flash_v2_5", "eleven_flash_v2_5"),
        ("model_id=eleven_flash_v2_5&model_id=eleven_v3", "eleven_flash_v2_5"),
        ("", DEFAULT_MODEL),
        ("model_id=", DEFAULT_MODEL),
        ("output_format=pcm_16000", DEFAULT_MODEL),
    ],
)
def test_requested_model_is_the_model_the_target_carries(query_string, expected):
    target = elevenlabs_tts_websocket_target(None, "voice1", "stream-input", query_string)
    assert elevenlabs_tts_requested_model(query_string) == expected
    assert elevenlabs_tts_model(target) == expected
    assert elevenlabs_tts_registry_key(target) == f"elevenlabs/{expected}"


@pytest.mark.parametrize("voice_id", ["21m00Tcm4TlvDq8ikWAM", "my_voice-2", "a"])
def test_plain_voice_ids_are_accepted(voice_id):
    assert is_elevenlabs_voice_id(voice_id)


@pytest.mark.parametrize(
    "voice_id",
    ["", "..", "a/b", "a?model_id=eleven_v3", "a#b", "a%2Fb", "a b", "voixé", "abc\n", "a\\b"],
)
def test_voice_ids_that_could_change_the_upstream_url_are_refused(voice_id):
    assert not is_elevenlabs_voice_id(voice_id)


def test_shipped_flash_row_is_priced():
    assert elevenlabs_tts_is_priced(FLASH_URL)


def test_operator_added_per_character_row_is_priced(monkeypatch):
    monkeypatch.setitem(
        litellm.model_cost,
        "elevenlabs/eleven_operator_custom",
        {"input_cost_per_character": 1e-05, "litellm_provider": "elevenlabs", "mode": "audio_speech"},
    )
    assert elevenlabs_tts_is_priced(FLASH_URL.replace("eleven_flash_v2_5", "eleven_operator_custom"))


@pytest.mark.parametrize(
    ("model_id", "row"),
    [
        pytest.param("eleven_unknown_model", None, id="no row"),
        pytest.param(
            "eleven_seconds_only",
            {"input_cost_per_second": 1e-04, "litellm_provider": "elevenlabs", "mode": "audio_transcription"},
            id="per second row",
        ),
        pytest.param(
            "eleven_zero_rate",
            {"input_cost_per_character": 0, "litellm_provider": "elevenlabs", "mode": "audio_speech"},
            id="zero rate",
        ),
    ],
)
def test_models_without_a_per_character_rate_are_unpriced(monkeypatch, model_id, row):
    if row is not None:
        monkeypatch.setitem(litellm.model_cost, f"elevenlabs/{model_id}", row)
    assert not elevenlabs_tts_is_priced(FLASH_URL.replace("eleven_flash_v2_5", model_id))


def test_a_row_under_another_key_does_not_price_the_elevenlabs_model(monkeypatch):
    monkeypatch.setitem(
        litellm.model_cost,
        "eleven_bare_key_only",
        {"input_cost_per_character": 1e-05, "litellm_provider": "elevenlabs", "mode": "audio_speech"},
    )
    assert not elevenlabs_tts_is_priced(FLASH_URL.replace("eleven_flash_v2_5", "eleven_bare_key_only"))


@pytest.mark.parametrize(
    ("text", "forwarded", "recorded"),
    [
        pytest.param(
            '{"text":"Hello ",   "flush":true}', {"text": "Hello ", "flush": True}, {"text": "Hello "}, id="text frame"
        ),
        pytest.param(
            '{"text": "The authorization form is ready "}',
            {"text": "The authorization form is ready "},
            {"text": "The authorization form is ready "},
            id="credential word inside the text",
        ),
        pytest.param('{"text": ""}', {"text": ""}, {"text": ""}, id="end of stream"),
        pytest.param(
            '{"context_id": "ctx", "close_context": true}',
            {"context_id": "ctx", "close_context": True},
            None,
            id="close context has no text",
        ),
        pytest.param('{"text": "a", "text": "bbbb"}', {"text": "bbbb"}, {"text": "bbbb"}, id="last duplicate wins"),
    ],
)
def test_client_text_forwards_the_parsed_frame_and_records_the_text_it_carries(text, forwarded, recorded):
    decision = elevenlabs_tts_client_text(text)

    assert isinstance(decision, ForwardedClientText)
    assert json.loads(decision.text) == forwarded
    assert decision.recorded == recorded


@pytest.mark.parametrize(
    "text",
    [
        pytest.param('{"text": " ", "xi_api_key": "sk-litellm-virtual", "voice_settings": {"a": 1}}', id="xi_api_key"),
        pytest.param('{"text": " ", "xi-api-key": "sk-litellm-virtual", "voice_settings": {"a": 1}}', id="xi-api-key"),
        pytest.param('{"text": " ", "authorization": "sk-litellm-virtual", "voice_settings": {"a": 1}}', id="auth"),
        pytest.param('{"text": " ", "xi\\u005fapi_key": "sk-litellm-virtual", "voice_settings": {"a": 1}}', id="esc"),
    ],
)
def test_caller_credential_fields_never_reach_elevenlabs(text):
    decision = elevenlabs_tts_client_text(text)

    assert isinstance(decision, ForwardedClientText)
    assert "sk-litellm-virtual" not in decision.text
    assert json.loads(decision.text) == {"text": " ", "voice_settings": {"a": 1}}
    assert decision.recorded == {"text": " "}


@pytest.mark.parametrize(
    "text",
    [
        pytest.param('{"text": "free audio", "x": "\\ud800"}', id="lone surrogate"),
        pytest.param('{"text": "free audio", "x": ' + "[" * 300 + "]" * 300 + "}", id="nested past the parser limit"),
        pytest.param('{"text": 5}', id="non string text"),
        pytest.param("not json", id="not json"),
        pytest.param('"just a string"', id="not an object"),
        pytest.param('["xi_api_key"]', id="array"),
    ],
)
def test_client_text_the_billing_parser_cannot_read_is_refused_instead_of_forwarded(text):
    decision = elevenlabs_tts_client_text(text)

    assert isinstance(decision, RefusedClientText)
    assert decision.code == 1007


def _audio_frame(spoken: str | None) -> dict[str, object]:
    alignment = None if spoken is None else {"chars": list(spoken), "charStartTimesMs": [0], "charDurationsMs": [50]}
    return {
        "audio": "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjYxLjcuMTAw",
        "alignment": alignment,
        "normalizedAlignment": alignment,
        "contextId": "ctx",
        "isFinal": None,
    }


def test_recorded_upstream_frame_drops_audio_and_alignment_but_keeps_the_aligned_character_count():
    assert elevenlabs_tts_recorded_upstream_frame(_audio_frame("Hello")) == {
        "contextId": "ctx",
        "isFinal": None,
        "aligned_characters": 5,
    }
    assert elevenlabs_tts_recorded_upstream_frame(_audio_frame(None)) == {
        "contextId": "ctx",
        "isFinal": None,
        "aligned_characters": 0,
    }
    assert elevenlabs_tts_recorded_upstream_frame({"isFinal": True}) == {"isFinal": True}
    assert elevenlabs_tts_recorded_upstream_frame({"message": "Invalid model", "error": "invalid_request"}) == {
        "message": "Invalid model",
        "error": "invalid_request",
    }


def _session(client_texts: tuple[str, ...], *upstream: dict[str, object]) -> int:
    return elevenlabs_tts_characters(
        tuple({"text": text} for text in client_texts),
        tuple(elevenlabs_tts_recorded_upstream_frame(frame) for frame in upstream),
    )


def test_characters_are_the_ones_elevenlabs_spoke_not_the_ones_sent():
    sent = (" ", "Hello world. ", "¿Qué tal? ", "This part was cut off before it was spoken. ")

    spoken = (_audio_frame("Hello world."), _audio_frame(None), _audio_frame("¿Qué tal?"), {"isFinal": True})

    assert _session(sent, *spoken) == len("Hello world.") + len("¿Qué tal?")


def test_a_session_that_got_no_audio_back_bills_nothing():
    assert _session((" ", "Text that never reached the generation threshold "), {"isFinal": True}) == 0
    assert (
        _session(
            (" ", "Text the caller dropped before any audio "),
        )
        == 0
    )


def test_audio_without_any_alignment_bills_every_character_sent():
    sent = (" ", "Hello world. ", "¿Qué tal? ", "")

    assert _session(sent, _audio_frame(None), _audio_frame(None), {"isFinal": True}) == (
        1 + len("Hello world. ") + len("¿Qué tal? ")
    )
    assert (
        elevenlabs_tts_characters(
            ({"close_context": True}, {"text": None}), (elevenlabs_tts_recorded_upstream_frame(_audio_frame(None)),)
        )
        == 0
    )
