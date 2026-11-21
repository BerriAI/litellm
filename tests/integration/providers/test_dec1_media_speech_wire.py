"""`POST /v1/audio/speech`: text in, binary audio out.

The proxy forwards the JSON body to the provider's speech endpoint and streams the provider's audio bytes back
to the caller unchanged. The response `content-type` is the provider's own `audio/*` type with any parameters
stripped, or else the type implied by the requested `response_format`, or else `audio/mpeg`
(litellm/litellm_core_utils/audio_utils/utils.py:403). Speech is billed per input character when the
deployment has `input_cost_per_character`, and the call is logged with `call_type` `aspeech`.

`compatibility/test_audio_speech_wire.py` already covers the full body passthrough, chunked audio and the wav,
flac, pcm and opus media types; this file covers the remaining media-type rules, the Azure wire, validation,
cost, limits and fallbacks.
"""

from __future__ import annotations

import uuid
from hashlib import sha256
from typing import Final
from urllib.parse import parse_qs

import pytest
from integration._support.client import Gateway, eventually
from integration._support.wire import Reply, Request, wire_server
from integration.providers.dec1_media_support import (
    AZURE_KEY,
    OPENAI_KEY,
    SPEECH_BYTES,
    error_message,
    error_of,
    json_body,
    key_spend,
    logged_rows,
    openai_error_reply,
    response_cost,
    settled_key_spend,
    targets,
)

_SPEECH: Final = "/v1/audio/speech"
_RATE: Final = 0.00003
_BUDGET: Final = 0.001


def _audio(content_type: str = "audio/mpeg", body: bytes = SPEECH_BYTES) -> Reply:
    return Reply(body=body, content_type=content_type)


def _mp3(_request: Request) -> Reply:
    return _audio()


def _line() -> str:
    return f"the quick brown fox {uuid.uuid4().hex}"


@pytest.mark.parametrize(("response_format", "media_type"), (("mp3", "audio/mpeg"), ("aac", "audio/aac")))
def test_speech_media_type_follows_the_requested_format_when_the_provider_sends_a_generic_type(
    gateway: Gateway, response_format: str, media_type: str
) -> None:
    """The provider answers the audio as `application/octet-stream`. With `response_format` `mp3` the caller
    must get `audio/mpeg`, and with `aac` it must get `audio/aac`: the requested format decides the type when
    the provider's own type is not an audio type. The bytes are returned unchanged."""
    with wire_server(lambda _request: _audio("application/octet-stream")) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-4o-mini-tts", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        response: Final = gateway.request(
            "POST", _SPEECH, {"model": model, "input": _line(), "voice": "alloy", "response_format": response_format}
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == media_type
        assert response.content == SPEECH_BYTES
        assert targets(wire) == [("POST", _SPEECH)]


def test_speech_defaults_to_audio_mpeg_without_a_format_or_an_audio_type_from_the_provider(
    gateway: Gateway,
) -> None:
    """The caller sends no `response_format` and the provider labels the audio `application/octet-stream`. The
    response `content-type` must be `audio/mpeg`, the documented default for speech."""
    with wire_server(lambda _request: _audio("application/octet-stream")) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-4o-mini-tts", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        response: Final = gateway.request("POST", _SPEECH, {"model": model, "input": _line(), "voice": "alloy"})
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "audio/mpeg"
        assert response.content == SPEECH_BYTES
        assert targets(wire) == [("POST", _SPEECH)]


def test_speech_returns_the_provider_audio_type_lowercased_without_its_parameters(gateway: Gateway) -> None:
    """The provider labels raw PCM as `Audio/L16; rate=24000; channels=1`. The caller must get `audio/l16`: the
    provider's audio type wins over the requested `mp3`, but it is lowercased and its parameters are dropped,
    so a client cannot read the sample rate from the header."""
    with (
        wire_server(lambda _request: _audio("Audio/L16; rate=24000; channels=1")) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model="openai/gpt-4o-mini-tts", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        response: Final = gateway.request(
            "POST", _SPEECH, {"model": model, "input": _line(), "voice": "alloy", "response_format": "mp3"}
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "audio/l16"
        assert response.content == SPEECH_BYTES
        assert targets(wire) == [("POST", _SPEECH)]


def test_speech_returns_three_megabytes_of_audio_byte_for_byte(gateway: Gateway) -> None:
    """The provider answers 3 MB of audio in one body. The bytes the caller reads must have the same length and
    SHA-256, so re-chunking the audio into the proxy's own stream neither drops nor reorders data."""
    audio: Final = uuid.uuid4().bytes * (3 * 1024 * 1024 // 16)

    with wire_server(lambda _request: _audio("audio/wav", audio)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-4o-mini-tts", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        response: Final = gateway.request(
            "POST", _SPEECH, {"model": model, "input": _line(), "voice": "alloy", "response_format": "wav"}
        )
        assert response.status_code == 200
        assert (len(response.content), sha256(response.content).hexdigest()) == (
            len(audio),
            sha256(audio).hexdigest(),
        )
        assert targets(wire) == [("POST", _SPEECH)]


def test_streamed_speech_delivers_every_provider_chunk_in_order(gateway: Gateway) -> None:
    """The provider sends the audio as 40 chunked pieces of uneven size with a pause between them. A caller
    that reads the response as a stream must end up with exactly the concatenation of those pieces, and the
    response must not carry a `content-length`, because the proxy streams speech instead of buffering it."""
    pieces: Final = tuple(bytes([index]) * (1 + index * 37) for index in range(40))

    def respond(_request: Request) -> Reply:
        return Reply(chunks=pieces, content_type="audio/mpeg", pause_between_chunks=0.005)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-4o-mini-tts", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        with gateway.client.stream(
            "POST",
            _SPEECH,
            json={"model": model, "input": _line(), "voice": "alloy"},
            headers={"Authorization": f"Bearer {gateway.key}"},
        ) as response:
            assert response.status_code == 200
            received: Final = b"".join(response.iter_bytes())
            assert "content-length" not in response.headers
        assert received == b"".join(pieces)
        assert targets(wire) == [("POST", _SPEECH)]


def test_speech_without_a_voice_answers_400_before_the_provider(gateway: Gateway) -> None:
    """An OpenAI speech deployment is called with `model` and `input` but no `voice`. The proxy must answer 400
    with a message containing `'voice' is required to be passed as a string for OpenAI TTS` and must not call
    the provider (litellm/main.py speech, OpenAI branch)."""
    with wire_server(_mp3) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-4o-mini-tts", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        response: Final = gateway.request("POST", _SPEECH, {"model": model, "input": _line()})
        assert response.status_code == 400, response.text
        assert "'voice' is required to be passed as a string for OpenAI TTS" in error_message(response)
        assert wire.drain() == ()


def test_speech_with_an_unknown_model_answers_400_and_names_the_route(gateway: Gateway) -> None:
    """The caller names a speech model no deployment serves. The proxy must answer 400 with error code `"400"`
    and a message containing `/audio/speech: Invalid model name passed in model=<name>`.
    Product bug: CI run 42966523075 answered 400 with a body that has no `error` object. `audio_speech`
    re-raises an `HTTPException` as is (litellm/proxy/proxy_server.py:13954), so FastAPI answers
    `{"detail": ...}`, while its neighbouring endpoints turn the same exception into the OpenAI error envelope
    (litellm/proxy/proxy_server.py:13809 and :14147)."""
    pytest.skip(
        "BUG: /v1/audio/speech answers an unknown model with a FastAPI detail body, not the OpenAI error envelope"
    )
    unknown: Final = f"no-such-tts-{uuid.uuid4().hex[:8]}"
    response: Final = gateway.request("POST", _SPEECH, {"model": unknown, "input": "hello", "voice": "alloy"})
    assert response.status_code == 400, response.text
    assert error_of(response)["code"] == "400"
    assert f"/audio/speech: Invalid model name passed in model={unknown}" in error_message(response)


def test_azure_speech_uses_the_deployment_path_api_version_and_api_key_header(gateway: Gateway) -> None:
    """An Azure OpenAI TTS deployment is called with a voice, a speed and a format. Azure must get one POST to
    `/openai/deployments/<deployment>/audio/speech?api-version=<version>` with the key in the `api-key` header
    and a JSON body carrying the input, voice, speed and response_format. The audio bytes come back unchanged."""
    line: Final = _line()

    def respond(request: Request) -> Reply:
        path, _, query = request.target.partition("?")
        assert (request.method, path) == ("POST", "/openai/deployments/tts-deployment/audio/speech"), request.target
        assert parse_qs(query) == {"api-version": ["2025-03-01-preview"]}, request.target
        assert request.headers["api-key"] == AZURE_KEY
        body: Final = json_body(request)
        assert (body["input"], body["voice"], body["speed"], body["response_format"]) == (line, "nova", 0.9, "wav")
        return _audio("audio/wav")

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="azure/tts-deployment", api_base=wire.url, api_key=AZURE_KEY, api_version="2025-03-01-preview"
        )
        response: Final = gateway.request(
            "POST",
            _SPEECH,
            {"model": model, "input": line, "voice": "nova", "speed": 0.9, "response_format": "wav"},
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "audio/wav"
        assert response.content == SPEECH_BYTES
        assert len(wire.drain()) == 1


def test_speech_bills_each_input_character_and_logs_an_aspeech_row(gateway: Gateway) -> None:
    """The deployment costs 0.00003 per input character and the caller sends a 50 character input. The response
    cost header must be 50 * 0.00003 and one spend row must land for the key with `call_type` `aspeech`,
    `status` `success` and the same spend. The input has no whitespace, because speech bills characters
    with whitespace removed (litellm/utils.py:6535).
    Product bug: CI run 42966523075 billed the `tts-1` cost map rate of 0.000015 per character, not the
    deployment's `input_cost_per_character`, which is a custom pricing parameter like `input_cost_per_token`.
    The speech branch of the cost calculator reads the model's cost map entry by its bare name
    (litellm/cost_calculator.py:700), so the deployment's own price never applies."""
    pytest.skip("BUG: speech ignores the deployment's input_cost_per_character and bills the cost map rate")
    line: Final = ("speechbilling" + uuid.uuid4().hex + "x" * 50)[:50]
    assert len(line) == 50 and not any(char.isspace() for char in line)
    with wire_server(_mp3) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/tts-1", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY, input_cost_per_character=_RATE
        )
        key: Final = scenario.key(models=[model])
        response: Final = gateway.request("POST", _SPEECH, {"model": model, "input": line, "voice": "alloy"}, key=key)
        assert response.status_code == 200, response.text
        assert response_cost(response) == pytest.approx(50 * _RATE)
        row: Final = logged_rows(key)[0]
        assert (row["call_type"], row["status"], row["model_group"]) == ("aspeech", "success", model), row
        assert float(str(row["spend"])) == pytest.approx(50 * _RATE)
        assert settled_key_spend(key, 50 * _RATE) == pytest.approx(50 * _RATE)
        assert len(wire.drain()) == 1


def test_failed_speech_logs_a_failure_row_and_charges_nothing(gateway: Gateway) -> None:
    """The provider rejects the speech request with 400. The caller gets 400 with the provider's message, one
    spend row with `status` `failure` and `spend` 0 lands for the key, and the key's spend stays 0."""

    def respond(_request: Request) -> Reply:
        return openai_error_reply(400, "scripted unsupported voice", "invalid_request_error")

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/tts-1", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY, input_cost_per_character=_RATE
        )
        key: Final = scenario.key(models=[model])
        response: Final = gateway.request(
            "POST", _SPEECH, {"model": model, "input": _line(), "voice": "alloy", "num_retries": 0}, key=key
        )
        assert response.status_code == 400, response.text
        assert "scripted unsupported voice" in error_message(response)
        row: Final = logged_rows(key)[0]
        assert (row["call_type"], row["status"]) == ("aspeech", "failure"), row
        assert float(str(row["spend"])) == 0.0
        assert len(wire.drain()) == 1


def test_speech_response_carries_the_call_id_and_model_id_headers(gateway: Gateway) -> None:
    """A successful speech call returns binary audio, so its metadata travels only in headers. The response
    must carry a non-empty `x-litellm-call-id` and `x-litellm-model-id`, and a client-supplied
    `x-litellm-call-id` request header must be echoed back unchanged."""
    call_id: Final = f"speech-{uuid.uuid4()}"
    with wire_server(_mp3) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/tts-1", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        response: Final = gateway.request(
            "POST",
            _SPEECH,
            {"model": model, "input": _line(), "voice": "alloy"},
            headers={"x-litellm-call-id": call_id},
        )
        assert response.status_code == 200, response.text
        assert response.headers["x-litellm-call-id"] == call_id
        assert response.headers["x-litellm-model-id"]
        assert targets(wire) == [("POST", _SPEECH)]


def test_speech_counts_against_the_key_rpm_limit(gateway: Gateway) -> None:
    """A key has `rpm_limit` 1. Its first speech call is served and the second must answer 429 without reaching
    the provider."""
    with wire_server(_mp3) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/tts-1", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        key: Final = scenario.key(models=[model], rpm_limit=1)
        first: Final = gateway.request("POST", _SPEECH, {"model": model, "input": _line(), "voice": "alloy"}, key=key)
        second: Final = gateway.request("POST", _SPEECH, {"model": model, "input": _line(), "voice": "alloy"}, key=key)
        assert (first.status_code, second.status_code) == (200, 429), (first.status_code, second.text)
        assert len(wire.drain()) == 1


def test_speech_is_refused_once_the_key_budget_is_spent(gateway: Gateway) -> None:
    """A key has `max_budget` 0.001 and one speech call of 100 characters costs more than that. After the first
    call's spend has landed on the key above the budget, the next speech call must answer 422 with error type
    `budget_exceeded` and must not reach the provider.
    The exact cost is not asserted here: the per-character rate speech bills is a separate, currently failing
    contract (`test_speech_bills_each_input_character_and_logs_an_aspeech_row`), and either rate puts 100
    characters over the budget."""
    with wire_server(_mp3) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/tts-1", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY, input_cost_per_character=_RATE
        )
        key: Final = scenario.key(models=[model], max_budget=_BUDGET)
        first: Final = gateway.request("POST", _SPEECH, {"model": model, "input": "a" * 100, "voice": "alloy"}, key=key)
        assert first.status_code == 200, first.text
        eventually(lambda: key_spend(key), lambda spend: spend > _BUDGET, seconds=70)
        refused: Final = gateway.request(
            "POST", _SPEECH, {"model": model, "input": "b" * 100, "voice": "alloy"}, key=key
        )
        assert refused.status_code == 422, refused.text
        assert error_of(refused)["type"] == "budget_exceeded", refused.text
        assert len(wire.drain()) == 1


def test_speech_request_fallback_serves_audio_from_the_second_deployment(gateway: Gateway) -> None:
    """The primary speech deployment's provider answers 500. The caller lists a second speech model in
    `fallbacks`. The call must answer 200 with the second provider's audio bytes, and the second provider must
    receive its own provider model name with the caller's input and voice."""
    line: Final = _line()
    fallback_audio: Final = b"fallback-audio-" + uuid.uuid4().bytes

    def fallback(request: Request) -> Reply:
        assert json_body(request) == {"model": "tts-1-hd", "input": line, "voice": "alloy"}
        return _audio("audio/mpeg", fallback_audio)

    with (
        wire_server(lambda _request: openai_error_reply(500, "scripted outage", "server_error")) as primary,
        wire_server(fallback) as secondary,
        gateway.scenario() as scenario,
    ):
        first: Final = scenario.model(model="openai/tts-1", api_base=f"{primary.url}/v1", api_key=OPENAI_KEY)
        second: Final = scenario.model(model="openai/tts-1-hd", api_base=f"{secondary.url}/v1", api_key=OPENAI_KEY)
        response: Final = gateway.request(
            "POST",
            _SPEECH,
            {"model": first, "input": line, "voice": "alloy", "fallbacks": [second], "num_retries": 0},
        )
        assert response.status_code == 200, response.text
        assert response.content == fallback_audio
        assert len(primary.drain()) >= 1
        assert len(secondary.drain()) == 1


def test_speech_fills_user_from_the_key_owner_when_the_caller_sends_none(gateway: Gateway) -> None:
    """A key owned by an internal user calls speech without a `user` field. The spend row for the call must be
    attributed to that user id in the `user` column, which the handler copies from the key before routing
    (litellm/proxy/proxy_server.py audio_speech)."""
    with wire_server(_mp3) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/tts-1", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        owner: Final = scenario.user(user_role="internal_user")
        key: Final = scenario.key(models=[model], user_id=owner)
        response: Final = gateway.request(
            "POST", _SPEECH, {"model": model, "input": _line(), "voice": "alloy"}, key=key
        )
        assert response.status_code == 200, response.text
        logged_rows(key)
        rows: Final = gateway.rows('SELECT "user", call_type FROM "LiteLLM_SpendLogs" WHERE "user" = %s', (owner,))
        assert [(row["user"], row["call_type"]) for row in rows] == [(owner, "aspeech")], rows
        assert len(wire.drain()) == 1
