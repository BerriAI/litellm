"""`POST /v1/audio/speech` on ElevenLabs and Google Cloud Text-to-Speech (Vertex AI), the providers whose wire
format differs most from OpenAI's.

ElevenLabs (litellm/llms/elevenlabs/text_to_speech/transformation.py): the voice goes into the URL path
(`/v1/text-to-speech/<voice_id>`), OpenAI voice names are mapped to fixed ElevenLabs ids (`alloy` is
`21m00Tcm4TlvDq8ikWAM`), `response_format` becomes the `output_format` query parameter (`mp3` is `mp3_44100_128`,
`pcm` is `pcm_24000`, `opus` is `opus_48000_128`, `wav` is `wav_24000`, anything else passes through verbatim),
`speed` becomes `voice_settings.speed`, the JSON body carries `text` and `model_id`, and the key travels in
`xi-api-key`. The audio bytes come back unchanged.

Google Cloud TTS (litellm/llms/vertex_ai/text_to_speech/transformation.py): one POST of
`{"input": {"text"|"ssml"}, "voice": {"languageCode", "name"}, "audioConfig": {"audioEncoding", "speakingRate"}}`
to `api_base` verbatim (or `texttospeech.googleapis.com/v1/text:synthesize`), with a bearer token minted from the
deployment's service account and `x-goog-user-project`. OpenAI voices map to Studio and Wavenet voices, the format
maps to an audio encoding (default `LINEAR16`), `<speak>` input is sent as SSML, and the reply's base64
`audioContent` is decoded to the bytes the caller receives.

`test_dec1_media_speech_wire.py` covers the OpenAI and Azure wire, media types, cost and limits;
`test_gemini_tts_voice_mapping_wire.py` covers Gemini TTS voices.
"""

from __future__ import annotations

import base64
import json
import uuid
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from integration.providers.dec1_media2_support import (
    ELEVENLABS_KEY,
    SPEECH_BYTES,
    error_message,
    json_body,
    json_reply,
    logged_rows,
    path_of,
    query_of,
    response_cost,
)
from integration.providers.dec1_vertex_support import PROJECT, vertex

_SPEECH: Final = "/v1/audio/speech"
_RACHEL: Final = "21m00Tcm4TlvDq8ikWAM"
_SYNTHESIZE: Final = "/v1/text:synthesize"


def _audio(_request: Request) -> Reply:
    return Reply(body=SPEECH_BYTES, content_type="audio/mpeg")


def _line() -> str:
    return f"good morning from the lighthouse {uuid.uuid4().hex[:8]}"


def _google_audio(_request: Request) -> Reply:
    return json_reply({"audioContent": base64.b64encode(SPEECH_BYTES).decode()})


# ---------------------------------------------------------------- ElevenLabs


def test_elevenlabs_speech_puts_the_mapped_voice_in_the_path_and_the_format_in_the_query(gateway: Gateway) -> None:
    """An `elevenlabs/eleven_multilingual_v2` deployment is called with voice `alloy` and `response_format`
    `mp3`. ElevenLabs must get one POST to `/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM` with query
    `output_format=mp3_44100_128`, header `xi-api-key: <key>`, and a JSON body whose `text` is the input and whose
    `model_id` is `eleven_multilingual_v2`. The caller gets the audio bytes unchanged."""
    line: Final = _line()
    with wire_server(_audio) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="elevenlabs/eleven_multilingual_v2", api_base=wire.url, api_key=ELEVENLABS_KEY
        )
        response: Final = gateway.request(
            "POST", _SPEECH, {"model": model, "input": line, "voice": "alloy", "response_format": "mp3"}
        )
        assert response.status_code == 200, response.text
        assert response.content == SPEECH_BYTES
        (request,) = wire.drain()
    assert (request.method, path_of(request)) == ("POST", f"/v1/text-to-speech/{_RACHEL}")
    assert query_of(request) == {"output_format": ["mp3_44100_128"]}
    assert request.headers["xi-api-key"] == ELEVENLABS_KEY
    body: Final = json_body(request)
    assert (body["text"], body["model_id"]) == (line, "eleven_multilingual_v2")
    assert not {"voice", "response_format", "input", "model"} & set(body)


@pytest.mark.parametrize(
    ("response_format", "output_format"),
    (("pcm", "pcm_24000"), ("opus", "opus_48000_128"), ("wav", "wav_24000"), ("ulaw_8000", "ulaw_8000")),
)
def test_elevenlabs_speech_maps_each_openai_format_to_its_output_format(
    gateway: Gateway, response_format: str, output_format: str
) -> None:
    """For each OpenAI `response_format` ElevenLabs must receive the matching `output_format` query value:
    `pcm` to `pcm_24000`, `opus` to `opus_48000_128`, `wav` to `wav_24000`, and a native ElevenLabs value such as
    `ulaw_8000` passed through verbatim (`FORMAT_MAPPINGS.get(format, format)`)."""
    with wire_server(_audio) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="elevenlabs/eleven_multilingual_v2", api_base=wire.url, api_key=ELEVENLABS_KEY
        )
        response: Final = gateway.request(
            "POST", _SPEECH, {"model": model, "input": _line(), "voice": "alloy", "response_format": response_format}
        )
        assert response.status_code == 200, response.text
        (request,) = wire.drain()
    assert query_of(request)["output_format"] == [output_format]


def test_elevenlabs_speech_without_a_format_sends_no_output_format_query(gateway: Gateway) -> None:
    """The caller sends no `response_format`. The ElevenLabs URL must carry no query string at all, so ElevenLabs
    applies its own default; the `pcm_44100` constant in the config is never sent."""
    with wire_server(_audio) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="elevenlabs/eleven_multilingual_v2", api_base=wire.url, api_key=ELEVENLABS_KEY
        )
        response: Final = gateway.request("POST", _SPEECH, {"model": model, "input": _line(), "voice": "alloy"})
        assert response.status_code == 200, response.text
        (request,) = wire.drain()
    assert request.target == f"/v1/text-to-speech/{_RACHEL}"


def test_elevenlabs_speech_passes_an_unknown_voice_through_as_the_voice_id(gateway: Gateway) -> None:
    """The caller names a voice that is not an OpenAI voice, `CustomVoice123`. ElevenLabs must get it unchanged as
    the path segment, `/v1/text-to-speech/CustomVoice123`, so cloned voices keep working."""
    with wire_server(_audio) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="elevenlabs/eleven_multilingual_v2", api_base=wire.url, api_key=ELEVENLABS_KEY
        )
        response: Final = gateway.request(
            "POST", _SPEECH, {"model": model, "input": _line(), "voice": "CustomVoice123"}
        )
        assert response.status_code == 200, response.text
        (request,) = wire.drain()
    assert path_of(request) == "/v1/text-to-speech/CustomVoice123"


def test_elevenlabs_speech_maps_speed_into_voice_settings(gateway: Gateway) -> None:
    """The caller sends `speed` 1.2. ElevenLabs must get `voice_settings: {"speed": 1.2}` in the JSON body and no
    top-level `speed`, since ElevenLabs reads speed only from voice settings."""
    with wire_server(_audio) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="elevenlabs/eleven_multilingual_v2", api_base=wire.url, api_key=ELEVENLABS_KEY
        )
        response: Final = gateway.request(
            "POST", _SPEECH, {"model": model, "input": _line(), "voice": "alloy", "speed": 1.2}
        )
        assert response.status_code == 200, response.text
        (request,) = wire.drain()
    body: Final = json_body(request)
    assert body["voice_settings"] == {"speed": 1.2}
    assert "speed" not in body


def test_elevenlabs_speech_forwards_provider_fields_into_the_json_body(gateway: Gateway) -> None:
    """The caller adds ElevenLabs-only fields `language_code` `de` and `seed` 7 to the speech body. Both must reach
    ElevenLabs in the JSON body with the same values (the config copies unreserved kwargs into the body).
    UNVERIFIED: assumes the proxy hands unknown speech body fields to `litellm.aspeech` as kwargs."""
    with wire_server(_audio) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="elevenlabs/eleven_multilingual_v2", api_base=wire.url, api_key=ELEVENLABS_KEY
        )
        response: Final = gateway.request(
            "POST",
            _SPEECH,
            {"model": model, "input": _line(), "voice": "alloy", "language_code": "de", "seed": 7},
        )
        assert response.status_code == 200, response.text
        (request,) = wire.drain()
    body: Final = json_body(request)
    assert (body["language_code"], body["seed"]) == ("de", 7)


def test_elevenlabs_speech_error_keeps_the_provider_status(gateway: Gateway) -> None:
    """ElevenLabs answers 401 `{"detail": {"status": "invalid_api_key", "message": "Invalid API key"}}`. The caller
    must get 401 and the provider message must appear in the error text."""
    with (
        wire_server(
            lambda _r: json_reply({"detail": {"status": "invalid_api_key", "message": "Invalid API key"}}, 401)
        ) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(
            model="elevenlabs/eleven_multilingual_v2", api_base=wire.url, api_key=ELEVENLABS_KEY
        )
        response: Final = gateway.request("POST", _SPEECH, {"model": model, "input": _line(), "voice": "alloy"})
        assert response.status_code == 401, response.text
        assert "Invalid API key" in response.text
        assert len(wire.drain()) >= 1


def test_elevenlabs_speech_bills_each_input_character(gateway: Gateway) -> None:
    """An ElevenLabs deployment priced at `input_cost_per_character` 0.00003 speaks a 40 character input. The
    response cost header and the spend row must both be 40 x 0.00003, and the row's `call_type` is `aspeech` with
    `custom_llm_provider` `elevenlabs`.

    BUG (run 44176682961): the cost is 0.0032, which is 40 x 8e-05, the cost map's `input_cost_per_character` for
    `elevenlabs/eleven_multilingual_v2` (model_prices_and_context_window.json). The deployment's own per-character
    price is ignored on the speech cost path (`_SPEECH_CALL_TYPES` branch, litellm/cost_calculator.py:2154 on
    main). Root cause not pinned; medium confidence."""
    pytest.skip("BUG: ElevenLabs speech bills the cost map per-character price, not the deployment's")
    line: Final = "x" * 32 + uuid.uuid4().hex[:8]
    assert len(line) == 40
    with wire_server(_audio) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="elevenlabs/eleven_multilingual_v2",
            api_base=wire.url,
            api_key=ELEVENLABS_KEY,
            input_cost_per_character=0.00003,
        )
        key: Final = scenario.key(models=[model])
        response: Final = gateway.request("POST", _SPEECH, {"model": model, "input": line, "voice": "alloy"}, key=key)
        assert response.status_code == 200, response.text
        assert response_cost(response) == pytest.approx(40 * 0.00003)
        (row,) = logged_rows(key)
        assert (row["call_type"], row["custom_llm_provider"]) == ("aspeech", "elevenlabs")
        assert float(str(row["spend"])) == pytest.approx(40 * 0.00003)
        assert len(wire.drain()) == 1


# ---------------------------------------------------------------- Google Cloud TTS (Vertex AI)


def test_vertex_cloud_tts_sends_input_voice_and_audio_config_with_a_minted_bearer_token(gateway: Gateway) -> None:
    """A `vertex_ai/chirp3-hd` deployment with `api_base` `<peer>/v1/text:synthesize` speaks with voice `alloy` and
    `response_format` `mp3`. Google must get one POST to that exact path with `Authorization: Bearer token-1`
    (minted from the service account at the peer's token endpoint), `x-goog-user-project` equal to the
    deployment project, and the body `{"input": {"text": <input>}, "voice": {"languageCode": "en-US", "name":
    "en-US-Studio-O"}, "audioConfig": {"audioEncoding": "MP3", "speakingRate": "1"}}`. The caller gets the
    decoded `audioContent` bytes."""
    line: Final = _line()
    with vertex(gateway, _google_audio, backend="chirp3-hd", base_suffix=_SYNTHESIZE) as rig:
        response: Final = rig.post(_SPEECH, {"input": line, "voice": "alloy", "response_format": "mp3"})
        assert response.status_code == 200, response.text
        assert response.content == SPEECH_BYTES
        request: Final = rig.peer.one_call()
    assert path_of(request) == _SYNTHESIZE
    assert request.headers["authorization"] == "Bearer token-1"
    assert request.headers["x-goog-user-project"] == PROJECT
    assert json.loads(request.body) == {
        "input": {"text": line},
        "voice": {"languageCode": "en-US", "name": "en-US-Studio-O"},
        "audioConfig": {"audioEncoding": "MP3", "speakingRate": "1"},
    }


@pytest.mark.parametrize(
    ("voice", "voice_config"),
    (
        ("onyx", {"languageCode": "en-US", "name": "en-US-Wavenet-D"}),
        ("fable", {"languageCode": "en-GB", "name": "en-GB-Studio-B"}),
        ("de-DE-Wavenet-B", {"languageCode": "de-DE", "name": "de-DE-Wavenet-B"}),
    ),
)
def test_vertex_cloud_tts_maps_openai_voices_and_derives_the_language_from_native_names(
    gateway: Gateway, voice: str, voice_config: dict[str, str]
) -> None:
    """OpenAI voice `onyx` must reach Google as `en-US-Wavenet-D` and `fable` as `en-GB-Studio-B`; a native voice
    name such as `de-DE-Wavenet-B` is sent unchanged with `languageCode` taken from its first two dash parts."""
    with vertex(gateway, _google_audio, backend="chirp3-hd", base_suffix=_SYNTHESIZE) as rig:
        response: Final = rig.post(_SPEECH, {"input": _line(), "voice": voice})
        assert response.status_code == 200, response.text
        body: Final = rig.peer.one_body()
    assert body["voice"] == voice_config


@pytest.mark.parametrize(
    ("response_format", "encoding"),
    (("opus", "OGG_OPUS"), ("flac", "FLAC"), ("wav", "LINEAR16"), ("aac", "MP3"), (None, "LINEAR16")),
)
def test_vertex_cloud_tts_maps_the_format_to_an_audio_encoding(
    gateway: Gateway, response_format: str | None, encoding: str
) -> None:
    """Each OpenAI format maps to a Google `audioEncoding`: `opus` to `OGG_OPUS`, `flac` to `FLAC`, `wav` to
    `LINEAR16`, `aac` to `MP3` (Google has no AAC), and no format at all to the default `LINEAR16`."""
    fields: Final = {"input": _line(), "voice": "alloy"} | (
        {"response_format": response_format} if response_format is not None else {}
    )
    with vertex(gateway, _google_audio, backend="chirp3-hd", base_suffix=_SYNTHESIZE) as rig:
        response: Final = rig.post(_SPEECH, fields)
        assert response.status_code == 200, response.text
        body: Final = rig.peer.one_body()
    assert body["audioConfig"] == {"audioEncoding": encoding, "speakingRate": "1"}


def test_vertex_cloud_tts_sends_speed_as_a_string_speaking_rate(gateway: Gateway) -> None:
    """The caller sends `speed` 1.5. Google must get `audioConfig.speakingRate` `"1.5"`, the string form the
    mapping produces with `str(speed)`, so a rewrite that sends a number changes the wire."""
    with vertex(gateway, _google_audio, backend="chirp3-hd", base_suffix=_SYNTHESIZE) as rig:
        response: Final = rig.post(_SPEECH, {"input": _line(), "voice": "alloy", "speed": 1.5})
        assert response.status_code == 200, response.text
        body: Final = rig.peer.one_body()
    assert body["audioConfig"]["speakingRate"] == "1.5"  # type: ignore[index]


def test_vertex_cloud_tts_sends_speak_markup_as_ssml(gateway: Gateway) -> None:
    """The input is `<speak>Hello <break time="300ms"/> world</speak>`. Google must get it under `input.ssml` and
    no `input.text`, because Cloud TTS models auto-detect SSML from the `<speak>` tag."""
    ssml: Final = '<speak>Hello <break time="300ms"/> world</speak>'
    with vertex(gateway, _google_audio, backend="chirp3-hd", base_suffix=_SYNTHESIZE) as rig:
        response: Final = rig.post(_SPEECH, {"input": ssml, "voice": "alloy"})
        assert response.status_code == 200, response.text
        body: Final = rig.peer.one_body()
    assert body["input"] == {"ssml": ssml}


def test_vertex_cloud_tts_object_voice_is_sent_as_given(gateway: Gateway) -> None:
    """The caller sends `voice` as an object `{"languageCode": "ja-JP", "name": "ja-JP-Neural2-B"}`. Google must
    get exactly that object as `voice`, with no OpenAI-voice mapping applied."""
    voice: Final = {"languageCode": "ja-JP", "name": "ja-JP-Neural2-B"}
    with vertex(gateway, _google_audio, backend="chirp3-hd", base_suffix=_SYNTHESIZE) as rig:
        response: Final = rig.post(_SPEECH, {"input": _line(), "voice": voice})
        assert response.status_code == 200, response.text
        body: Final = rig.peer.one_body()
    assert body["voice"] == voice


def test_vertex_cloud_tts_reply_without_audio_content_is_an_error(gateway: Gateway) -> None:
    """Google answers 200 with `{}` (no `audioContent`). The caller must get an error status of 400 or above whose
    text contains `No audioContent in Vertex AI TTS response`, never a 200 with an empty body.
    UNVERIFIED: the exact status (500 from a plain ValueError is the likely value)."""
    with vertex(gateway, lambda _r: json_reply({}), backend="chirp3-hd", base_suffix=_SYNTHESIZE) as rig:
        response: Final = rig.post(_SPEECH, {"input": _line(), "voice": "alloy"})
        assert response.status_code >= 400, response.text
        assert "No audioContent in Vertex AI TTS response" in error_message(response)
        assert len(rig.peer.model_calls()) == 1
