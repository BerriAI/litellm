import base64
import json
from typing import Final
from unittest.mock import MagicMock

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.speechify.text_to_speech.transformation import (
    SpeechifyTextToSpeechConfig,
    SpeechifyTextToSpeechException,
)

SPEECH_URL: Final = "https://api.speechify.ai/v1/audio/speech"
AUDIO_BYTES: Final = b"ID3-fake-mp3-bytes"


def _speechify_reply(audio_format: str = "mp3") -> dict[str, object]:
    return {
        "audio_data": base64.b64encode(AUDIO_BYTES).decode(),
        "audio_format": audio_format,
        "billable_characters_count": 5,
        "speech_marks": {},
    }


def _client_recording(
    requests: list[httpx.Request], reply: httpx.Response
) -> AsyncHTTPHandler:  # mutable-ok: collects requests the mock transport sees
    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return reply

    return AsyncHTTPHandler(transport=httpx.MockTransport(handle))


async def test_speech_sends_speechify_request_and_returns_decoded_audio():
    requests: Final[list[httpx.Request]] = []  # mutable-ok: filled by the mock transport
    client: Final = _client_recording(requests, httpx.Response(200, json=_speechify_reply()))

    response: Final = await litellm.aspeech(
        model="speechify/simba-3.5",
        input="hello",
        voice="geffen_32",
        api_key="sk-speechify-test",
        client=client,
    )

    assert response.content == AUDIO_BYTES
    assert response.response.headers["content-type"] == "audio/mpeg"
    assert response.response.headers["content-length"] == str(len(AUDIO_BYTES))
    (request,) = requests
    assert str(request.url) == SPEECH_URL
    assert request.headers["authorization"] == "Bearer sk-speechify-test"
    assert json.loads(request.content) == {
        "model": "simba-3.5",
        "input": "hello",
        "voice_id": "geffen_32",
        "audio_format": "mp3",
    }


async def test_speech_maps_speechify_errors_by_their_status():
    requests: Final[list[httpx.Request]] = []  # mutable-ok: filled by the mock transport
    error_body: Final = {"error": {"code": "voice_not_found", "message": "voice not found"}}
    client: Final = _client_recording(requests, httpx.Response(404, json=error_body))

    with pytest.raises(litellm.NotFoundError, match="voice_not_found"):
        await litellm.aspeech(
            model="speechify/simba-3.5",
            input="hello",
            voice="not_a_voice",
            api_key="sk-speechify-test",
            client=client,
        )


@pytest.mark.parametrize(
    ("response_format", "audio_format"),
    [(None, "mp3"), ("mp3", "mp3"), ("wav", "wav"), ("pcm", "pcm"), ("aac", "aac"), ("opus", "ogg")],
)
def test_map_openai_params_maps_response_format_to_audio_format(response_format: str | None, audio_format: str):
    optional_params: Final = {} if response_format is None else {"response_format": response_format}
    voice, params = SpeechifyTextToSpeechConfig().map_openai_params(
        model="simba-3.5",
        optional_params=optional_params,
        voice=" geffen_32 ",
    )
    assert voice == "geffen_32"
    assert params == {"audio_format": audio_format}


def test_map_openai_params_drops_unsupported_openai_params_and_keeps_speechify_ones():
    _, params = SpeechifyTextToSpeechConfig().map_openai_params(
        model="simba-3.0",
        optional_params={"response_format": "mp3", "speed": 1.5, "instructions": "sound cheerful"},
        voice="geffen_32",
        kwargs={"language": "fr-FR", "output_format": "mp3_24000_128", "unrelated": "value"},
    )
    assert params == {"audio_format": "mp3", "language": "fr-FR", "output_format": "mp3_24000_128"}


@pytest.mark.parametrize(
    "api_base",
    [None, "https://api.speechify.ai", "https://api.speechify.ai/", "https://api.speechify.ai/v1"],
)
def test_get_complete_url_always_targets_the_speech_route(api_base: str | None):
    url: Final = SpeechifyTextToSpeechConfig().get_complete_url(model="simba-3.5", api_base=api_base, litellm_params={})
    assert url == SPEECH_URL


def test_validate_environment_reads_key_from_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SPEECHIFY_API_KEY", "sk-from-env")
    headers: Final = SpeechifyTextToSpeechConfig().validate_environment(headers={}, model="simba-3.5")
    assert headers["Authorization"] == "Bearer sk-from-env"


def test_validate_environment_requires_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SPEECHIFY_API_KEY", raising=False)
    with pytest.raises(SpeechifyTextToSpeechException, match="SPEECHIFY_API_KEY"):
        SpeechifyTextToSpeechConfig().validate_environment(headers={}, model="simba-3.5")


@pytest.mark.parametrize(
    ("returned_format", "content_type"),
    [
        ("wav", "audio/wav"),
        ("ogg", "audio/ogg"),
        ("L16;rate=24000;channels=1", "audio/pcm"),
        ("new-codec", "application/octet-stream"),
    ],
)
def test_transform_response_content_type_follows_the_returned_format(returned_format: str, content_type: str):
    raw_response: Final = httpx.Response(
        200,
        json=_speechify_reply(audio_format=returned_format),
        headers={"speechify-request-id": "req-123"},
        request=httpx.Request("POST", SPEECH_URL),
    )
    result: Final = SpeechifyTextToSpeechConfig().transform_text_to_speech_response(
        model="simba-3.5",
        raw_response=raw_response,
        logging_obj=MagicMock(),
    )
    assert result.content == AUDIO_BYTES
    assert result.response.headers["content-type"] == content_type
    assert result.response.headers["speechify-request-id"] == "req-123"


@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        json.dumps({"audio_format": "mp3"}).encode(),
        json.dumps(_speechify_reply() | {"audio_data": "QUJD!"}).encode(),
    ],
)
def test_transform_response_rejects_malformed_bodies(body: bytes):
    raw_response: Final = httpx.Response(200, content=body, request=httpx.Request("POST", SPEECH_URL))
    with pytest.raises(SpeechifyTextToSpeechException, match="Unexpected Speechify speech response"):
        SpeechifyTextToSpeechConfig().transform_text_to_speech_response(
            model="simba-3.5",
            raw_response=raw_response,
            logging_obj=MagicMock(),
        )


def test_price_map_lists_simba_3_5_at_the_per_character_rate():
    entry: Final = litellm.model_cost["speechify/simba-3.5"]
    assert entry["litellm_provider"] == "speechify"
    assert entry["mode"] == "audio_speech"
    assert entry["input_cost_per_character"] == 2e-05
    assert "speechify/simba-3.5" in litellm.speechify_models
