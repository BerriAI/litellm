import json
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from openai import OpenAI

_SPEECH_BYTES: Final = bytes(range(256)) * 32
_SPEECH_REQUEST: Final = {
    "model": "gpt-4o-mini-tts",
    "input": "hello",
    "voice": "alloy",
    "response_format": "wav",
    "speed": 1.25,
    "instructions": "cheerful",
}
_FORMAT_MEDIA_TYPES: Final = {
    "wav": "audio/wav",
    "flac": "audio/flac",
    "pcm": "audio/pcm",
    "opus": "audio/opus",
}


def _sdk(gateway: Gateway, path: str) -> OpenAI:
    return OpenAI(
        base_url=str(gateway.client.base_url).rstrip("/") + path,
        api_key=gateway.key,
        max_retries=0,
        http_client=httpx.Client(trust_env=False),
    )


@pytest.mark.parametrize("base_path", ("/v1", ""))
def test_audio_speech_forwards_the_full_json_body_and_streams_the_audio(gateway: Gateway, base_path: str) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/audio/speech"
        assert request.headers["authorization"] == "Bearer synthetic-openai-key"
        body: Final = json.loads(request.body)
        assert body == _SPEECH_REQUEST
        assert type(body["speed"]) is float
        return Reply(body=_SPEECH_BYTES, content_type="audio/wav")

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/gpt-4o-mini-tts", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        response: Final = _sdk(gateway, base_path).audio.speech.with_raw_response.create(
            model=model,
            input="hello",
            voice="alloy",
            response_format="wav",
            speed=1.25,
            instructions="cheerful",
        )
        assert response.status_code == 200, response.read()
        assert response.content == _SPEECH_BYTES
        assert response.headers["content-type"] == "audio/wav"
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/v1/audio/speech")]


def test_audio_speech_streams_chunked_audio_through_intact(gateway: Gateway) -> None:
    audio: Final = bytes((index % 256 for index in range(64 * 1024)))
    chunks: Final = tuple(audio[start : start + 4096] for start in range(0, len(audio), 4096))
    assert len(chunks) == 16

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/audio/speech"
        return Reply(chunks=chunks, content_type="audio/wav")

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/gpt-4o-mini-tts", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        response: Final = _sdk(gateway, "/v1").audio.speech.with_raw_response.create(
            model=model,
            input="hello",
            voice="alloy",
            response_format="mp3",
        )
        assert response.status_code == 200, response.read()
        assert response.content == audio
        assert response.headers["content-type"] == "audio/wav"
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/v1/audio/speech")]


@pytest.mark.parametrize("response_format,media_type", tuple(_FORMAT_MEDIA_TYPES.items()))
def test_audio_speech_derives_the_media_type_from_the_requested_format(
    gateway: Gateway, response_format: str, media_type: str
) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/audio/speech"
        return Reply(body=_SPEECH_BYTES, content_type="application/octet-stream")

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/gpt-4o-mini-tts", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        response: Final = _sdk(gateway, "/v1").audio.speech.with_raw_response.create(
            model=model,
            input="hello",
            voice="alloy",
            response_format=response_format,
        )
        assert response.status_code == 200, response.read()
        assert response.headers["content-type"] == media_type
        assert response.content == _SPEECH_BYTES
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/v1/audio/speech")]
