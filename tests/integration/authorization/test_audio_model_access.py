import json
import mimetypes
from email.parser import BytesParser
from email.policy import HTTP
from typing import Final

import pytest

from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server

_WAV_BYTES: Final = (
    b"RIFF\x30\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
    b"\x40\x1f\x00\x00\x40\x1f\x00\x00\x01\x00\x08\x00data\x08\x00\x00\x00"
    b"\x00\x01\x02\x03\x04\x05\x06\x07"
)
_WAV_CONTENT_TYPE: Final = mimetypes.guess_type("a.wav")[0] or "application/octet-stream"
_SPEECH_BYTES: Final = bytes(range(256)) * 32


def _transcription_model(request: Request) -> str:
    envelope: Final = f"content-type: {request.headers['content-type']}\r\n\r\n".encode() + request.body
    parsed: Final = BytesParser(policy=HTTP).parsebytes(envelope)
    assert parsed.is_multipart(), request.headers["content-type"]
    fields: Final = {
        part.get_param("name", header="content-disposition"): part.get_payload(decode=True).decode()
        for part in parsed.iter_parts()
        if part.get_filename() is None
    }
    return fields["model"]


@pytest.mark.parametrize("path", ("/v1/audio/transcriptions", "/audio/transcriptions"))
def test_audio_transcription_enforces_key_model_access(gateway: Gateway, path: str) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/audio/transcriptions"
        assert _transcription_model(request) == "whisper-1"
        return Reply(body=json.dumps({"text": "hello world"}).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        allowed: Final = scenario.model(
            model="openai/whisper-1", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        other: Final = scenario.model(
            model="openai/whisper-1", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        key: Final = scenario.key(models=[allowed])
        denied: Final = gateway.request_multipart(
            path, {"model": other}, {"file": ("a.wav", _WAV_BYTES, _WAV_CONTENT_TYPE)}, key=key
        )
        assert denied.status_code == 403, denied.text
        error: Final = denied.json()["error"]
        assert error["type"] == "key_model_access_denied" and error["param"] == "model", denied.text
        assert wire.drain() == ()
        granted: Final = gateway.request_multipart(
            path, {"model": allowed}, {"file": ("a.wav", _WAV_BYTES, _WAV_CONTENT_TYPE)}, key=key
        )
        assert granted.status_code == 200, granted.text
        assert [(request.method, request.target) for request in wire.drain()] == [
            ("POST", "/v1/audio/transcriptions")
        ]


@pytest.mark.parametrize("path", ("/v1/audio/speech", "/audio/speech"))
def test_audio_speech_enforces_key_model_access(gateway: Gateway, path: str) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/audio/speech"
        assert request.headers["authorization"] == "Bearer synthetic-openai-key"
        assert json.loads(request.body) == {
            "model": "gpt-4o-mini-tts",
            "input": "hello",
            "voice": "alloy",
        }
        return Reply(body=_SPEECH_BYTES, content_type="audio/wav")

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        allowed: Final = scenario.model(
            model="openai/gpt-4o-mini-tts", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        other: Final = scenario.model(
            model="openai/gpt-4o-mini-tts", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        key: Final = scenario.key(models=[allowed])
        denied: Final = gateway.request(
            "POST",
            path,
            {"model": other, "input": "hello", "voice": "alloy"},
            key=key,
        )
        assert denied.status_code == 403, denied.text
        error: Final = denied.json()["error"]
        assert error["type"] == "key_model_access_denied" and error["param"] == "model", denied.text
        assert wire.drain() == ()
        granted: Final = gateway.request(
            "POST",
            path,
            {"model": allowed, "input": "hello", "voice": "alloy"},
            key=key,
        )
        assert granted.status_code == 200, granted.text
        assert granted.content == _SPEECH_BYTES
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/v1/audio/speech")]
