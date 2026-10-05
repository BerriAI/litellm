import json
from email.parser import BytesParser
from email.policy import HTTP
from typing import Final

from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server

_WAV_BYTES: Final = (
    b"RIFF\x30\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
    b"\x40\x1f\x00\x00\x40\x1f\x00\x00\x01\x00\x08\x00data\x08\x00\x00\x00"
    b"\x00\x01\x02\x03\x04\x05\x06\x07"
)


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


def test_audio_transcription_enforces_key_model_access(gateway: Gateway) -> None:
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
            "/v1/audio/transcriptions", {"model": other}, {"file": ("a.wav", _WAV_BYTES, "audio/wav")}, key=key
        )
        assert denied.status_code == 403, denied.text
        error: Final = denied.json()["error"]
        assert error["type"] == "key_model_access_denied" and error["param"] == "model", denied.text
        assert wire.drain() == ()
        granted: Final = gateway.request_multipart(
            "/v1/audio/transcriptions", {"model": allowed}, {"file": ("a.wav", _WAV_BYTES, "audio/wav")}, key=key
        )
        assert granted.status_code == 200, granted.text
        assert [(request.method, request.target) for request in wire.drain()] == [
            ("POST", "/v1/audio/transcriptions")
        ]
