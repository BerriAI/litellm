"""Audio transcription through an OpenAI-compatible JSON-configured provider against a scripted upstream (#34503).

`test_regression_*` fails on a build without the fix.
"""

import json
import mimetypes
import uuid
from email.message import Message
from email.parser import BytesParser
from email.policy import HTTP
from typing import Final

import httpx
from integration._support.client import JSON_OBJECT, Gateway
from integration._support.wire import Reply, Request, wire_server
from openai import OpenAI

_HOSTINGER_KEY: Final = "synthetic-hostinger-key"
_WAV_BYTES: Final = (
    b"RIFF\x30\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
    b"\x40\x1f\x00\x00\x40\x1f\x00\x00\x01\x00\x08\x00data\x08\x00\x00\x00"
    b"\x00\x01\x02\x03\x04\x05\x06\x07"
)
_WAV_CONTENT_TYPE: Final = mimetypes.guess_type("a.wav")[0] or "application/octet-stream"
_TRANSCRIPT: Final = "bonjour le monde"


def _multipart_parts(request: Request) -> tuple[Message, ...]:
    envelope: Final = f"content-type: {request.headers['content-type']}\r\n\r\n".encode() + request.body
    parsed: Final = BytesParser(policy=HTTP).parsebytes(envelope)
    assert parsed.is_multipart(), request.headers["content-type"]
    return tuple(parsed.iter_parts())


def _text_fields(request: Request) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (part.get_param("name", header="content-disposition"), part.get_payload(decode=True).decode())
            for part in _multipart_parts(request)
            if part.get_filename() is None
        )
    )


def _file_fields(request: Request) -> tuple[tuple[str, str, bytes], ...]:
    return tuple(
        (part.get_param("name", header="content-disposition"), part.get_filename(), part.get_payload(decode=True))
        for part in _multipart_parts(request)
        if part.get_filename() is not None
    )


def _sdk(gateway: Gateway) -> OpenAI:
    return OpenAI(
        base_url=str(gateway.client.base_url).rstrip("/") + "/v1",
        api_key=gateway.key,
        max_retries=0,
        http_client=httpx.Client(trust_env=False),
    )


def test_regression_hostinger_router_transcription_forwards_language_and_response_format(gateway: Gateway) -> None:
    """language and response_format reach an OpenAI-compatible JSON provider's transcription upstream"""
    wav: Final = _WAV_BYTES + uuid.uuid4().bytes
    with (
        wire_server(lambda _: Reply(body=json.dumps({"text": _TRANSCRIPT}).encode())) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(
            model="hostinger_router/whisper-1", api_base=f"{wire.url}/v1", api_key=_HOSTINGER_KEY
        )
        with _sdk(gateway) as sdk:
            response: Final = sdk.audio.transcriptions.with_raw_response.create(
                model=model,
                file=("a.wav", wav, _WAV_CONTENT_TYPE),
                language="fr",
                prompt="salutations",
                temperature=0.2,
                response_format="json",
            )
        requests: Final = wire.drain()
        assert [(request.method, request.target, request.headers["authorization"]) for request in requests] == [
            ("POST", "/v1/audio/transcriptions", f"Bearer {_HOSTINGER_KEY}")
        ]
        assert _text_fields(requests[0]) == (
            ("language", "fr"),
            ("model", "whisper-1"),
            ("prompt", "salutations"),
            ("response_format", "json"),
            ("temperature", "0.2"),
        )
        assert _file_fields(requests[0]) == (("file", "a.wav", wav),)
        assert (response.status_code, JSON_OBJECT.validate_json(response.content)) == (
            200,
            {"text": _TRANSCRIPT, "usage": None},
        ), response.text
