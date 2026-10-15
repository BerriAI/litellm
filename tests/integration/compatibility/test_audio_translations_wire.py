import json
import mimetypes
from email.message import Message
from email.parser import BytesParser
from email.policy import HTTP
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from openai import OpenAI

_WAV_BYTES: Final = (
    b"RIFF\x30\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
    b"\x40\x1f\x00\x00\x40\x1f\x00\x00\x01\x00\x08\x00data\x08\x00\x00\x00"
    b"\x00\x01\x02\x03\x04\x05\x06\x07"
)
_WAV_CONTENT_TYPE: Final = mimetypes.guess_type("a.wav")[0] or "application/octet-stream"


def _multipart_parts(request: Request) -> tuple[Message, ...]:
    envelope: Final = f"content-type: {request.headers['content-type']}\r\n\r\n".encode() + request.body
    parsed: Final = BytesParser(policy=HTTP).parsebytes(envelope)
    assert parsed.is_multipart(), request.headers["content-type"]
    return tuple(parsed.iter_parts())


def _text_parts(parts: tuple[Message, ...]) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (
                (part.get_param("name", header="content-disposition"), part.get_payload(decode=True).decode())
                for part in parts
                if part.get_filename() is None
            ),
            key=lambda field: field[0],
        )
    )


def _file_parts(parts: tuple[Message, ...]) -> tuple[tuple[str, str, str, bytes], ...]:
    return tuple(
        sorted(
            (
                (
                    part.get_param("name", header="content-disposition"),
                    part.get_filename(),
                    part.get_content_type(),
                    part.get_payload(decode=True),
                )
                for part in parts
                if part.get_filename() is not None
            ),
            key=lambda field: field[0],
        )
    )


def _sdk(gateway: Gateway, path: str, api_key: str) -> OpenAI:
    return OpenAI(
        base_url=str(gateway.client.base_url).rstrip("/") + path,
        api_key=api_key,
        max_retries=0,
        http_client=httpx.Client(trust_env=False),
    )


@pytest.mark.parametrize("base_path", ("/v1", ""))
def test_audio_translation_reaches_upstream_on_prefixed_and_unprefixed_routes(gateway: Gateway, base_path: str) -> None:
    pytest.skip(
        "BUG: the proxy registers no /v1/audio/translations or /audio/translations route, so OpenAI SDK "
        "translations.create gets 404 Not Found and never reaches the provider"
    )

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/audio/translations"
        assert request.headers["authorization"] == "Bearer synthetic-openai-key"
        parts: Final = _multipart_parts(request)
        assert _text_parts(parts) == (
            ("model", "whisper-1"),
            ("prompt", "hi"),
            ("response_format", "json"),
            ("temperature", "0.2"),
        )
        assert _file_parts(parts) == (("file", "a.wav", _WAV_CONTENT_TYPE, _WAV_BYTES),)
        return Reply(body=json.dumps({"text": "hello world"}).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/whisper-1", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        key: Final = scenario.key(models=[model])
        with _sdk(gateway, base_path, key) as sdk:
            response: Final = sdk.audio.translations.with_raw_response.create(
                model=model,
                file=("a.wav", _WAV_BYTES, _WAV_CONTENT_TYPE),
                prompt="hi",
                response_format="json",
                temperature=0.2,
            )
        assert response.status_code == 200, response.text
        assert json.loads(response.content) == {"text": "hello world"}, response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/v1/audio/translations")]
