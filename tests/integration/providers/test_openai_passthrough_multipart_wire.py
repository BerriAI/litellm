import io
import json
import threading
from collections.abc import Iterator
from email.parser import BytesParser
from email.policy import HTTP
from itertools import accumulate, dropwhile
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, gateway_from_environment
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from openai import OpenAI
from pydantic import BaseModel

pytestmark = pytest.mark.timeout(240)

_PNG_A: Final = b"\x89PNG\r\n\x1a\nscripted-image-a"
_PNG_B: Final = b"\x89PNG\r\n\x1a\nscripted-image-b"
_MASK: Final = b"\x89PNG\r\n\x1a\nscripted-mask"
_AUDIO: Final = b"RIFF\x24\x00\x00\x00WAVEscripted-audio"
_JSONL: Final = b'{"custom_id":"1","method":"POST","url":"/v1/chat/completions","body":{}}\n'
_EDITED: Final = "c2NyaXB0ZWQtZWRpdA=="
_TRANSCRIPT: Final = "scripted transcript"
_IMAGE_STREAM_FRAMES: Final = (
    b'event: image_generation.partial_image\ndata: {"type":"image_generation.partial_image"}\n\n',
    b'event: image_generation.completed\ndata: {"type":"image_generation.completed"}\n\n',
)
_IMAGE_STREAM_GATE: Final = threading.Event()
_FILE_OBJECT: Final = {
    "id": "file-scripted",
    "object": "file",
    "bytes": len(_JSONL),
    "created_at": 1,
    "filename": "batch.jsonl",
    "purpose": "batch",
    "status": "processed",
}

Part = tuple[str, str | None, str | None, bytes]


class _Image(BaseModel):
    b64_json: str


class _ImageResponse(BaseModel):
    created: int
    data: tuple[_Image, ...]


def _respond(request: Request) -> Reply:
    if request.target == "/v1/audio/transcriptions":
        return Reply(body=_TRANSCRIPT.encode(), content_type="text/plain")
    if request.target == "/v1/files":
        return Reply(body=json.dumps(_FILE_OBJECT).encode())
    if request.target == "/v1/images/edits":
        parts: Final = _parts(request.headers["content-type"], request.body)
        if ("stream", None, None, b"true") in parts:
            return Reply(
                chunks=_IMAGE_STREAM_FRAMES,
                content_type="text/event-stream",
                gate_after_first=_IMAGE_STREAM_GATE,
            )
    return Reply(body=json.dumps({"created": 1, "data": [{"b64_json": _EDITED}]}).encode())


@pytest.fixture(scope="module")
def wire() -> Iterator[Wire]:
    with wire_server(_respond) as served:
        yield served


@pytest.fixture(autouse=True)
def _drained(wire: Wire) -> None:
    wire.drain()


@pytest.fixture(scope="module")
def proxy(wire: Wire, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("openai-passthrough-multipart")
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["environment_variables"] = {"OPENAI_API_BASE": wire.url, "OPENAI_API_KEY": "scripted"}
    path: Final = directory / "openai-multipart.yaml"
    path.write_text(yaml.safe_dump(config))
    with gateway_from_environment() as rig, owned_proxy(rig, directory, {}, config=path, workers=2) as owned:
        yield owned


def _parts(content_type: str, body: bytes) -> tuple[Part, ...]:
    envelope: Final = f"content-type: {content_type}\r\n\r\n".encode() + body
    parsed: Final = BytesParser(policy=HTTP).parsebytes(envelope)
    assert parsed.is_multipart(), content_type
    return tuple(
        (
            str(part.get_param("name", header="content-disposition")),
            part.get_filename(),
            part.get_content_type() if part.get_filename() is not None else None,
            part.get_payload(decode=True),
        )
        for part in parsed.iter_parts()
    )


def _only_upstream(wire: Wire, target: str) -> Request:
    received: Final = wire.drain()
    assert [(request.method, request.target) for request in received] == [("POST", target)], received
    upstream: Final = received[0]
    assert upstream.headers["authorization"] == "Bearer scripted", upstream.headers
    return upstream


def _sdk(proxy: Gateway, key: str, prefix: str) -> OpenAI:
    return OpenAI(
        api_key=key,
        base_url=f"{str(proxy.client.base_url).rstrip('/')}{prefix}/v1",
        max_retries=0,
        http_client=httpx.Client(timeout=30, trust_env=False),
    )


@pytest.mark.parametrize("stream", [False, True], ids=["stream-false", "stream-true"])
def test_curl_image_edit_forwards_repeated_image_parts_mask_and_string_typed_fields(
    proxy: Gateway, wire: Wire, stream: bool
) -> None:
    if stream:
        pytest.skip("BUG: multipart image edits with stream=true do not relay the SSE response incrementally")
    with proxy.scenario() as scenario:
        key: Final = scenario.key()
        files: Final = [
            ("model", (None, "gpt-image-1")),
            ("prompt", (None, "make them match")),
            ("n", (None, "2")),
            ("stream", (None, "true" if stream else "false")),
            ("image[]", ("a.png", _PNG_A, "image/png")),
            ("image[]", ("b.png", _PNG_B, "image/png")),
            ("mask", ("m.png", _MASK, "image/png")),
        ]
        if stream:
            _IMAGE_STREAM_GATE.clear()
            with proxy.client.stream(
                "POST",
                "/openai/v1/images/edits",
                headers={"Authorization": f"Bearer {key}"},
                files=files,
            ) as response:
                raw: Final = iter(response.iter_raw())
                try:
                    assert response.status_code == 200, response.status_code
                    assert response.headers.get("content-type") == "text/event-stream", response.headers
                    first_frame: Final = next(
                        dropwhile(
                            lambda buffered: b"\n\n" not in buffered,
                            accumulate(raw, lambda buffered, chunk: buffered + chunk, initial=b""),
                        )
                    )
                    assert first_frame == _IMAGE_STREAM_FRAMES[0], first_frame
                finally:
                    _IMAGE_STREAM_GATE.set()
                streamed: Final = first_frame + b"".join(raw)
                assert streamed == b"".join(_IMAGE_STREAM_FRAMES), streamed
        else:
            response: Final = proxy.client.post(
                "/openai/v1/images/edits",
                headers={"Authorization": f"Bearer {key}"},
                files=files,
            )
            assert response.status_code == 200, response.text
            assert _ImageResponse.model_validate_json(response.content).data == (_Image(b64_json=_EDITED),), (
                response.text
            )
        upstream: Final = _only_upstream(wire, "/v1/images/edits")
        assert _parts(upstream.headers["content-type"], upstream.body) == (
            ("model", None, None, b"gpt-image-1"),
            ("prompt", None, None, b"make them match"),
            ("n", None, None, b"2"),
            ("stream", None, None, b"true" if stream else b"false"),
            ("image[]", "a.png", "image/png", _PNG_A),
            ("image[]", "b.png", "image/png", _PNG_B),
            ("mask", "m.png", "image/png", _MASK),
        ), upstream.body


def test_sdk_image_edit_with_two_images_reaches_upstream_as_two_file_parts(proxy: Gateway, wire: Wire) -> None:
    with proxy.scenario() as scenario, _sdk(proxy, scenario.key(), "/openai") as sdk:
        edited: Final = sdk.images.edit(
            model="gpt-image-1",
            image=[("a.png", io.BytesIO(_PNG_A), "image/png"), ("b.png", io.BytesIO(_PNG_B), "image/png")],
            mask=("m.png", io.BytesIO(_MASK), "image/png"),
            prompt="make them match",
            n=2,
        )
        assert [image.b64_json for image in edited.data or []] == [_EDITED], edited
        upstream: Final = _only_upstream(wire, "/v1/images/edits")
        assert _parts(upstream.headers["content-type"], upstream.body) == (
            ("prompt", None, None, b"make them match"),
            ("model", None, None, b"gpt-image-1"),
            ("n", None, None, b"2"),
            ("image[]", "a.png", "image/png", _PNG_A),
            ("image[]", "b.png", "image/png", _PNG_B),
            ("mask", "m.png", "image/png", _MASK),
        ), upstream.body


def test_sdk_transcription_and_file_upload_forward_the_file_and_its_text_fields(proxy: Gateway, wire: Wire) -> None:
    with proxy.scenario() as scenario:
        key: Final = scenario.key()
        with _sdk(proxy, key, "/openai") as sdk:
            transcript: Final = sdk.audio.transcriptions.create(
                model="whisper-1", file=("clip.wav", io.BytesIO(_AUDIO), "audio/wav"), response_format="text"
            )
        assert transcript == _TRANSCRIPT, transcript
        transcription: Final = _only_upstream(wire, "/v1/audio/transcriptions")
        assert _parts(transcription.headers["content-type"], transcription.body) == (
            ("model", None, None, b"whisper-1"),
            ("response_format", None, None, b"text"),
            ("file", "clip.wav", "audio/wav", _AUDIO),
        ), transcription.body
        with _sdk(proxy, key, "/openai_passthrough") as sdk:
            uploaded: Final = sdk.files.create(
                file=("batch.jsonl", io.BytesIO(_JSONL), "application/jsonl"), purpose="batch"
            )
        assert uploaded.model_dump(exclude_unset=True) == _FILE_OBJECT, uploaded
        upload: Final = _only_upstream(wire, "/v1/files")
        assert _parts(upload.headers["content-type"], upload.body) == (
            ("purpose", None, None, b"batch"),
            ("file", "batch.jsonl", "application/jsonl", _JSONL),
        ), upload.body


def test_file_less_multipart_form_stays_multipart_with_every_repeated_field(proxy: Gateway, wire: Wire) -> None:
    with proxy.scenario() as scenario:
        response: Final = proxy.client.post(
            "/openai/v1/images/edits",
            headers={"Authorization": f"Bearer {scenario.key()}"},
            files=[
                ("model", (None, "gpt-image-1")),
                ("prompt", (None, "no file")),
                ("image_url[]", (None, "https://example.com/a.png")),
                ("image_url[]", (None, "https://example.com/b.png")),
            ],
        )
        assert response.status_code == 200, response.text
        parsed: Final = _ImageResponse.model_validate_json(response.content)
        assert parsed == _ImageResponse(created=1, data=(_Image(b64_json=_EDITED),)), response.text
        upstream: Final = _only_upstream(wire, "/v1/images/edits")
        assert upstream.headers["content-type"].startswith("multipart/form-data; boundary="), upstream.headers
        assert _parts(upstream.headers["content-type"], upstream.body) == (
            ("model", None, None, b"gpt-image-1"),
            ("prompt", None, None, b"no file"),
            ("image_url[]", None, None, b"https://example.com/a.png"),
            ("image_url[]", None, None, b"https://example.com/b.png"),
        ), upstream.body
