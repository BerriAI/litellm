import json
import struct
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from email.message import Message
from email.parser import BytesParser
from email.policy import HTTP
from typing import Final, cast

import httpx
import openai
import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import BaseModel

_PNG_BYTES: Final = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
    b"\x1f\x15\xc4\x89\x00\x00\x00\rIDAT\x08\xd7c\xf8\xcf\xc0\xf0\x1f\x00\x05\x00\x01\xff"
    b"\x89\x99=\x1d\x00\x00\x00\x00IEND\xaeB`\x82"
)
_PROMPT: Final = "turn the red circle green"
_EDITED_IMAGE_B64: Final = "aW50ZWdyYXRpb24tZWRpdGVkLWltYWdl"


def _png_chunk(chunk_type: bytes, chunk_data: bytes) -> bytes:
    checksum: Final = zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(chunk_data))
        + chunk_type
        + chunk_data
        + struct.pack(">I", checksum)
    )


def _png_from_rgba_pixel(pixel: tuple[int, int, int, int]) -> bytes:
    header: Final = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    compressed_pixel: Final = zlib.compress(b"\x00" + bytes(pixel))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", compressed_pixel)
        + _png_chunk(b"IEND", b"")
    )


_SECOND_PNG: Final = _png_from_rgba_pixel((0, 0, 255, 255))
_MASK_PNG: Final = _png_from_rgba_pixel((0, 0, 0, 0))


class _Image(BaseModel):
    b64_json: str


class _ImageResponse(BaseModel):
    data: tuple[_Image, ...]


def _multipart_parts(request: Request) -> tuple[Message, ...]:
    envelope: Final = f"content-type: {request.headers['content-type']}\r\n\r\n".encode() + request.body
    parsed: Final = BytesParser(policy=HTTP).parsebytes(envelope)
    assert parsed.is_multipart(), request.headers["content-type"]
    return tuple(parsed.iter_parts())


@dataclass(frozen=True, slots=True)
class _FilePart:
    name: str
    filename: str
    content_type: str
    content: bytes


def _text_parts(parts: tuple[Message, ...]) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (
                cast(str, part.get_param("name", header="content-disposition")),
                cast(bytes, part.get_payload(decode=True)).decode(),
            )
            for part in parts
            if part.get_filename() is None
        )
    )


def _file_parts(parts: tuple[Message, ...]) -> tuple[_FilePart, ...]:
    return tuple(
        _FilePart(
            name=cast(str, part.get_param("name", header="content-disposition")),
            filename=filename,
            content_type=part.get_content_type(),
            content=cast(bytes, part.get_payload(decode=True)),
        )
        for part in parts
        if (filename := part.get_filename()) is not None
    )


def _edit(
    gateway: Gateway,
    fields: Mapping[str, str],
    files: Sequence[tuple[str, tuple[str, bytes, str]]],
    path: str = "/v1/images/edits",
) -> httpx.Response:
    return gateway.client.post(
        path,
        data=dict(fields),
        files=list(files),
        headers={"Authorization": f"Bearer {gateway.key}"},
    )


@pytest.mark.covers("other.provider_wire.openai.image_edit_forwards_provider_specific_form_fields")
def test_openai_compatible_image_edit_forwards_seed_form_field_to_backend(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/images/edits"
        assert request.headers["authorization"] == "Bearer synthetic-openai-key"
        assert request.headers["content-type"].startswith("multipart/form-data; boundary=")
        parts: Final = _multipart_parts(request)
        assert _text_parts(parts) == (("model", "gpt-image-1"), ("prompt", _PROMPT), ("seed", "42"))
        assert _file_parts(parts) == (_FilePart("image[]", "image.png", "image/png", _PNG_BYTES),)
        return Reply(body=json.dumps({"created": 1700000000, "data": [{"b64_json": _EDITED_IMAGE_B64}]}).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/gpt-image-1", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        response: Final = gateway.request_multipart(
            "/v1/images/edits",
            {"model": model, "prompt": _PROMPT, "seed": "42"},
            {"image": ("red_circle.png", _PNG_BYTES, "image/png")},
        )
        assert response.status_code == 200, response.text
        payload: Final = _ImageResponse.model_validate_json(response.content)
        assert [image.b64_json for image in payload.data] == [_EDITED_IMAGE_B64], response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/v1/images/edits")]


@pytest.mark.parametrize(
    "path_template",
    ("/v1/images/edits", "/images/edits", "/openai/deployments/{model}/images/edits"),
    ids=("v1", "unversioned", "azure_deployment"),
)
def test_image_edit_bracketed_image_field_reaches_openai_as_one_file_part(
    gateway: Gateway, path_template: str
) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/images/edits"
        assert request.headers["authorization"] == "Bearer synthetic-openai-key"
        assert request.headers["content-type"].startswith("multipart/form-data; boundary=")
        parts: Final = _multipart_parts(request)
        assert _text_parts(parts) == (("model", "gpt-image-1"), ("prompt", _PROMPT), ("seed", "42"))
        assert _file_parts(parts) == (_FilePart("image[]", "image.png", "image/png", _PNG_BYTES),)
        return Reply(body=json.dumps({"created": 1700000000, "data": [{"b64_json": _EDITED_IMAGE_B64}]}).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/gpt-image-1", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        response: Final = _edit(
            gateway,
            {"model": model, "prompt": _PROMPT, "seed": "42"},
            (("image[]", ("red_circle.png", _PNG_BYTES, "image/png")),),
            path_template.format(model=model),
        )
        assert response.status_code == 200, response.text
        payload: Final = _ImageResponse.model_validate_json(response.content)
        assert [image.b64_json for image in payload.data] == [_EDITED_IMAGE_B64], response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/v1/images/edits")]


def test_image_edit_repeated_bracketed_image_parts_reach_openai_in_upload_order(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/images/edits"
        assert request.headers["authorization"] == "Bearer synthetic-openai-key"
        assert request.headers["content-type"].startswith("multipart/form-data; boundary=")
        parts: Final = _multipart_parts(request)
        assert _text_parts(parts) == (("model", "gpt-image-1"), ("n", "2"), ("prompt", _PROMPT))
        assert _file_parts(parts) == (
            _FilePart("image[]", "image.png", "image/png", _PNG_BYTES),
            _FilePart("image[]", "image.png", "image/png", _SECOND_PNG),
        )
        return Reply(body=json.dumps({"created": 1700000000, "data": [{"b64_json": _EDITED_IMAGE_B64}]}).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/gpt-image-1", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        response: Final = _edit(
            gateway,
            {"model": model, "prompt": _PROMPT, "n": "2"},
            (
                ("image[]", ("first.png", _PNG_BYTES, "image/png")),
                ("image[]", ("second.png", _SECOND_PNG, "image/png")),
            ),
        )
        assert response.status_code == 200, response.text
        payload: Final = _ImageResponse.model_validate_json(response.content)
        assert [image.b64_json for image in payload.data] == [_EDITED_IMAGE_B64], response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/v1/images/edits")]


def test_image_edit_bracketed_mask_reaches_openai_as_mask_file_part(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/images/edits"
        assert request.headers["authorization"] == "Bearer synthetic-openai-key"
        assert request.headers["content-type"].startswith("multipart/form-data; boundary=")
        parts: Final = _multipart_parts(request)
        assert _text_parts(parts) == (("model", "gpt-image-1"), ("prompt", _PROMPT))
        assert _file_parts(parts) == (
            _FilePart("image[]", "image.png", "image/png", _PNG_BYTES),
            _FilePart("mask", "mask.png", "image/png", _MASK_PNG),
        )
        return Reply(body=json.dumps({"created": 1700000000, "data": [{"b64_json": _EDITED_IMAGE_B64}]}).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/gpt-image-1", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        response: Final = _edit(
            gateway,
            {"model": model, "prompt": _PROMPT},
            (
                ("image[]", ("red_circle.png", _PNG_BYTES, "image/png")),
                ("mask[]", ("mask.png", _MASK_PNG, "image/png")),
            ),
        )
        assert response.status_code == 200, response.text
        payload: Final = _ImageResponse.model_validate_json(response.content)
        assert [image.b64_json for image in payload.data] == [_EDITED_IMAGE_B64], response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/v1/images/edits")]


def test_image_edit_repeated_bracketed_mask_parts_reach_openai_as_first_mask_only(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/images/edits"
        assert request.headers["authorization"] == "Bearer synthetic-openai-key"
        assert request.headers["content-type"].startswith("multipart/form-data; boundary=")
        parts: Final = _multipart_parts(request)
        assert _text_parts(parts) == (("model", "gpt-image-1"), ("prompt", _PROMPT))
        assert _file_parts(parts) == (
            _FilePart("image[]", "image.png", "image/png", _PNG_BYTES),
            _FilePart("mask", "mask.png", "image/png", _MASK_PNG),
        )
        return Reply(body=json.dumps({"created": 1700000000, "data": [{"b64_json": _EDITED_IMAGE_B64}]}).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/gpt-image-1", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        response: Final = _edit(
            gateway,
            {"model": model, "prompt": _PROMPT},
            (
                ("image[]", ("red_circle.png", _PNG_BYTES, "image/png")),
                ("mask[]", ("first-mask.png", _MASK_PNG, "image/png")),
                ("mask[]", ("second-mask.png", _SECOND_PNG, "image/png")),
            ),
        )
        assert response.status_code == 200, response.text
        payload: Final = _ImageResponse.model_validate_json(response.content)
        assert [image.b64_json for image in payload.data] == [_EDITED_IMAGE_B64], response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/v1/images/edits")]


def test_image_edit_plain_mask_reaches_openai_as_mask_file_part(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/images/edits"
        assert request.headers["authorization"] == "Bearer synthetic-openai-key"
        assert request.headers["content-type"].startswith("multipart/form-data; boundary=")
        parts: Final = _multipart_parts(request)
        assert _text_parts(parts) == (("model", "gpt-image-1"), ("prompt", _PROMPT))
        assert _file_parts(parts) == (
            _FilePart("image[]", "image.png", "image/png", _PNG_BYTES),
            _FilePart("mask", "mask.png", "image/png", _MASK_PNG),
        )
        return Reply(body=json.dumps({"created": 1700000000, "data": [{"b64_json": _EDITED_IMAGE_B64}]}).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/gpt-image-1", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        response: Final = _edit(
            gateway,
            {"model": model, "prompt": _PROMPT},
            (
                ("image", ("red_circle.png", _PNG_BYTES, "image/png")),
                ("mask", ("mask.png", _MASK_PNG, "image/png")),
            ),
        )
        assert response.status_code == 200, response.text
        payload: Final = _ImageResponse.model_validate_json(response.content)
        assert [image.b64_json for image in payload.data] == [_EDITED_IMAGE_B64], response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/v1/images/edits")]


def test_image_edit_repeated_plain_image_parts_reach_openai_as_two_file_parts(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/images/edits"
        assert request.headers["authorization"] == "Bearer synthetic-openai-key"
        assert request.headers["content-type"].startswith("multipart/form-data; boundary=")
        parts: Final = _multipart_parts(request)
        assert _text_parts(parts) == (("model", "gpt-image-1"), ("prompt", _PROMPT))
        assert _file_parts(parts) == (
            _FilePart("image[]", "image.png", "image/png", _PNG_BYTES),
            _FilePart("image[]", "image.png", "image/png", _SECOND_PNG),
        )
        return Reply(body=json.dumps({"created": 1700000000, "data": [{"b64_json": _EDITED_IMAGE_B64}]}).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/gpt-image-1", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        response: Final = _edit(
            gateway,
            {"model": model, "prompt": _PROMPT},
            (
                ("image", ("first.png", _PNG_BYTES, "image/png")),
                ("image", ("second.png", _SECOND_PNG, "image/png")),
            ),
        )
        assert response.status_code == 200, response.text
        payload: Final = _ImageResponse.model_validate_json(response.content)
        assert [image.b64_json for image in payload.data] == [_EDITED_IMAGE_B64], response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/v1/images/edits")]


def test_openai_sdk_image_edit_with_image_list_and_mask_reaches_openai_as_file_parts(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/images/edits"
        assert request.headers["authorization"] == "Bearer synthetic-openai-key"
        assert request.headers["content-type"].startswith("multipart/form-data; boundary=")
        parts: Final = _multipart_parts(request)
        assert _text_parts(parts) == (("model", "gpt-image-1"), ("n", "2"), ("prompt", _PROMPT))
        assert _file_parts(parts) == (
            _FilePart("image[]", "image.png", "image/png", _PNG_BYTES),
            _FilePart("image[]", "image.png", "image/png", _SECOND_PNG),
            _FilePart("mask", "mask.png", "image/png", _MASK_PNG),
        )
        return Reply(body=json.dumps({"created": 1700000000, "data": [{"b64_json": _EDITED_IMAGE_B64}]}).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/gpt-image-1", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        client: Final = openai.OpenAI(
            base_url=str(gateway.client.base_url) + "/v1",
            api_key=gateway.key,
            max_retries=0,
        )
        result: Final = client.images.edit(
            model=model,
            image=[
                ("first.png", _PNG_BYTES, "image/png"),
                ("second.png", _SECOND_PNG, "image/png"),
            ],
            mask=("mask.png", _MASK_PNG, "image/png"),
            prompt=_PROMPT,
            n=2,
        )
        assert [image.b64_json for image in result.data] == [_EDITED_IMAGE_B64]
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/v1/images/edits")]


@pytest.mark.parametrize(
    ("files", "detail"),
    (
        pytest.param(
            (
                ("image", ("red_circle.png", _PNG_BYTES, "image/png")),
                ("image[]", ("second.png", _SECOND_PNG, "image/png")),
            ),
            "Cannot specify both 'image' and 'image[]'",
            id="image",
        ),
        pytest.param(
            (
                ("image", ("red_circle.png", _PNG_BYTES, "image/png")),
                ("mask", ("mask.png", _MASK_PNG, "image/png")),
                ("mask[]", ("mask-array.png", _MASK_PNG, "image/png")),
            ),
            "Cannot specify both 'mask' and 'mask[]'",
            id="mask",
        ),
    ),
)
def test_image_edit_rejects_plain_and_bracketed_spelling_together_without_calling_upstream(
    gateway: Gateway,
    files: Sequence[tuple[str, tuple[str, bytes, str]]],
    detail: str,
) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/images/edits"
        assert request.headers["authorization"] == "Bearer synthetic-openai-key"
        assert request.headers["content-type"].startswith("multipart/form-data; boundary=")
        return Reply(body=json.dumps({"created": 1700000000, "data": [{"b64_json": _EDITED_IMAGE_B64}]}).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/gpt-image-1", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        response: Final = _edit(
            gateway,
            {"model": model, "prompt": _PROMPT},
            files,
        )
        assert response.status_code == 422, response.text
        assert response.json() == {"detail": detail}, response.text
        assert wire.drain() == ()
