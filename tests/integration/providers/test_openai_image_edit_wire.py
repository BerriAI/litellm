import json
from email.message import Message
from email.parser import BytesParser
from email.policy import HTTP
from typing import Final

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


class _Image(BaseModel):
    b64_json: str


class _ImageResponse(BaseModel):
    data: tuple[_Image, ...]


def _multipart_parts(request: Request) -> tuple[Message, ...]:
    envelope: Final = f"content-type: {request.headers['content-type']}\r\n\r\n".encode() + request.body
    parsed: Final = BytesParser(policy=HTTP).parsebytes(envelope)
    assert parsed.is_multipart(), request.headers["content-type"]
    return tuple(parsed.iter_parts())


def _text_fields(parts: tuple[Message, ...]) -> dict[str, str]:
    return {
        part.get_param("name", header="content-disposition"): part.get_payload(decode=True).decode()
        for part in parts
        if part.get_filename() is None
    }


def _file_fields(parts: tuple[Message, ...]) -> dict[str, bytes]:
    return {
        part.get_param("name", header="content-disposition"): part.get_payload(decode=True)
        for part in parts
        if part.get_filename() is not None
    }


@pytest.mark.covers("other.provider_wire.openai.image_edit_forwards_provider_specific_form_fields")
def test_openai_compatible_image_edit_forwards_seed_form_field_to_backend(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/images/edits"
        assert request.headers["authorization"] == "Bearer synthetic-openai-key"
        parts: Final = _multipart_parts(request)
        assert _text_fields(parts) == {"model": "gpt-image-1", "prompt": _PROMPT, "seed": "42"}
        assert _file_fields(parts) == {"image[]": _PNG_BYTES}
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
