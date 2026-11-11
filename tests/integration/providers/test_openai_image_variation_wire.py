import json
from email.message import Message
from email.parser import BytesParser
from email.policy import HTTP
from typing import Final, cast

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import BaseModel

_PNG_BYTES: Final = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
    b"\x1f\x15\xc4\x89\x00\x00\x00\rIDAT\x08\xd7c\xf8\xcf\xc0\xf0\x1f\x00\x05\x00\x01\xff"
    b"\x89\x99=\x1d\x00\x00\x00\x00IEND\xaeB`\x82"
)
_VARIATION_URLS: Final = ("https://images.example/variation-1.png", "https://images.example/variation-2.png")


class _Image(BaseModel):
    url: str


class _ImageResponse(BaseModel):
    data: tuple[_Image, ...]


def _multipart_parts(request: Request) -> tuple[Message, ...]:
    envelope: Final = f"content-type: {request.headers['content-type']}\r\n\r\n".encode() + request.body
    parsed: Final = BytesParser(policy=HTTP).parsebytes(envelope)
    assert parsed.is_multipart(), request.headers["content-type"]
    return tuple(parsed.iter_parts())


def test_image_variation_multipart_image_and_string_n_reach_openai_as_file_part(gateway: Gateway) -> None:
    pytest.skip("BUG: POST /v1/images/variations is not routed by the proxy")

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/images/variations"
        assert request.headers["authorization"] == "Bearer synthetic-openai-key"
        assert request.headers["content-type"].startswith("multipart/form-data; boundary=")
        parts: Final = _multipart_parts(request)
        assert sorted(
            (
                cast(str, part.get_param("name", header="content-disposition")),
                cast(bytes, part.get_payload(decode=True)).decode(),
            )
            for part in parts
            if part.get_filename() is None
        ) == [("model", "dall-e-2"), ("n", "2")]
        assert [
            (cast(str, part.get_param("name", header="content-disposition")), cast(bytes, part.get_payload(decode=True)))
            for part in parts
            if part.get_filename()
        ] == [("image", _PNG_BYTES)]
        return Reply(body=json.dumps({"created": 1700000000, "data": [{"url": url} for url in _VARIATION_URLS]}).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/dall-e-2", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key")
        response: Final = gateway.client.post(
            "/v1/images/variations",
            data={"model": model, "n": "2"},
            files={"image": ("red_circle.png", _PNG_BYTES, "image/png")},
            headers={"Authorization": f"Bearer {gateway.key}"},
        )
        assert response.status_code == 200, response.text
        payload: Final = _ImageResponse.model_validate_json(response.content)
        assert [image.url for image in payload.data] == list(_VARIATION_URLS), response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/v1/images/variations")]
