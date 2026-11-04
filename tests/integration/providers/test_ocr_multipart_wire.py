import base64
import json
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import BaseModel

_PDF_BYTES: Final = b"%PDF-1.4\n%integration\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n"
_PNG_BYTES: Final = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
    b"\x1f\x15\xc4\x89\x00\x00\x00\rIDAT\x08\xd7c\xf8\xcf\xc0\xf0\x1f\x00\x05\x00\x01\xff"
    b"\x89\x99=\x1d\x00\x00\x00\x00IEND\xaeB`\x82"
)
_MISTRAL_KEY: Final = "synthetic-mistral-key"
_MISTRAL_PAGES: Final = (
    {"index": 0, "markdown": "# page zero", "images": [], "dimensions": {"dpi": 200, "height": 1100, "width": 850}},
    {"index": 1, "markdown": "page one", "images": [], "dimensions": {"dpi": 200, "height": 1100, "width": 850}},
)


class _Dimensions(BaseModel):
    dpi: int
    height: int
    width: int


class _Page(BaseModel):
    index: int
    markdown: str
    images: tuple[object, ...]
    dimensions: _Dimensions


class _OCRResponse(BaseModel):
    object: str
    pages: tuple[_Page, ...]


def _data_url(media_type: str, content: bytes) -> str:
    return f"data:{media_type};base64,{base64.b64encode(content).decode()}"


def _expected_body(
    upstream_model: str, document: dict[str, str], include_image_base64: bool = True
) -> dict[str, object]:
    return {"model": upstream_model, "document": document, "pages": [0, 1], "include_image_base64": include_image_base64}


def _mistral_reply(request: Request) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "pages": list(_MISTRAL_PAGES),
                "model": "mistral-ocr-latest",
                "usage_info": {"pages_processed": 2, "doc_size_bytes": len(request.body)},
                "document_annotation": None,
            }
        ).encode()
    )


def _expected_bodies(upstream_model: str) -> tuple[dict[str, object], ...]:
    pdf: Final = _expected_body(
        upstream_model, {"type": "document_url", "document_url": _data_url("application/pdf", _PDF_BYTES)}
    )
    png: Final = _expected_body(upstream_model, {"type": "image_url", "image_url": _data_url("image/png", _PNG_BYTES)})
    return (pdf, png, pdf, pdf, pdf, png)


def test_multipart_uploads_reach_mistral_as_data_urls_typed_by_the_client_content_type(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", "/v1/ocr"), request.target
        assert request.headers["authorization"] == f"Bearer {_MISTRAL_KEY}", request.headers
        assert request.headers["content-type"] == "application/json", request.headers
        body: Final = json.loads(request.body)
        assert body in _expected_bodies("mistral-ocr-latest"), body
        return _mistral_reply(request)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="mistral/mistral-ocr-latest",
            api_base=f"{wire.url}/v1",
            api_key=_MISTRAL_KEY,
            model_info={"mode": "ocr"},
        )
        fields: Final = {"model": model, "pages": "[0,1]", "include_image_base64": "true"}
        uploads: Final = (
            ("doc.pdf", _PDF_BYTES, "application/pdf"),
            ("scan.png", _PNG_BYTES, "image/png"),
            ("blob", _PDF_BYTES, "application/pdf"),
            ("upload.bin", _PDF_BYTES, "application/pdf"),
            ("doc.pdf", _PDF_BYTES, "application/octet-stream"),
            ("scan.png", _PNG_BYTES, "application/octet-stream"),
        )
        for path, upload in ((path, upload) for path in ("/v1/ocr", "/ocr") for upload in uploads):
            response = gateway.request_multipart(path, fields, {"file": upload})
            assert response.status_code == 200, f"{path}: {response.text}"
            payload = _OCRResponse.model_validate_json(response.content)
            assert payload == _OCRResponse(object="ocr", pages=tuple(_Page(**page) for page in _MISTRAL_PAGES)), (
                response.text
            )
        received: Final = wire.drain()
    assert [(request.method, request.target) for request in received] == [("POST", "/v1/ocr")] * (2 * len(uploads))
    assert tuple(json.loads(request.body) for request in received) == _expected_bodies("mistral-ocr-latest") * 2


def test_python_requests_boolean_spelling_reaches_mistral_as_a_boolean(gateway: Gateway) -> None:
    pytest.skip(
        "BUG: multipart include_image_base64='True' or 'False' (Python requests' str(bool)) reaches Mistral "
        "as the string 'True' or 'False' instead of a JSON boolean"
    )

    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", "/v1/ocr"), request.target
        return _mistral_reply(request)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="mistral/mistral-ocr-latest",
            api_base=f"{wire.url}/v1",
            api_key=_MISTRAL_KEY,
            model_info={"mode": "ocr"},
        )
        for spelling in ("True", "False"):
            response = gateway.request_multipart(
                "/v1/ocr",
                {"model": model, "pages": "[0,1]", "include_image_base64": spelling},
                {"file": ("doc.pdf", _PDF_BYTES, "application/pdf")},
            )
            assert response.status_code == 200, f"{spelling}: {response.text}"
        received: Final = wire.drain()
    pdf: Final = {"type": "document_url", "document_url": _data_url("application/pdf", _PDF_BYTES)}
    assert tuple(json.loads(request.body) for request in received) == (
        _expected_body("mistral-ocr-latest", pdf, include_image_base64=True),
        _expected_body("mistral-ocr-latest", pdf, include_image_base64=False),
    ), [request.body for request in received]
