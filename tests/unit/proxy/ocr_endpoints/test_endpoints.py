"""
Tests for the proxy OCR endpoint helpers that select the response format
(`x-req-format: native | litellm`) and return the provider's native payload.
"""

from typing import Final
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
from fastapi import HTTPException

from litellm.llms.base_llm.ocr.transformation import OCRPage, OCRResponse
from litellm.proxy._types import ProxyException
from litellm.proxy.ocr_endpoints.endpoints import (
    _MAX_FILE_BYTES,
    _native_response,
    _parse_ocr_request,
)

AZURE_NATIVE_OPERATION = {
    "status": "succeeded",
    "createdDateTime": "2026-07-02T00:00:00Z",
    "analyzeResult": {
        "content": "Invoice",
        "pages": [{"pageNumber": 1, "words": [{"content": "Invoice", "confidence": 0.99}]}],
        "paragraphs": [{"content": "Invoice"}],
    },
}


def _json_request(body: dict, headers: dict[str, str]) -> MagicMock:
    request = MagicMock()
    request.headers = {"content-type": "application/json", **headers}
    request.body = AsyncMock(return_value=orjson.dumps(body))
    request._form = None
    return request


class _FakeUpload:
    def __init__(self, content: bytes, oversized: bool = False) -> None:
        self._content = content
        self._oversized = oversized
        self.filename = "doc.pdf"
        self.content_type = "application/pdf"

    async def seek(self, offset: int) -> None:
        pass

    async def read(self, size: int = -1) -> bytes:
        if self._oversized:
            return b"x" * (_MAX_FILE_BYTES + 1)
        return self._content


def _multipart_request(form_fields: dict[str, object]) -> MagicMock:
    request = MagicMock()
    request.headers = {"content-type": "multipart/form-data; boundary=audit"}
    request.form = AsyncMock(return_value=form_fields)
    return request


def _ocr_response(native_payload: dict[str, object] | None) -> OCRResponse:
    response = OCRResponse(pages=[OCRPage(index=0, markdown="Invoice")], model="azure-prebuilt-layout")
    if native_payload is not None:
        response.set_provider_native_response(native_payload)
    return response


@pytest.mark.asyncio
@pytest.mark.parametrize("header_value", ["native", "NATIVE", " native "])
async def test_should_read_req_format_from_header(header_value):
    request = _json_request(
        {"model": "azure-prebuilt-layout", "document": {"type": "document_url", "document_url": "https://x/y.pdf"}},
        {"x-req-format": header_value},
    )

    assert (await _parse_ocr_request(request))["req_format"] == "native"


@pytest.mark.asyncio
async def test_should_prefer_body_req_format_over_header():
    request = _json_request(
        {
            "model": "azure-prebuilt-layout",
            "document": {"type": "document_url", "document_url": "https://x/y.pdf"},
            "req_format": "litellm",
        },
        {"x-req-format": "native"},
    )

    assert (await _parse_ocr_request(request))["req_format"] == "litellm"


@pytest.mark.asyncio
async def test_should_omit_req_format_when_header_absent():
    request = _json_request(
        {"model": "azure-prebuilt-layout", "document": {"type": "document_url", "document_url": "https://x/y.pdf"}},
        {},
    )

    assert "req_format" not in await _parse_ocr_request(request)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body_format, headers",
    [
        (None, {"x-req-format": "azure"}),
        ("azure", {}),
        ("azure", {"x-req-format": "native"}),
    ],
)
async def test_should_reject_unknown_req_format(body_format, headers):
    body = {"model": "azure-prebuilt-layout", "document": {"type": "document_url", "document_url": "https://x/y.pdf"}}
    request = _json_request(
        body if body_format is None else {**body, "req_format": body_format},
        headers,
    )

    with pytest.raises(HTTPException) as exc_info:
        await _parse_ocr_request(request)

    assert exc_info.value.status_code == 400
    assert "Invalid `req_format`" in f"{exc_info.value.detail}"


def test_should_return_native_payload_with_litellm_response_headers():
    fastapi_response = MagicMock()
    fastapi_response.headers = {"x-litellm-response-cost": "0.0015"}

    native = _native_response(_ocr_response(AZURE_NATIVE_OPERATION), fastapi_response)

    assert native is not None
    assert orjson.loads(native.body) == AZURE_NATIVE_OPERATION
    assert native.headers["x-litellm-response-cost"] == "0.0015"


def test_should_return_normalized_response_when_no_native_payload():
    assert _native_response(_ocr_response(None), MagicMock()) is None


def test_upload_builds_a_file_document_for_rust_mime_inference():
    from litellm.proxy.ocr_endpoints.endpoints import (
        _build_document_from_upload,  # pyright: ignore[reportPrivateUsage]  # tests the upload projection
    )

    document = _build_document_from_upload(b"%PDF-1.4", "receipt.pdf", None)

    assert document["type"] == "file"
    upload = document["file"]
    assert upload.read() == b"%PDF-1.4"
    assert upload.name == "receipt.pdf"
    assert "mime_type" not in document


def test_upload_keeps_the_supplied_content_type_over_filename_inference():
    from litellm.proxy.ocr_endpoints.endpoints import (
        _build_document_from_upload,  # pyright: ignore[reportPrivateUsage]  # tests the upload projection
    )

    document = _build_document_from_upload(b"data", "photo.bin", "image/png; charset=binary")

    assert document["mime_type"] == "image/png"
    assert document["file"].name == "photo.bin"


@pytest.mark.asyncio
async def test_multipart_missing_file_field_returns_400():
    with pytest.raises(ProxyException) as exc_info:
        await _parse_ocr_request(_multipart_request({"model": "azure-prebuilt-layout"}))

    assert exc_info.value.code == "400"
    assert exc_info.value.type == "invalid_request_error"
    assert exc_info.value.param == "file"
    assert "'file' field" in exc_info.value.message


@pytest.mark.asyncio
async def test_multipart_empty_file_returns_400():
    with pytest.raises(ProxyException) as exc_info:
        await _parse_ocr_request(
            _multipart_request({"model": "azure-prebuilt-layout", "file": _FakeUpload(b"")})
        )

    assert exc_info.value.code == "400"
    assert exc_info.value.type == "invalid_request_error"
    assert exc_info.value.param == "file"
    assert exc_info.value.message == "Uploaded file is empty"


@pytest.mark.asyncio
async def test_multipart_oversized_file_returns_413():
    with pytest.raises(ProxyException) as exc_info:
        await _parse_ocr_request(
            _multipart_request({"model": "azure-prebuilt-layout", "file": _FakeUpload(b"", oversized=True)})
        )

    assert exc_info.value.code == "413"
    assert exc_info.value.type == "invalid_request_error"
    assert exc_info.value.param == "file"


@pytest.mark.asyncio
async def test_json_body_empty_returns_400():
    request: Final = _json_request({}, {})
    request.body = AsyncMock(return_value=b"")

    with pytest.raises(ProxyException) as exc_info:
        await _parse_ocr_request(request)

    assert exc_info.value.code == "400"
    assert exc_info.value.type == "invalid_request_error"


@pytest.mark.asyncio
async def test_json_document_type_file_returns_400():
    request: Final = _json_request({"model": "m", "document": {"type": "file", "file": "/etc/passwd"}}, {})

    with pytest.raises(ProxyException) as exc_info:
        await _parse_ocr_request(request)

    assert exc_info.value.code == "400"
    assert exc_info.value.type == "invalid_request_error"
    assert exc_info.value.param == "document"


def test_upload_octet_stream_content_type_falls_back_to_filename_inference():
    from litellm.proxy.ocr_endpoints.endpoints import (
        _build_document_from_upload,  # pyright: ignore[reportPrivateUsage]  # tests the upload projection
    )

    document = _build_document_from_upload(b"%PDF-1.4", "receipt.pdf", "application/octet-stream")

    assert "mime_type" not in document
    assert document["file"].name == "receipt.pdf"
