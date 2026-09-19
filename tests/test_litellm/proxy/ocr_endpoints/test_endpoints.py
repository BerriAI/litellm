"""
Tests for the proxy OCR endpoint helpers that select the response format
(`x-req-format: native | litellm`) and return the provider's native payload.
"""

from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
from fastapi import HTTPException

from litellm.llms.base_llm.ocr.transformation import OCRPage, OCRResponse
from litellm.proxy.ocr_endpoints.endpoints import _native_response, _parse_ocr_request

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
