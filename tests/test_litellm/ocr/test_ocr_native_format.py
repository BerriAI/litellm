"""
Tests for the OCR `req_format` option in the SDK request path.
"""

import pytest

import litellm

DOCUMENT = {"type": "document_url", "document_url": "https://example.com/doc.pdf"}


@pytest.mark.asyncio
async def test_native_format_rejected_for_provider_without_support_as_bad_request():
    with pytest.raises(litellm.BadRequestError, match="not supported for provider") as exc_info:
        await litellm.aocr(
            model="mistral/mistral-ocr-latest",
            document=DOCUMENT,
            api_key="fake-key",
            req_format="native",
        )

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_unknown_format_rejected_for_provider_without_support_as_bad_request():
    with pytest.raises(litellm.BadRequestError, match="Invalid `req_format`") as exc_info:
        await litellm.aocr(
            model="mistral/mistral-ocr-latest",
            document=DOCUMENT,
            api_key="fake-key",
            req_format="raw",
        )

    assert exc_info.value.status_code == 400
