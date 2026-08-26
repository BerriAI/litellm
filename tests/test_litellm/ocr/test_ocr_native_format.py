"""
Tests for the OCR `req_format` option in the SDK request path:
providers that don't support a native response must reject it, and the Rust
bridge (which only returns the normalized shape) must not serve native requests.
"""

from unittest.mock import MagicMock

import pytest

import litellm
from litellm.ocr.main import _PreparedOCRRequest, _rust_ocr_supported

DOCUMENT = {"type": "document_url", "document_url": "https://example.com/doc.pdf"}


def _prepared(optional_params: dict[str, object]) -> _PreparedOCRRequest:
    return _PreparedOCRRequest(
        model="doc-intelligence/prebuilt-layout",
        document=dict(DOCUMENT),
        api_key="fake-key",
        api_base="https://example.cognitiveservices.azure.com",
        custom_llm_provider="azure_ai",
        extra_headers=None,
        provider_config=MagicMock(),
        optional_params=optional_params,
        litellm_params={},
        effective_timeout=60.0,
        litellm_logging_obj=MagicMock(),
    )


@pytest.mark.parametrize("optional_params", [{}, {"req_format": "litellm"}])
def test_rust_ocr_serves_default_format(optional_params):
    assert _rust_ocr_supported(_prepared(optional_params)) is True


def test_rust_ocr_skipped_for_native_format():
    assert _rust_ocr_supported(_prepared({"req_format": "native"})) is False


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
