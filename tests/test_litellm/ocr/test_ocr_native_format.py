"""
Tests for the OCR `req_format` option in the SDK request path:
providers that don't support a native response must reject it, and the Rust
bridge (which only returns the normalized shape) must not serve native requests.
"""

import pytest

import litellm
from litellm.rust_bridge import ocr as rust_ocr_bridge

DOCUMENT = {"type": "document_url", "document_url": "https://example.com/doc.pdf"}


def test_native_decline_wrapper_uses_runtime_result():
    rust_ocr_bridge.use_litellm_rust(
        True,
        ocr_decline=lambda model, custom_llm_provider, optional_params: (
            "native OCR response format requires Python" if optional_params.get("req_format") == "native" else None
        ),
    )
    try:
        assert (
            rust_ocr_bridge.ocr_decline(
                model="azure_ai/doc-intelligence/prebuilt-layout",
                custom_llm_provider=None,
                optional_params={"req_format": "native"},
            )
            == "native OCR response format requires Python"
        )
    finally:
        rust_ocr_bridge.use_litellm_rust(False, ocr_decline=None)


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
