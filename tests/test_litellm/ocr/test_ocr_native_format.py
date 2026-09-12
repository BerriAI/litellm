"""
Tests for the OCR `req_format` option in the SDK request path.
"""

import pytest

import litellm
from litellm.rust_bridge import ocr as rust_ocr_bridge
from litellm.rust_bridge.ocr import LiteLLMOcrRequest

DOCUMENT = {"type": "document_url", "document_url": "https://example.com/doc.pdf"}


def _request(
    optional_params: dict[str, object], model: str = "azure_ai/doc-intelligence/prebuilt-layout"
) -> LiteLLMOcrRequest:
    return LiteLLMOcrRequest(
        model=model,
        document=DOCUMENT,
        api_key="fake-key",
        api_base=None,
        custom_llm_provider=None,
        extra_headers=None,
        timeout=60.0,
        kwargs=optional_params,
    )


@pytest.mark.parametrize("optional_params", [{}, {"req_format": "litellm"}])
def test_rust_ocr_serves_default_format(optional_params):
    assert rust_ocr_bridge.supported(_request(optional_params)) is True


def test_rust_ocr_serves_native_format_for_document_intelligence():
    assert rust_ocr_bridge.supported(_request({"req_format": "native"})) is True


def test_rust_ocr_response_retains_provider_native_response():
    provider_response = {"status": "succeeded", "analyzeResult": {"content": "native"}}
    response = rust_ocr_bridge._response(
        {
            "pages": [],
            "model": "prebuilt-layout",
            "document_annotation": None,
            "usage_info": {"pages_processed": 0},
            "object": "ocr",
            "provider_native_response": provider_response,
        }
    )

    assert response.get_provider_native_response() == provider_response
    assert response.model_dump().get("provider_native_response") is None


@pytest.mark.parametrize("model", ["cohere/cohere-parse", "azure_ai/cohere-parse"])
def test_rust_ocr_skipped_for_unsupported_models(model):
    assert rust_ocr_bridge.supported(_request({}, model)) is False


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
