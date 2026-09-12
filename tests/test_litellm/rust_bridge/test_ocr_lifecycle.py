from collections.abc import Mapping
from typing import Final
from unittest.mock import Mock

import pytest

import litellm
from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.rust_bridge.ocr import LiteLLMOcrRequest
from litellm.rust_bridge.ocr_lifecycle import NATIVE_OCR_LIFECYCLE


@pytest.mark.parametrize("enabled", [True, False])
def test_public_selection_requires_available_native_ocr(enabled: bool) -> None:
    native: Final = Mock(side_effect=AssertionError("must not admit"))
    litellm.rust(enabled)
    NATIVE_OCR_LIFECYCLE.override(None)
    try:
        with pytest.raises(RuntimeError, match="Rust OCR is unavailable or does not support this request"):
            litellm.ocr("mistral/mistral-ocr-latest", {"type": "document_url", "document_url": "https://example.com"})
    finally:
        NATIVE_OCR_LIFECYCLE.reset()
        litellm.rust(None)
    assert native.call_count == 0


def test_admitted_failure_is_returned_without_replay() -> None:
    failure: Final = RuntimeError("admitted")
    native: Final = Mock(side_effect=failure)
    litellm.rust(True)
    NATIVE_OCR_LIFECYCLE.override(native)
    try:
        with pytest.raises(RuntimeError) as caught:
            litellm.ocr("mistral/mistral-ocr-latest", {"type": "document_url", "document_url": "https://example.com"})
        assert caught.value is failure
    finally:
        NATIVE_OCR_LIFECYCLE.reset()
        litellm.rust(None)
    assert native.call_count == 1


def test_public_binding_keeps_positional_fields_and_defaults_out_of_native_hook_kwargs() -> None:
    document: Final = {"type": "document_url", "document_url": "https://example.com"}
    captured: Final = []

    def native(
        request: LiteLLMOcrRequest,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
        asynchronous: bool,
    ) -> OCRResponse:
        captured.append((request, args, kwargs, asynchronous))
        return OCRResponse(pages=[], model=request.model)

    litellm.rust(True)
    NATIVE_OCR_LIFECYCLE.override(native)
    try:
        response: Final = litellm.ocr("mistral/mistral-ocr-latest", document)
    finally:
        NATIVE_OCR_LIFECYCLE.reset()
        litellm.rust(None)

    request, call_args, hook_kwargs, asynchronous = captured[0]
    assert response.model == "mistral/mistral-ocr-latest"
    assert request.model == "mistral/mistral-ocr-latest"
    assert request.document is document
    assert call_args == ("mistral/mistral-ocr-latest", document)
    assert hook_kwargs == {}
    assert asynchronous is False


def test_public_binding_keeps_keyword_model_and_document_in_native_hook_kwargs() -> None:
    document: Final = {"type": "document_url", "document_url": "https://example.com"}
    captured: Final = []

    def native(
        request: LiteLLMOcrRequest,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
        asynchronous: bool,
    ) -> OCRResponse:
        assert args == ()
        captured.append(kwargs)
        return OCRResponse(pages=[], model=request.model)

    litellm.rust(True)
    NATIVE_OCR_LIFECYCLE.override(native)
    try:
        litellm.ocr(model="mistral/mistral-ocr-latest", document=document)
    finally:
        NATIVE_OCR_LIFECYCLE.reset()
        litellm.rust(None)

    assert captured[0]["model"] == "mistral/mistral-ocr-latest"
    assert captured[0]["document"] is document
    assert "timeout" not in captured[0]


@pytest.mark.parametrize("enabled", [False, True], ids=["flag-disabled", "flag-enabled"])
def test_public_duplicate_argument_error_does_not_depend_on_native_selection(enabled: bool) -> None:
    native: Final = Mock(side_effect=AssertionError("binding errors precede admission"))
    document: Final = {"type": "document_url", "document_url": "https://example.com"}
    litellm.rust(enabled)
    NATIVE_OCR_LIFECYCLE.override(native)
    try:
        with pytest.raises(TypeError, match=r"ocr\(\) got multiple values for argument 'model'"):
            litellm.ocr("mistral/mistral-ocr-latest", document, model="duplicate")
    finally:
        NATIVE_OCR_LIFECYCLE.reset()
        litellm.rust(None)
    assert native.call_count == 0


@pytest.mark.parametrize("enabled", [False, True], ids=["flag-disabled", "flag-enabled"])
def test_public_missing_required_argument_error_does_not_depend_on_native_selection(enabled: bool) -> None:
    native: Final = Mock(side_effect=AssertionError("binding errors precede admission"))
    litellm.rust(enabled)
    NATIVE_OCR_LIFECYCLE.override(native)
    try:
        with pytest.raises(TypeError, match=r"ocr\(\) missing 1 required positional argument: 'document'"):
            litellm.ocr("mistral/mistral-ocr-latest")
    finally:
        NATIVE_OCR_LIFECYCLE.reset()
        litellm.rust(None)
    assert native.call_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("enabled", [False, True, None])
async def test_public_ocr_ignores_rust_flag(
    monkeypatch: pytest.MonkeyPatch, asynchronous: bool, enabled: bool | None
) -> None:
    from unittest.mock import AsyncMock

    monkeypatch.setenv("LITELLM_RUST", "0")
    response: Final = OCRResponse(pages=[], model="mistral-ocr-latest")
    native: Final = AsyncMock(return_value=response) if asynchronous else Mock(return_value=response)
    litellm.rust(enabled)
    NATIVE_OCR_LIFECYCLE.override(native)
    try:
        if asynchronous:
            assert await litellm.aocr("mistral/mistral-ocr-latest", {}) is response
        else:
            assert litellm.ocr("mistral/mistral-ocr-latest", {}) is response
        assert native.call_count == 1
    finally:
        NATIVE_OCR_LIFECYCLE.reset()
        litellm.rust(None)
