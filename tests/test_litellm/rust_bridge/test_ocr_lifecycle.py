from collections.abc import Mapping
from typing import Final
from unittest.mock import Mock

import pytest

import litellm
from litellm.ocr import legacy
from litellm.rust_bridge.ocr_lifecycle import NATIVE_OCR_LIFECYCLE
from litellm.llms.base_llm.ocr.transformation import OCRResponse


@pytest.mark.parametrize("enabled,available", [(True, False), (False, True), (False, False)])
def test_public_selection_falls_back_before_native_admission(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, available: bool
) -> None:
    response: Final = OCRResponse(pages=[], model="fallback")
    fallback: Final = Mock(return_value=response)
    native: Final = Mock(side_effect=AssertionError("must not admit"))
    monkeypatch.setattr(legacy, "ocr", fallback)
    litellm.rust(enabled)
    NATIVE_OCR_LIFECYCLE.override(native if available else None)
    try:
        assert (
            litellm.ocr("mistral/mistral-ocr-latest", {"type": "document_url", "document_url": "https://example.com"})
            is response
        )
    finally:
        NATIVE_OCR_LIFECYCLE.reset()
        litellm.rust(None)
    assert fallback.call_count == 1
    assert native.call_count == 0


def test_admitted_failure_is_never_restarted_through_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    failure: Final = RuntimeError("admitted")
    fallback: Final = Mock(side_effect=AssertionError("must not restart"))
    native: Final = Mock(side_effect=failure)
    monkeypatch.setattr(legacy, "ocr", fallback)
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
    assert fallback.call_count == 0
