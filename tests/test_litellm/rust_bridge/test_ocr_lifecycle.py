from collections.abc import Generator, Mapping
from typing import Final
from unittest.mock import AsyncMock, Mock

import pytest

import litellm
from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.ocr import legacy
from litellm.rust_bridge import bindings, configuration
from litellm.rust_bridge.ocr import LiteLLMOcrRequest
from litellm.rust_bridge.ocr_lifecycle import NATIVE_OCR_LIFECYCLE


@pytest.fixture(autouse=True)
def isolated_ocr_configuration(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    monkeypatch.delenv("LITELLM_RUST", raising=False)
    configuration.reset_rust_configuration()
    yield
    NATIVE_OCR_LIFECYCLE.reset()
    configuration.reset_rust_configuration()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_unavailable_native_uses_legacy(monkeypatch: pytest.MonkeyPatch, asynchronous: bool) -> None:
    response: Final = OCRResponse(pages=[], model="mistral-ocr-latest")
    fallback: Final = AsyncMock(return_value=response) if asynchronous else Mock(return_value=response)
    monkeypatch.setattr(legacy, "aocr" if asynchronous else "ocr", fallback)
    NATIVE_OCR_LIFECYCLE.override(None)
    document: Final = {"type": "document_url", "document_url": "https://example.com"}

    result: Final = (
        await litellm.aocr("mistral/mistral-ocr-latest", document, pages=[0])
        if asynchronous
        else litellm.ocr("mistral/mistral-ocr-latest", document, pages=[0])
    )

    assert result is response
    fallback.assert_called_once_with("mistral/mistral-ocr-latest", document, pages=[0])


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
async def test_environment_opt_out_never_loads_native(
    monkeypatch: pytest.MonkeyPatch, asynchronous: bool, enabled: bool | None
) -> None:
    monkeypatch.setenv("LITELLM_RUST", "0")
    response: Final = OCRResponse(pages=[], model="mistral-ocr-latest")
    fallback: Final = AsyncMock(return_value=response) if asynchronous else Mock(return_value=response)
    monkeypatch.setattr(legacy, "aocr" if asynchronous else "ocr", fallback)
    load: Final = Mock(side_effect=AssertionError("native must not be loaded"))
    monkeypatch.setattr(bindings, "get_native_bridge", load)
    litellm.rust(enabled)
    document: Final = {"type": "file", "file": b"pdf"}

    result: Final = (
        await litellm.aocr("mistral/mistral-ocr-latest", document, pages=[1])
        if asynchronous
        else litellm.ocr("mistral/mistral-ocr-latest", document, pages=[1])
    )

    assert result is response
    fallback.assert_called_once_with("mistral/mistral-ocr-latest", document, pages=[1])
    load.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("environment", [None, "1"])
async def test_native_is_enabled_by_default(
    monkeypatch: pytest.MonkeyPatch, asynchronous: bool, environment: str | None
) -> None:
    if environment is not None:
        monkeypatch.setenv("LITELLM_RUST", environment)
    response: Final = OCRResponse(pages=[], model="mistral-ocr-latest")
    native: Final = AsyncMock(return_value=response) if asynchronous else Mock(return_value=response)
    NATIVE_OCR_LIFECYCLE.override(native)
    fallback: Final = Mock(side_effect=AssertionError("legacy must not run"))
    monkeypatch.setattr(legacy, "aocr" if asynchronous else "ocr", fallback)

    result: Final = (
        await litellm.aocr("mistral/mistral-ocr-latest", {})
        if asynchronous
        else litellm.ocr("mistral/mistral-ocr-latest", {})
    )

    assert result is response
    assert native.call_count == 1
    fallback.assert_not_called()


class Declined(Exception):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("declined", [False, True])
async def test_only_native_declines_replay_on_legacy(
    monkeypatch: pytest.MonkeyPatch, asynchronous: bool, declined: bool
) -> None:
    failure: Final = Declined("unsupported") if declined else RuntimeError("provider already called")
    native: Final = AsyncMock(side_effect=failure) if asynchronous else Mock(side_effect=failure)
    NATIVE_OCR_LIFECYCLE.override(native)
    import importlib

    main: Final = importlib.import_module("litellm.ocr.main")
    monkeypatch.setattr(main, "native_exception_types", lambda: (Declined, RuntimeError))
    response: Final = OCRResponse(pages=[], model="mistral-ocr-latest")
    fallback: Final = AsyncMock(return_value=response) if asynchronous else Mock(return_value=response)
    monkeypatch.setattr(legacy, "aocr" if asynchronous else "ocr", fallback)
    document: Final = {"type": "file", "file": b"pdf"}

    async def call() -> object:
        if asynchronous:
            return await litellm.aocr("mistral/mistral-ocr-latest", document, pages=[0])
        return litellm.ocr("mistral/mistral-ocr-latest", document, pages=[0])

    if declined:
        assert await call() is response
        fallback.assert_called_once_with("mistral/mistral-ocr-latest", document, pages=[0])
    else:
        with pytest.raises(RuntimeError) as caught:
            await call()
        assert caught.value is failure
        fallback.assert_not_called()
    assert native.call_count == 1
