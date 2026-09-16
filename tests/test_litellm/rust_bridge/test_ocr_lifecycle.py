from collections.abc import Generator
from typing import Final
from unittest.mock import AsyncMock, Mock

import pytest

import litellm
from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.ocr import legacy
from litellm.rust_bridge import bindings, configuration
from litellm.rust_bridge.ocr import NATIVE_AOCR, NATIVE_OCR, post_call, pre_call, update_logging


@pytest.fixture(autouse=True)
def isolated_ocr_configuration(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    monkeypatch.delenv("LITELLM_RUST", raising=False)
    configuration.reset_rust_configuration()
    yield
    NATIVE_OCR.reset()
    NATIVE_AOCR.reset()
    configuration.reset_rust_configuration()


def test_logging_redacts_views_and_preserves_opaque_arguments_and_pricing() -> None:
    logger: Final = Mock()
    opaque: Final = object()
    logger_fn: Final = object()
    kwargs: Final = {
        "vertex_credentials": "secret",
        "proxy_server_request": opaque,
        "metadata": opaque,
        "logger_fn": logger_fn,
        "litellm_request_debug": False,
        "litellm_call_id": "call-id",
        "input_cost_per_token": 0,
        "output_cost_per_token": None,
    }
    optional: Final = {"vertex_credentials": "secret", "pages": [1], "proxy_server_request": opaque}
    update_logging(logger, kwargs, "model", "vertex_ai", optional, ("vertex_credentials",), "https://provider")
    logger.update_from_kwargs.assert_called_once_with(
        kwargs={
            "vertex_credentials": "****",
            "metadata": opaque,
            "logger_fn": logger_fn,
            "litellm_request_debug": False,
            "litellm_call_id": "call-id",
            "input_cost_per_token": 0,
            "output_cost_per_token": None,
        },
        model="model",
        custom_llm_provider="vertex_ai",
        optional_params={"vertex_credentials": "****", "pages": [1]},
        litellm_params={
            "litellm_call_id": "call-id",
            "api_base": "https://provider",
            "logger_fn": logger_fn,
            "litellm_request_debug": False,
            "input_cost_per_token": 0,
        },
    )
    assert kwargs["vertex_credentials"] == optional["vertex_credentials"] == "secret"
    assert kwargs["proxy_server_request"] is optional["proxy_server_request"] is opaque
    assert logger.update_from_kwargs.call_args.kwargs["kwargs"]["metadata"] is opaque
    assert logger.update_from_kwargs.call_args.kwargs["optional_params"]["pages"] is optional["pages"]


def test_logging_callbacks_receive_captured_payload_roots_and_propagate_errors() -> None:
    logger: Final = Mock()
    body: Final[dict[str, object]] = {"document": "original"}
    headers: Final = {"authorization": "key"}
    response: Final = object()
    pre_call(logger, "key", body, headers, "https://provider")
    post_call(logger, response, body, headers)
    logger.pre_call.assert_called_once_with(
        input="OCR document processing",
        api_key="key",
        additional_args={"complete_input_dict": body, "headers": headers, "api_base": "https://provider"},
    )
    for callback in (logger.pre_call, logger.post_call):
        assert callback.call_args.kwargs["additional_args"]["complete_input_dict"] is body
        assert callback.call_args.kwargs["additional_args"]["headers"] is headers
    assert logger.post_call.call_args.kwargs["original_response"] is response
    failure: Final = RuntimeError("callback failed")
    failing_logger: Final = Mock(pre_call=Mock(side_effect=failure))
    with pytest.raises(RuntimeError) as caught:
        pre_call(failing_logger, None, body, headers, "https://provider")
    assert caught.value is failure


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_unavailable_native_uses_legacy(monkeypatch: pytest.MonkeyPatch, asynchronous: bool) -> None:
    response: Final = OCRResponse(pages=[], model="mistral-ocr-latest")
    fallback: Final = AsyncMock(return_value=response) if asynchronous else Mock(return_value=response)
    monkeypatch.setattr(legacy, "aocr" if asynchronous else "ocr", fallback)
    (NATIVE_AOCR if asynchronous else NATIVE_OCR).override(None)
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
    NATIVE_OCR.override(native)
    try:
        with pytest.raises(RuntimeError) as caught:
            litellm.ocr("mistral/mistral-ocr-latest", {"type": "document_url", "document_url": "https://example.com"})
        assert caught.value is failure
    finally:
        NATIVE_OCR.reset()
        litellm.rust(None)
    assert native.call_count == 1


def test_public_binding_keeps_positional_fields_and_defaults_out_of_native_hook_kwargs() -> None:
    document: Final = {"type": "document_url", "document_url": "https://example.com"}
    captured: Final = []

    def native(*args: object, **kwargs: object) -> OCRResponse:
        captured.append((args, kwargs))
        return OCRResponse(pages=[], model=args[0])

    litellm.rust(True)
    NATIVE_OCR.override(native)
    try:
        response: Final = litellm.ocr("mistral/mistral-ocr-latest", document)
    finally:
        NATIVE_OCR.reset()
        litellm.rust(None)

    call_args, hook_kwargs = captured[0]
    assert response.model == "mistral/mistral-ocr-latest"
    assert call_args[1] is document
    assert call_args == ("mistral/mistral-ocr-latest", document)
    assert hook_kwargs == {}


def test_public_binding_keeps_keyword_model_and_document_in_native_hook_kwargs() -> None:
    document: Final = {"type": "document_url", "document_url": "https://example.com"}
    captured: Final = []

    def native(*args: object, **kwargs: object) -> OCRResponse:
        assert args == ()
        captured.append(kwargs)
        return OCRResponse(pages=[], model=kwargs["model"])

    litellm.rust(True)
    NATIVE_OCR.override(native)
    try:
        litellm.ocr(model="mistral/mistral-ocr-latest", document=document)
    finally:
        NATIVE_OCR.reset()
        litellm.rust(None)

    assert captured[0]["model"] == "mistral/mistral-ocr-latest"
    assert captured[0]["document"] is document
    assert "timeout" not in captured[0]


@pytest.mark.parametrize("enabled", [False, True], ids=["flag-disabled", "flag-enabled"])
@pytest.mark.parametrize(
    "call, message",
    [
        (
            lambda: litellm.ocr("mistral/mistral-ocr-latest", {"type": "document_url"}, model="duplicate"),
            r"ocr\(\) got multiple values for argument 'model'",
        ),
        (
            lambda: litellm.ocr("mistral/mistral-ocr-latest"),
            r"ocr\(\) missing 1 required positional argument: 'document'",
        ),
    ],
    ids=["duplicate", "missing"],
)
def test_public_binding_errors_do_not_depend_on_native_selection(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, call, message: str
) -> None:
    fallback: Final = Mock(side_effect=AssertionError("legacy must not run after a native binding error"))
    if enabled and NATIVE_OCR.load() is not None:
        monkeypatch.setattr(legacy, "ocr", fallback)
    litellm.rust(enabled)
    try:
        with pytest.raises(TypeError, match=message):
            call()
    finally:
        litellm.rust(None)
    assert fallback.call_count == 0


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
    (NATIVE_AOCR if asynchronous else NATIVE_OCR).override(native)
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
    (NATIVE_AOCR if asynchronous else NATIVE_OCR).override(native)
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
