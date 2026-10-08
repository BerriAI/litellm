from collections.abc import Awaitable, Callable, Mapping
from typing import Final, cast  # noqa: TID251  # narrows legacy callable signatures

import httpx
import pytest

import litellm
from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.ocr.dispatch import (
    _ADISPATCH,  # pyright: ignore[reportPrivateUsage]  # tests configured dispatch
    _DISPATCH,  # pyright: ignore[reportPrivateUsage]  # tests configured dispatch
)
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.catalog import Route, RouteContext, Rust
from litellm.rust_bridge.dispatch import NativeDispatch
from litellm.rust_bridge.ocr.entrypoints import NATIVE_AOCR, NATIVE_OCR, NativeOcr
from litellm.rust_bridge.public_call import NativeCall

DOCUMENT: Final[Mapping[str, object]] = {"type": "file", "file": b"pdf"}


def response(model: str = "mistral/mistral-ocr-latest") -> OCRResponse:
    return OCRResponse(pages=[], model=model)


def test_request_binds_positional_and_keyword_arguments() -> None:
    timeout: Final = httpx.Timeout(30)
    extra_headers: Final[dict[str, object]] = {"x-test": "1"}
    pages: Final = [0, 2]
    kwargs: Final[Mapping[str, object]] = {
        "api_key": "test-key",
        "api_base": "https://example.invalid",
        "timeout": timeout,
        "custom_llm_provider": "mistral",
        "extra_headers": extra_headers,
        "pages": pages,
    }

    request: Final = _DISPATCH.request(("mistral/mistral-ocr-latest", DOCUMENT), kwargs)

    assert request.bound["model"] == "mistral/mistral-ocr-latest"
    assert request.bound["document"] is DOCUMENT
    assert request.bound["api_key"] == "test-key"
    assert request.bound["api_base"] == "https://example.invalid"
    assert request.bound["timeout"] is timeout
    assert request.bound["custom_llm_provider"] == "mistral"
    assert request.bound["extra_headers"] is extra_headers
    assert request.bound["pages"] is pages
    assert request.kwargs is kwargs


def test_request_binds_keyword_model_and_document() -> None:
    request: Final = _DISPATCH.request((), {"model": "mistral/mistral-ocr-latest", "document": DOCUMENT})

    assert request.bound["model"] == "mistral/mistral-ocr-latest"
    assert request.bound["document"] is DOCUMENT
    assert "kwargs" not in request.bound


@pytest.mark.parametrize(
    ("model", "custom_llm_provider", "provider"),
    (
        pytest.param("mistral/mistral-ocr-latest", None, "mistral", id="model-prefix"),
        pytest.param("mistral-ocr-latest", None, None, id="no-prefix"),
        pytest.param("mistral/mistral-ocr-latest", "azure_ai", "azure_ai", id="declared-provider-wins"),
        pytest.param(None, None, None, id="unnamed-model"),
    ),
)
def test_context_reads_the_provider_from_the_declaration_or_the_model_prefix(
    model: object, custom_llm_provider: str | None, provider: str | None
) -> None:
    request: Final = _DISPATCH.request((model, DOCUMENT), {"custom_llm_provider": custom_llm_provider})

    assert _DISPATCH.context(request) == RouteContext(
        Route.OCR, provider=provider, model=model if isinstance(model, str) else None
    )


@pytest.mark.parametrize(("dispatch", "name"), ((_DISPATCH, "ocr"), (_ADISPATCH, "aocr")), ids=("ocr", "aocr"))
@pytest.mark.parametrize(
    ("args", "kwargs", "message"),
    (
        pytest.param(
            ("mistral/mistral-ocr-latest", DOCUMENT),
            {"model": "duplicate"},
            "got multiple values for argument 'model'",
            id="duplicate-model",
        ),
        pytest.param(
            ("mistral/mistral-ocr-latest",),
            {},
            "missing 1 required positional argument: 'document'",
            id="missing-document",
        ),
    ),
)
def test_request_raises_the_public_signature_error(
    dispatch: NativeDispatch, name: str, args: tuple[object, ...], kwargs: Mapping[str, object], message: str
) -> None:
    with pytest.raises(TypeError, match=rf"^{name}\(\) {message}$"):
        dispatch.request(args, kwargs)


@pytest.mark.parametrize("model", (None, 7, {"name": "mistral/mistral-ocr-latest"}))
def test_non_string_model_reaches_native_validation(model: object) -> None:
    seen: Final[list[NativeCall]] = []

    def native(request: NativeCall) -> OCRResponse:
        seen.append(request)
        return response()

    binding: Final[NativeBinding[NativeOcr]] = NativeBinding("ocr", validate=lambda _: None)
    binding.override(native)
    _DISPATCH.run(
        (model, DOCUMENT),
        {},
        binding=binding,
        native=lambda hook, request, call_args, call_kwargs: hook(request),
        policy=Rust(required=True),
    )

    assert [request.bound["model"] for request in seen] == [model]


def test_public_ocr_runs_natively_under_the_shipped_catalog() -> None:
    captured: Final[list[NativeCall]] = []
    expected: Final = response()

    def native(request: NativeCall) -> OCRResponse:
        captured.append(request)
        return expected

    NATIVE_OCR.override(native)
    public_ocr: Final = cast(Callable[..., OCRResponse], litellm.ocr)
    try:
        result: Final = public_ocr(model="mistral/mistral-ocr-latest", document=DOCUMENT)
    finally:
        NATIVE_OCR.reset()
    assert result is expected
    assert [request.bound["model"] for request in captured] == ["mistral/mistral-ocr-latest"]


@pytest.mark.asyncio
async def test_public_aocr_runs_natively_under_the_shipped_catalog() -> None:
    captured: Final[list[NativeCall]] = []
    expected: Final = response()

    async def native(request: NativeCall) -> OCRResponse:
        captured.append(request)
        return expected

    NATIVE_AOCR.override(native)
    public_aocr: Final = cast(Callable[..., Awaitable[OCRResponse]], litellm.aocr)
    try:
        result: Final = await public_aocr(model="mistral/mistral-ocr-latest", document=DOCUMENT)
    finally:
        NATIVE_AOCR.reset()
    assert result is expected
    assert [request.bound["model"] for request in captured] == ["mistral/mistral-ocr-latest"]
