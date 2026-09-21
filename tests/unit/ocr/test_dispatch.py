from collections.abc import Awaitable, Callable, Mapping
from typing import Final, cast  # noqa: TID251  # narrows legacy callable signatures for inspect

import httpx
import pytest

import litellm
from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.ocr.dispatch import (
    _ADISPATCH,  # pyright: ignore[reportPrivateUsage]  # tests configured dispatch
    _DISPATCH,  # pyright: ignore[reportPrivateUsage]  # tests configured dispatch
)
from litellm.rust_bridge import catalog
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.catalog import Route, Rule, Rules
from litellm.rust_bridge.configuration import Rollout
from litellm.rust_bridge.ocr.entrypoints import (
    NATIVE_AOCR,
    NATIVE_OCR,
    LiteLLMOcrRequest,
    NativeAocr,
    NativeOcr,
)

PYTHON_RULES: Final[Rules] = (Rule(Route.OCR, Rollout.PYTHON_ONLY),)
RUST_RULES: Final[Rules] = (Rule(Route.OCR, Rollout.RUST_REQUIRED),)


def ocr_binding(native: NativeOcr | None) -> NativeBinding[NativeOcr]:
    binding: Final[NativeBinding[NativeOcr]] = NativeBinding("ocr", validate=lambda _: None)
    binding.override(native)
    return binding


def aocr_binding(native: NativeAocr | None) -> NativeBinding[NativeAocr]:
    binding: Final[NativeBinding[NativeAocr]] = NativeBinding("aocr", validate=lambda _: None)
    binding.override(native)
    return binding


def response(model: str = "mistral/mistral-ocr-latest") -> OCRResponse:
    return OCRResponse(pages=[], model=model)


def test_python_route_forwards_original_call_shape() -> None:
    document: Final[Mapping[str, object]] = {
        "type": "document_url",
        "document_url": "https://example.invalid/document.pdf",
    }
    pages: Final = [0]
    args: Final[tuple[object, ...]] = ("mistral/mistral-ocr-latest", document)
    kwargs: Final[Mapping[str, object]] = {"pages": pages}
    captured: Final[list[tuple[tuple[object, ...], Mapping[str, object]]]] = []
    expected: Final = response()

    def python(*call_args: object, **call_kwargs: object) -> OCRResponse:  # kwargs-ok: records public call shape
        captured.append((call_args, call_kwargs))
        return expected

    def native(
        request: LiteLLMOcrRequest,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> OCRResponse:
        pytest.fail("Python-only dispatch must not call native")

    result: Final = _DISPATCH.run(
        args,
        kwargs,
        python=python,
        binding=ocr_binding(native),
        native=lambda hook, request, call_args, call_kwargs: hook(request, call_args, call_kwargs),
        rules=PYTHON_RULES,
    )

    assert result is expected
    call_args, call_kwargs = captured[0]
    assert call_args == args
    assert call_args[1] is document
    assert call_kwargs == kwargs
    assert call_kwargs["pages"] is pages
    assert kwargs == {"pages": pages}


@pytest.mark.asyncio
async def test_async_python_route_forwards_original_call_shape() -> None:
    document: Final[Mapping[str, object]] = {"type": "file", "file": b"pdf"}
    pages: Final = [1]
    args: Final[tuple[object, ...]] = ("mistral/mistral-ocr-latest", document)
    kwargs: Final[Mapping[str, object]] = {"pages": pages}
    captured: Final[list[tuple[tuple[object, ...], Mapping[str, object]]]] = []
    expected: Final = response()

    async def python(
        *call_args: object,
        **call_kwargs: object,  # kwargs-ok: records public call shape
    ) -> OCRResponse:
        captured.append((call_args, call_kwargs))
        return expected

    async def native(
        request: LiteLLMOcrRequest,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> OCRResponse:
        pytest.fail("Python-only dispatch must not call native")

    result: Final = await _ADISPATCH.arun(
        args,
        kwargs,
        python=python,
        binding=aocr_binding(native),
        native=lambda hook, request, call_args, call_kwargs: hook(request, call_args, call_kwargs),
        rules=PYTHON_RULES,
    )

    assert result is expected
    call_args, call_kwargs = captured[0]
    assert call_args == args
    assert call_args[1] is document
    assert call_kwargs == kwargs
    assert call_kwargs["pages"] is pages
    assert kwargs == {"pages": pages}


def test_native_receives_normalized_positional_request_and_original_call_shape() -> None:
    document: Final[Mapping[str, object]] = {"type": "file", "file": b"pdf"}
    timeout: Final = httpx.Timeout(30)
    extra_headers: Final[dict[str, object]] = {"x-test": "1"}
    pages: Final = [0, 2]
    args: Final[tuple[object, ...]] = ("mistral/mistral-ocr-latest", document)
    kwargs: Final[Mapping[str, object]] = {
        "api_key": "test-key",
        "api_base": "https://example.invalid",
        "timeout": timeout,
        "custom_llm_provider": "mistral",
        "extra_headers": extra_headers,
        "pages": pages,
    }
    captured: Final[list[tuple[LiteLLMOcrRequest, tuple[object, ...], Mapping[str, object]]]] = []
    expected: Final = response()

    def python(*call_args: object, **call_kwargs: object) -> OCRResponse:  # kwargs-ok: rejected fallback
        pytest.fail("Required Rust dispatch must not call Python")

    def native(
        request: LiteLLMOcrRequest,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> OCRResponse:
        captured.append((request, args, kwargs))
        return expected

    result: Final = _DISPATCH.run(
        args,
        kwargs,
        python=python,
        binding=ocr_binding(native),
        native=lambda hook, request, call_args, call_kwargs: hook(request, call_args, call_kwargs),
        rules=RUST_RULES,
    )

    request, call_args, call_kwargs = captured[0]
    assert result is expected
    assert request.model == "mistral/mistral-ocr-latest"
    assert request.document is document
    assert request.api_key == "test-key"
    assert request.api_base == "https://example.invalid"
    assert request.timeout is timeout
    assert request.custom_llm_provider == "mistral"
    assert request.extra_headers is extra_headers
    assert request.kwargs == {"pages": pages}
    assert request.kwargs["pages"] is pages
    assert call_args is args
    assert call_kwargs is kwargs


def test_native_preserves_keyword_model_and_document_in_original_call_shape() -> None:
    document: Final[Mapping[str, object]] = {
        "type": "document_url",
        "document_url": "https://example.invalid/document.pdf",
    }
    pages: Final = [1]
    args: Final[tuple[object, ...]] = ()
    kwargs: Final[Mapping[str, object]] = {
        "model": "mistral/mistral-ocr-latest",
        "document": document,
        "pages": pages,
    }
    captured: Final[list[tuple[LiteLLMOcrRequest, tuple[object, ...], Mapping[str, object]]]] = []
    expected: Final = response()

    def python(*call_args: object, **call_kwargs: object) -> OCRResponse:  # kwargs-ok: rejected fallback
        pytest.fail("Required Rust dispatch must not call Python")

    def native(
        request: LiteLLMOcrRequest,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> OCRResponse:
        captured.append((request, args, kwargs))
        return expected

    result: Final = _DISPATCH.run(
        args,
        kwargs,
        python=python,
        binding=ocr_binding(native),
        native=lambda hook, request, call_args, call_kwargs: hook(request, call_args, call_kwargs),
        rules=RUST_RULES,
    )

    request, call_args, call_kwargs = captured[0]
    assert result is expected
    assert request.model == "mistral/mistral-ocr-latest"
    assert request.document is document
    assert request.kwargs == {"pages": pages}
    assert call_args is args
    assert call_kwargs is kwargs
    assert call_kwargs["model"] == "mistral/mistral-ocr-latest"
    assert call_kwargs["document"] is document


def test_aocr_marker_bypasses_native() -> None:
    document: Final[Mapping[str, object]] = {"type": "file", "file": b"pdf"}
    args: Final[tuple[object, ...]] = ("mistral/mistral-ocr-latest", document)
    kwargs: Final[Mapping[str, object]] = {"aocr": True}
    captured: Final[list[tuple[tuple[object, ...], Mapping[str, object]]]] = []
    expected: Final = response()

    def python(*call_args: object, **call_kwargs: object) -> OCRResponse:  # kwargs-ok: records public call shape
        captured.append((call_args, call_kwargs))
        return expected

    def native(
        request: LiteLLMOcrRequest,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> OCRResponse:
        pytest.fail("aocr's inner ocr call must stay on Python")

    result: Final = _DISPATCH.run(
        args,
        kwargs,
        python=python,
        binding=ocr_binding(native),
        native=lambda hook, request, call_args, call_kwargs: hook(request, call_args, call_kwargs),
        rules=RUST_RULES,
    )

    assert result is expected
    assert captured == [(args, kwargs)]


@pytest.mark.parametrize(
    ("args", "kwargs", "message"),
    (
        (
            ("mistral/mistral-ocr-latest", {"type": "file", "file": b"pdf"}),
            {"model": "duplicate"},
            r"ocr\(\) got multiple values for argument 'model'",
        ),
        (
            ("mistral/mistral-ocr-latest",),
            {},
            r"ocr\(\) missing 1 required positional argument: 'document'",
        ),
    ),
)
def test_ocr_parser_errors_before_python_or_native(
    args: tuple[object, ...], kwargs: Mapping[str, object], message: str
) -> None:
    def python(*call_args: object, **call_kwargs: object) -> OCRResponse:  # kwargs-ok: rejects parser failures
        pytest.fail("OCR parser failures must not call Python")

    def native(
        request: LiteLLMOcrRequest,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> OCRResponse:
        pytest.fail("OCR parser failures must not call native")

    with pytest.raises(TypeError, match=message):
        _DISPATCH.run(
            args,
            kwargs,
            python=python,
            binding=ocr_binding(native),
            native=lambda hook, request, call_args, call_kwargs: hook(request, call_args, call_kwargs),
            rules=RUST_RULES,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("args", "kwargs", "message"),
    (
        (
            ("mistral/mistral-ocr-latest", {"type": "file", "file": b"pdf"}),
            {"model": "duplicate"},
            r"aocr\(\) got multiple values for argument 'model'",
        ),
        (
            ("mistral/mistral-ocr-latest",),
            {},
            r"aocr\(\) missing 1 required positional argument: 'document'",
        ),
    ),
)
async def test_aocr_parser_errors_before_python_or_native(
    args: tuple[object, ...], kwargs: Mapping[str, object], message: str
) -> None:
    async def python(
        *call_args: object,
        **call_kwargs: object,  # kwargs-ok: rejects parser failures
    ) -> OCRResponse:
        pytest.fail("OCR parser failures must not call Python")

    async def native(
        request: LiteLLMOcrRequest,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> OCRResponse:
        pytest.fail("OCR parser failures must not call native")

    with pytest.raises(TypeError, match=message):
        await _ADISPATCH.arun(
            args,
            kwargs,
            python=python,
            binding=aocr_binding(native),
            native=lambda hook, request, call_args, call_kwargs: hook(request, call_args, call_kwargs),
            rules=RUST_RULES,
        )


def test_public_ocr_routes_through_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    document: Final[Mapping[str, object]] = {
        "type": "document_url",
        "document_url": "https://example.invalid/document.pdf",
    }
    captured: Final[list[LiteLLMOcrRequest]] = []
    expected: Final = response()

    def native(
        request: LiteLLMOcrRequest,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> OCRResponse:
        captured.append(request)
        return expected

    NATIVE_OCR.override(native)
    monkeypatch.setattr(catalog, "RULES", RUST_RULES)
    public_ocr: Final = cast(Callable[..., OCRResponse], litellm.ocr)
    try:
        result: Final = public_ocr(model="mistral/mistral-ocr-latest", document=document)
    finally:
        NATIVE_OCR.reset()
    assert result is expected
    assert [request.model for request in captured] == ["mistral/mistral-ocr-latest"]


@pytest.mark.asyncio
async def test_public_aocr_routes_through_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    document: Final[Mapping[str, object]] = {
        "type": "document_url",
        "document_url": "https://example.invalid/document.pdf",
    }
    captured: Final[list[LiteLLMOcrRequest]] = []
    expected: Final = response()

    async def native(
        request: LiteLLMOcrRequest,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> OCRResponse:
        captured.append(request)
        return expected

    NATIVE_AOCR.override(native)
    monkeypatch.setattr(catalog, "RULES", RUST_RULES)
    public_aocr: Final = cast(Callable[..., Awaitable[OCRResponse]], litellm.aocr)
    try:
        result: Final = await public_aocr(model="mistral/mistral-ocr-latest", document=document)
    finally:
        NATIVE_AOCR.reset()
    assert result is expected
    assert [request.model for request in captured] == ["mistral/mistral-ocr-latest"]


@pytest.mark.parametrize(
    ("model", "custom_llm_provider", "expected"),
    (
        ("aws_textract/detect-document-text", None, "native"),
        ("detect-document-text", "aws_textract", "native"),
        ("mistral/mistral-ocr-latest", None, "python"),
        ("mistral/mistral-ocr-latest", "aws_textract", "native"),
        ("aws_textract", None, "python"),
    ),
)
def test_provider_scoped_rule_sees_the_provider_named_by_the_model_prefix(
    model: str, custom_llm_provider: str | None, expected: str
) -> None:
    rules: Final[Rules] = (
        Rule(Route.OCR, Rollout.RUST_REQUIRED, providers=frozenset({"aws_textract"})),
        Rule(Route.OCR, Rollout.PYTHON_ONLY),
    )
    document: Final[Mapping[str, object]] = {"type": "image_url", "image_url": "data:image/png;base64,YQ=="}
    kwargs: Final[Mapping[str, object]] = (
        {} if custom_llm_provider is None else {"custom_llm_provider": custom_llm_provider}
    )
    python_response: Final = response("python")
    native_response: Final = response("native")

    result: Final = _DISPATCH.run(
        (model, document),
        kwargs,
        python=lambda *_args, **_kwargs: python_response,
        binding=ocr_binding(lambda *_args, **_kwargs: native_response),
        native=lambda _hook, _request, _args, _kwargs: native_response,
        rules=rules,
    )

    assert cast(OCRResponse, result).model == expected  # noqa: TID251  # sync dispatch returns the response itself
