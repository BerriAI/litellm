from collections.abc import Mapping
from typing import Final, NoReturn

import httpx
import pytest

import litellm
from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.ocr.dispatch import (
    _ADISPATCH,  # pyright: ignore[reportPrivateUsage]  # tests configured dispatch
    _DISPATCH,  # pyright: ignore[reportPrivateUsage]  # tests configured dispatch
)
from litellm.rust_bridge import runtime
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.catalog import Route, RouteRule, Rules
from litellm.rust_bridge.configuration import Rollout
from litellm.rust_bridge.ocr.entrypoints import NativeAocr, NativeOcr
from litellm.rust_bridge.public_call import NativeCall
from tests.test_litellm_rust.ocr.data import OCR_DOCUMENT, OCR_MODEL
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.routing import assert_served_natively

pytestmark = pytest.mark.requires_rust_extension


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("caching", [None, False, True])
async def test_ocr_does_not_depend_on_chat_cache(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch, asynchronous: bool, caching: bool | None
) -> None:
    from litellm.caching.caching import Cache

    monkeypatch.setattr(litellm, "cache", Cache(type="local", supported_call_types=["completion", "acompletion"]))
    arguments: Final = {
        "model": OCR_MODEL,
        "document": OCR_DOCUMENT,
        "api_key": "test-key",
        "api_base": server.base_url,
        "caching": caching,
    }
    response: Final = await litellm.aocr(**arguments) if asynchronous else litellm.ocr(**arguments)
    assert response.pages[0].markdown == "native OCR response"
    assert len(server.requests) == 1


@pytest.mark.parametrize("failure", ["invalid", "live", "schema"])
def test_native_projection_errors_never_select_python(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    import dataclasses
    import ssl

    from litellm.rust_bridge import runtime
    from litellm.rust_bridge.catalog import Route, RouteContext
    from litellm.rust_bridge.host import settings
    from litellm.rust_bridge.ocr.entrypoints import NATIVE_OCR
    from litellm.rust_bridge.public_call import NativeCall

    server.expected_requests = 0
    snapshot: Final = dataclasses.replace(settings.http_settings(), user_agent=1)
    if failure == "schema":
        monkeypatch.setattr(settings, "http_settings", lambda: snapshot)
    else:
        monkeypatch.setattr(
            litellm, "ssl_verify", ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT) if failure == "live" else object()
        )
    request: Final = NativeCall(
        args=(),
        kwargs={},
        base={
            "model": "mistral/mistral-ocr-latest",
            "document": OCR_DOCUMENT,
            "api_key": "test-key",
            "api_base": server.base_url,
            "timeout": None,
            "custom_llm_provider": "mistral",
            "extra_headers": None,
        },
    )

    def python_fallback() -> NoReturn:
        pytest.fail("projection failures must not select Python")

    with pytest.raises(RuntimeError if failure == "schema" else ValueError, match="http_settings"):
        runtime.run(
            RouteContext(Route.OCR, provider="mistral"),
            binding=NATIVE_OCR,
            native=lambda native: native(request),
            python=python_fallback,
        )
    assert server.requests == []


MODELS: Final = ("cohere/parse-v5.0", "azure_ai/Cohere-parse-v5.0")


IMAGE: Final = {"type": "image_url", "image_url": "data:image/png;base64,YWJj"}


BOX: Final = {"top_left_x": 0, "top_left_y": 0, "bottom_right_x": 32, "bottom_right_y": 32}


PAYLOAD: Final = {
    "pages": [
        {
            "index": 4,
            "markdown": {"content": "receipt", "images": [{"id": "image", "bounding_box": BOX, "description": "scan"}]},
        },
        {"markdown": {"content": "page two"}},
    ],
    "meta": {"billed_units": {"pages": 3}},
}


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
async def test_ocr_health_check_uses_the_native_route(recording_server: RecordingServer, model: str) -> None:
    recording_server.enqueue(ResponseSpec(body=PAYLOAD))
    response: Final = await litellm.ahealth_check(
        model_params={"model": model, "api_key": "test-key", "api_base": recording_server.base_url}, mode="ocr"
    )
    assert "error" not in response
    assert_served_natively(recording_server)
    assert recording_server.requests[0].body["document"]["image_url"].startswith("data:image/png;base64,")


RUST_RULES: Final[Rules] = (RouteRule(Route.OCR, Rollout.RUST_REQUIRED),)


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
    captured: Final[list[tuple[NativeCall, tuple[object, ...], Mapping[str, object]]]] = []
    expected: Final = response()

    def native(
        request: NativeCall,
    ) -> OCRResponse:
        args: Final = request.args
        kwargs: Final = request.kwargs
        captured.append((request, args, kwargs))
        return expected

    result: Final = _DISPATCH.run(
        args,
        kwargs,
        python=runtime.NO_PYTHON,
        binding=ocr_binding(native),
        native=lambda hook, request, call_args, call_kwargs: hook(request),
        rules=RUST_RULES,
    )

    request, call_args, call_kwargs = captured[0]
    assert result is expected
    assert request.resolved["model"] == "mistral/mistral-ocr-latest"
    assert request.resolved["document"] is document
    assert request.resolved["api_key"] == "test-key"
    assert request.resolved["api_base"] == "https://example.invalid"
    assert request.resolved["timeout"] is timeout
    assert request.resolved["custom_llm_provider"] == "mistral"
    assert request.resolved["extra_headers"] is extra_headers
    assert request.kwargs == kwargs
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
    captured: Final[list[tuple[NativeCall, tuple[object, ...], Mapping[str, object]]]] = []
    expected: Final = response()

    def native(
        request: NativeCall,
    ) -> OCRResponse:
        args: Final = request.args
        kwargs: Final = request.kwargs
        captured.append((request, args, kwargs))
        return expected

    result: Final = _DISPATCH.run(
        args,
        kwargs,
        python=runtime.NO_PYTHON,
        binding=ocr_binding(native),
        native=lambda hook, request, call_args, call_kwargs: hook(request),
        rules=RUST_RULES,
    )

    request, call_args, call_kwargs = captured[0]
    assert result is expected
    assert request.resolved["model"] == "mistral/mistral-ocr-latest"
    assert request.resolved["document"] is document
    assert request.kwargs == kwargs
    assert call_args is args
    assert call_kwargs is kwargs
    assert call_kwargs["model"] == "mistral/mistral-ocr-latest"
    assert call_kwargs["document"] is document


def test_missing_native_binding_is_a_required_rust_error() -> None:
    args: Final[tuple[object, ...]] = ("mistral/mistral-ocr-latest", {"type": "file", "file": b"pdf"})

    with pytest.raises(RuntimeError, match="Rust ocr bridge"):
        _DISPATCH.run(
            args,
            {},
            python=runtime.NO_PYTHON,
            binding=ocr_binding(None),
            native=lambda hook, request, call_args, call_kwargs: hook(request),
            rules=RUST_RULES,
        )


def test_non_required_rule_cannot_be_served_without_python() -> None:
    args: Final[tuple[object, ...]] = ("mistral/mistral-ocr-latest", {"type": "file", "file": b"pdf"})

    def native(
        request: NativeCall,
    ) -> OCRResponse:
        return response()

    with pytest.raises(runtime.NoPythonImplementationError, match="RUST_REQUIRED"):
        _DISPATCH.run(
            args,
            {},
            python=runtime.NO_PYTHON,
            binding=ocr_binding(native),
            native=lambda hook, request, call_args, call_kwargs: hook(request),
            rules=(RouteRule(Route.OCR, Rollout.PYTHON_ONLY),),
        )


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
def test_ocr_parser_errors_before_native(args: tuple[object, ...], kwargs: Mapping[str, object], message: str) -> None:
    def native(
        request: NativeCall,
    ) -> OCRResponse:
        pytest.fail("OCR parser failures must not call native")

    with pytest.raises(TypeError, match=message):
        _DISPATCH.run(
            args,
            kwargs,
            python=runtime.NO_PYTHON,
            binding=ocr_binding(native),
            native=lambda hook, request, call_args, call_kwargs: hook(request),
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
async def test_aocr_parser_errors_before_native(
    args: tuple[object, ...], kwargs: Mapping[str, object], message: str
) -> None:
    async def native(
        request: NativeCall,
    ) -> OCRResponse:
        pytest.fail("OCR parser failures must not call native")

    with pytest.raises(TypeError, match=message):
        await _ADISPATCH.arun(
            args,
            kwargs,
            python=runtime.NO_PYTHON,
            binding=aocr_binding(native),
            native=lambda hook, request, call_args, call_kwargs: hook(request),
            rules=RUST_RULES,
        )
