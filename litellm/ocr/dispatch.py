from collections.abc import Awaitable, Callable, Coroutine, Mapping
from typing import Final, cast  # noqa: TID251  # native binding selects a sync result or an async awaitable

import httpx

from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.ocr import main
from litellm.ocr.input import convert_file_document_to_url_document, get_mime_type
from litellm.rust_bridge.catalog import Context, Route
from litellm.rust_bridge.dispatch import PublicDispatch
from litellm.rust_bridge.ocr.entrypoints import NATIVE_AOCR, NATIVE_OCR, LiteLLMOcrRequest

__all__ = ("aocr", "convert_file_document_to_url_document", "get_mime_type", "ocr")


def _bind_request(
    model: str,
    document: Mapping[str, object],
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: dict[str, object] | None = None,
    **kwargs: object,  # kwargs-ok: public OCR accepts provider-specific options
) -> LiteLLMOcrRequest:
    return LiteLLMOcrRequest(
        model=model,
        document=document,
        api_key=api_key,
        api_base=api_base,
        timeout=timeout,
        custom_llm_provider=custom_llm_provider,
        extra_headers=extra_headers,
        kwargs=kwargs,
    )


def _public_request(name: str, args: tuple[object, ...], kwargs: Mapping[str, object]) -> LiteLLMOcrRequest:
    try:
        return _bind_request(*args, **kwargs)  # pyright: ignore[reportArgumentType]  # Python binds the public arguments before native validation
    except TypeError as error:
        raise TypeError(str(error).replace("_bind_request()", f"{name}()")) from None


_DISPATCH: Final = PublicDispatch(
    route=Route.OCR,
    request=lambda args, kwargs: _public_request("ocr", args, kwargs),
    context=lambda request: _context(request),
    bypass=lambda request: request.kwargs.get("aocr") is True,
)

_ADISPATCH: Final = PublicDispatch(
    route=Route.OCR,
    request=lambda args, kwargs: _public_request("aocr", args, kwargs),
    context=lambda request: _context(request),
)


def ocr(
    *args: object,
    **kwargs: object,  # kwargs-ok: preserve the public OCR call shape
) -> OCRResponse | Coroutine[object, object, OCRResponse]:
    python_ocr: Final = cast(  # cast-ok: forward the original call shape through the Python @client decorator
        Callable[..., OCRResponse | Coroutine[object, object, OCRResponse]], main.ocr
    )
    return _DISPATCH.run(
        args,
        kwargs,
        python=python_ocr,
        binding=NATIVE_OCR,
        native=lambda hook, request, call_args, call_kwargs: hook(request, call_args, call_kwargs),
    )


async def aocr(*args: object, **kwargs: object) -> OCRResponse:  # kwargs-ok: preserve the public OCR call shape
    fallback: Final = cast(  # cast-ok: forward the original call shape through the Python @client decorator
        Callable[..., Awaitable[OCRResponse]], main.aocr
    )
    return await _ADISPATCH.arun(
        args,
        kwargs,
        python=fallback,
        binding=NATIVE_AOCR,
        native=lambda hook, request, call_args, call_kwargs: hook(request, call_args, call_kwargs),
    )


def _context(request: LiteLLMOcrRequest) -> Context:
    return Context(Route.OCR, provider=request.custom_llm_provider, model=request.model)
