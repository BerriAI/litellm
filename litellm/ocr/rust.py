from collections.abc import Awaitable, Callable, Coroutine, Mapping
from typing import Final, cast  # noqa: TID251  # native binding selects a sync result or an async awaitable

import httpx

from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.ocr import main
from litellm.ocr.input import convert_file_document_to_url_document, get_mime_type
from litellm.rust_bridge.catalog import Context, Route
from litellm.rust_bridge.ocr.lifecycle import NATIVE_OCR_LIFECYCLE, NativeOcrLifecycle
from litellm.rust_bridge.ocr.native import LiteLLMOcrRequest
from litellm.rust_bridge.runtime import arun, run

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


def _public_request(name: str, args: tuple[object, ...], kwargs: dict[str, object]) -> LiteLLMOcrRequest:
    try:
        return _bind_request(*args, **kwargs)  # pyright: ignore[reportArgumentType]  # Python binds the public arguments before native validation
    except TypeError as error:
        raise TypeError(str(error).replace("_bind_request()", f"{name}()")) from None


def ocr(
    *args: object,
    **kwargs: object,  # kwargs-ok: preserve the public OCR call shape
) -> OCRResponse | Coroutine[object, object, OCRResponse]:
    request: Final = _public_request("ocr", args, kwargs)
    fallback: Final = cast(  # cast-ok: forward the original call shape through the Python @client decorator
        Callable[..., OCRResponse | Coroutine[object, object, OCRResponse]], main.ocr
    )
    if request.kwargs.get("aocr"):
        return fallback(*args, **kwargs)
    return run(
        _context(request),
        binding=NATIVE_OCR_LIFECYCLE,
        native=lambda hook: cast(  # cast-ok: False selects the synchronous result
            OCRResponse, hook(request, args, kwargs, False)
        ),
        python=lambda: fallback(*args, **kwargs),
    )


async def aocr(*args: object, **kwargs: object) -> OCRResponse:  # kwargs-ok: preserve the public OCR call shape
    request: Final = _public_request("aocr", args, kwargs)
    fallback: Final = cast(  # cast-ok: forward the original call shape through the Python @client decorator
        Callable[..., Awaitable[OCRResponse]], main.aocr
    )

    async def native(hook: NativeOcrLifecycle) -> OCRResponse:
        return await cast(  # cast-ok: True selects the asynchronous result
            Awaitable[OCRResponse], hook(request, args, kwargs, True)
        )

    return await arun(
        _context(request), binding=NATIVE_OCR_LIFECYCLE, native=native, python=lambda: fallback(*args, **kwargs)
    )


def _context(request: LiteLLMOcrRequest) -> Context:
    return Context(Route.OCR, provider=request.custom_llm_provider, model=request.model)
