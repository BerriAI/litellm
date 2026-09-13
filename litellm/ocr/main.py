from collections.abc import Awaitable, Callable, Coroutine, Mapping
from typing import Final, cast  # noqa: TID251  # native binding selects a sync result or an async awaitable

import httpx

from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.ocr import legacy
from litellm.ocr.input import convert_file_document_to_url_document, get_mime_type
from litellm.rust_bridge.bindings import native_exception_types
from litellm.rust_bridge.configuration import rust_ocr_enabled
from litellm.rust_bridge.ocr import LiteLLMOcrRequest
from litellm.rust_bridge.ocr_lifecycle import select

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
    native: Final = select(request) if rust_ocr_enabled() else None
    if native is not None:
        try:
            return cast(  # cast-ok: False selects the synchronous result
                OCRResponse, native(request, args, kwargs, False)
            )
        except _decline_types():
            pass
    fallback: Final = cast(  # cast-ok: forward the original call shape through the legacy @client decorator
        Callable[..., OCRResponse | Coroutine[object, object, OCRResponse]], legacy.ocr
    )
    return fallback(*args, **kwargs)


async def aocr(*args: object, **kwargs: object) -> OCRResponse:  # kwargs-ok: preserve the public OCR call shape
    request: Final = _public_request("aocr", args, kwargs)
    native: Final = select(request) if rust_ocr_enabled() else None
    if native is not None:
        try:
            return await cast(  # cast-ok: True selects the asynchronous result
                Awaitable[OCRResponse], native(request, args, kwargs, True)
            )
        except _decline_types():
            pass
    fallback: Final = cast(  # cast-ok: forward the original call shape through the legacy @client decorator
        Callable[..., Awaitable[OCRResponse]], legacy.aocr
    )
    return await fallback(*args, **kwargs)


def _decline_types() -> tuple[type[BaseException], ...]:
    exception_types: Final = native_exception_types()
    return (exception_types[0],) if exception_types is not None else ()
