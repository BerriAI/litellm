from collections.abc import Awaitable, Callable, Coroutine
from typing import Final, cast  # noqa: TID251  # native binding selects a sync result or an async awaitable

from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.ocr import legacy
from litellm.ocr.legacy import convert_file_document_to_url_document, get_mime_type
from litellm.rust_bridge.bindings import native_exception_types
from litellm.rust_bridge.configuration import rust_ocr_enabled
from litellm.rust_bridge.ocr import NATIVE_AOCR, NATIVE_OCR

__all__ = ("aocr", "convert_file_document_to_url_document", "get_mime_type", "ocr")


def ocr(
    *args: object,
    **kwargs: object,  # kwargs-ok: preserve the public OCR call shape
) -> OCRResponse | Coroutine[object, object, OCRResponse]:
    native: Final = NATIVE_OCR.load() if rust_ocr_enabled() and not kwargs.get("aocr") else None
    if native is not None:
        try:
            return native(*args, **kwargs)
        except _decline_types():
            pass
    fallback: Final = cast(  # cast-ok: forward the original call shape through the legacy @client decorator
        Callable[..., OCRResponse | Coroutine[object, object, OCRResponse]], legacy.ocr
    )
    return fallback(*args, **kwargs)


async def aocr(*args: object, **kwargs: object) -> OCRResponse:  # kwargs-ok: preserve the public OCR call shape
    native: Final = NATIVE_AOCR.load() if rust_ocr_enabled() and not kwargs.get("aocr") else None
    if native is not None:
        try:
            return await native(*args, **kwargs)
        except _decline_types():
            pass
    fallback: Final = cast(  # cast-ok: forward the original call shape through the legacy @client decorator
        Callable[..., Awaitable[OCRResponse]], legacy.aocr
    )
    return await fallback(*args, **kwargs)


def _decline_types() -> tuple[type[BaseException], ...]:
    exception_types: Final = native_exception_types()
    return (exception_types[0],) if exception_types is not None else ()
