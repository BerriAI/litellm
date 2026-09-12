from collections.abc import Awaitable, Coroutine, Mapping
from typing import Final, cast  # noqa: TID251  # native binding selects a sync result or an async awaitable

import httpx

from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.ocr import legacy
from litellm.ocr.legacy import convert_file_document_to_url_document, get_mime_type
from litellm.rust_bridge.ocr import LiteLLMOcrRequest
from litellm.rust_bridge.ocr_lifecycle import arguments, select

__all__ = ("aocr", "convert_file_document_to_url_document", "get_mime_type", "ocr")


def ocr(
    model: str,
    document: Mapping[str, object],
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: dict[str, object] | None = None,
    **kwargs: object,  # kwargs-ok: public OCR accepts provider-specific options
) -> OCRResponse | Coroutine[object, object, OCRResponse]:
    request: Final = LiteLLMOcrRequest(
        model, document, api_key, api_base, timeout, custom_llm_provider, extra_headers, kwargs
    )
    native: Final = select(request)
    if native is not None:
        return cast(  # cast-ok: False selects the native synchronous signature
            OCRResponse, native(arguments(request), False)
        )
    return legacy.ocr(model, document, api_key, api_base, timeout, custom_llm_provider, extra_headers, **kwargs)


async def aocr(
    model: str,
    document: Mapping[str, object],
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: dict[str, object] | None = None,
    **kwargs: object,  # kwargs-ok: public OCR accepts provider-specific options
) -> OCRResponse:
    request: Final = LiteLLMOcrRequest(
        model, document, api_key, api_base, timeout, custom_llm_provider, extra_headers, kwargs
    )
    native: Final = select(request)
    if native is not None:
        return await cast(  # cast-ok: True selects the native asynchronous signature
            Awaitable[OCRResponse], native(arguments(request), True)
        )
    return await legacy.aocr(model, document, api_key, api_base, timeout, custom_llm_provider, extra_headers, **kwargs)
