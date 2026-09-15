from collections.abc import Awaitable, Callable, Coroutine, Mapping
from typing import Final, cast  # noqa: TID251  # native binding selects a sync result or an async awaitable

import httpx

from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.ocr import legacy
from litellm.ocr.input import convert_file_document_to_url_document, get_mime_type
from litellm.rust_bridge.ocr import LiteLLMOcrRequest
from litellm.rust_bridge.ocr.definition import COMPONENT
from litellm.rust_bridge.ocr.host import HOST
from litellm.rust_bridge.ocr.lifecycle import select
from litellm.rust_bridge.runtime import BridgeErrorContext, ainvoke, invoke

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
    execution: Final = COMPONENT.resolve()
    native: Final = select(request, execution)
    fallback: Final = cast(  # cast-ok: forward the original call shape through the legacy @client decorator
        Callable[..., OCRResponse | Coroutine[object, object, OCRResponse]], legacy.ocr
    )
    native_call: Final[Callable[[], OCRResponse] | None] = (
        (lambda: native(request, args, kwargs, False, HOST)) if native is not None else None
    )
    return invoke(
        execution=execution,
        native_call=native_call,
        python_fallback=lambda: fallback(*args, **kwargs),
        adapt=lambda value: value,
        context=BridgeErrorContext(
            route=COMPONENT.name.value,
            provider=request.custom_llm_provider or "",
            model=request.model,
        ),
    )


async def aocr(*args: object, **kwargs: object) -> OCRResponse:  # kwargs-ok: preserve the public OCR call shape
    request: Final = _public_request("aocr", args, kwargs)
    execution: Final = COMPONENT.resolve()
    native: Final = select(request, execution)
    fallback: Final = cast(  # cast-ok: forward the original call shape through the legacy @client decorator
        Callable[..., Awaitable[OCRResponse]], legacy.aocr
    )
    native_call: Final[Callable[[], Awaitable[OCRResponse]] | None] = (
        (lambda: native(request, args, kwargs, True, HOST)) if native is not None else None
    )

    async def python_fallback() -> OCRResponse:
        return await fallback(*args, **kwargs)

    async def adapt(value: OCRResponse) -> OCRResponse:
        return value

    return await ainvoke(
        execution=execution,
        native_call=native_call,
        python_fallback=python_fallback,
        adapt=adapt,
        context=BridgeErrorContext(
            route=COMPONENT.name.value,
            provider=request.custom_llm_provider or "",
            model=request.model,
        ),
    )
