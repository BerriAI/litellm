from collections.abc import Coroutine, Mapping
from typing import Final

import httpx

from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.rust_bridge import runtime
from litellm.rust_bridge.catalog import Route, RouteContext
from litellm.rust_bridge.dispatch import PublicDispatch, call_hook
from litellm.rust_bridge.ocr.entrypoints import NATIVE_AOCR, NATIVE_OCR, LiteLLMOcrRequest

__all__ = ("aocr", "ocr")


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


def _context(request: LiteLLMOcrRequest) -> RouteContext:
    prefix, separator, _ = request.model.partition("/")
    provider: Final = request.custom_llm_provider or (prefix if separator else None)
    return RouteContext(Route.OCR, provider=provider, model=request.model)


_DISPATCH: Final = PublicDispatch(
    route=Route.OCR,
    request=lambda args, kwargs: _public_request("ocr", args, kwargs),
    context=_context,
    bypass=lambda request: request.kwargs.get("aocr") is True,
)

_ADISPATCH: Final = PublicDispatch(
    route=Route.OCR,
    request=lambda args, kwargs: _public_request("aocr", args, kwargs),
    context=_context,
)


def ocr(
    *args: object,
    **kwargs: object,  # kwargs-ok: preserve the public OCR call shape
) -> OCRResponse | Coroutine[object, object, OCRResponse]:
    return _DISPATCH.run(
        args,
        kwargs,
        python=runtime.NO_PYTHON,
        binding=NATIVE_OCR,
        native=call_hook,
    )


async def aocr(*args: object, **kwargs: object) -> OCRResponse:  # kwargs-ok: preserve the public OCR call shape
    return await _ADISPATCH.arun(
        args,
        kwargs,
        python=runtime.NO_PYTHON,
        binding=NATIVE_AOCR,
        native=call_hook,
    )
