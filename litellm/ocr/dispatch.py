from collections.abc import Coroutine, Mapping
from types import MappingProxyType
from typing import Final

import httpx

from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.rust_bridge import runtime
from litellm.rust_bridge.catalog import Route, RouteContext
from litellm.rust_bridge.dispatch import PublicDispatch
from litellm.rust_bridge.ocr.entrypoints import NATIVE_AOCR, NATIVE_OCR
from litellm.rust_bridge.public_call import NativeCall, native_call, native_call_hook, optional_str

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
) -> Mapping[str, object]:
    return MappingProxyType(
        {
            "model": model,
            "document": document,
            "api_key": api_key,
            "api_base": api_base,
            "timeout": timeout,
            "custom_llm_provider": custom_llm_provider,
            "extra_headers": extra_headers,
            "kwargs": kwargs,
        }
    )


def _public_request(name: str, args: tuple[object, ...], kwargs: Mapping[str, object]) -> NativeCall:
    try:
        fields: Final = _bind_request(*args, **kwargs)  # pyright: ignore[reportArgumentType]  # Python binds the public arguments before native validation
        return native_call(args, kwargs, fields)
    except TypeError as error:
        raise TypeError(str(error).replace("_bind_request()", f"{name}()")) from None


def _context(request: NativeCall) -> RouteContext:
    prefix, separator, _ = str(request.bound["model"]).partition("/")
    provider: Final = optional_str(request.bound.get("custom_llm_provider")) or (prefix if separator else None)
    return RouteContext(Route.OCR, provider=provider, model=str(request.bound["model"]))


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
        native=native_call_hook,
    )


async def aocr(*args: object, **kwargs: object) -> OCRResponse:  # kwargs-ok: preserve the public OCR call shape
    return await _ADISPATCH.arun(
        args,
        kwargs,
        python=runtime.NO_PYTHON,
        binding=NATIVE_AOCR,
        native=native_call_hook,
    )
