from collections.abc import Coroutine, Mapping
from types import MappingProxyType
from typing import Final

import httpx

from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.rust_bridge import runtime
from litellm.rust_bridge.catalog import Route
from litellm.rust_bridge.dispatch import PublicDispatch
from litellm.rust_bridge.ocr.entrypoints import NATIVE_AOCR, NATIVE_OCR
from litellm.rust_bridge.public_call import Bind, native_call_hook, optional_str

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


def _binder(name: str) -> Bind:
    def bind(args: tuple[object, ...], kwargs: Mapping[str, object]) -> Mapping[str, object]:
        try:
            return _bind_request(*args, **kwargs)  # pyright: ignore[reportArgumentType]  # Python binds the public arguments before native validation
        except TypeError as error:
            raise TypeError(str(error).replace("_bind_request()", f"{name}()")) from None

    return bind


def _provider_prefix(fields: Mapping[str, object]) -> str | None:
    prefix, separator, _ = (optional_str(fields.get("model")) or "").partition("/")
    return optional_str(fields.get("custom_llm_provider")) or (prefix if separator else None)


_DISPATCH: Final = PublicDispatch(Route.OCR, bind=_binder("ocr"), internal_hop="aocr", provider=_provider_prefix)
_ADISPATCH: Final = PublicDispatch(Route.OCR, bind=_binder("aocr"), provider=_provider_prefix)


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
