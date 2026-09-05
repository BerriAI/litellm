"""Thin Python wrapper for the native Rust OCR bridge."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Final, TypeVar

from . import configuration as _configuration
from .protocols import RustAocr, RustOcr
from .request import NativeOCRRequest, PreparedNativeCall, call_native
from .runtime import (
    BridgeErrorContext,
    EndpointDispatch,
)

ResultT = TypeVar("ResultT")


_OCR: Final[EndpointDispatch[RustOcr, RustAocr]] = EndpointDispatch.native(
    route="ocr",
    sync=lambda native: native.ocr,
    asynchronous=lambda native: native.aocr,
    enabled=_configuration.rust_enabled,
)


def load_rust_ocr() -> RustOcr | None:
    return _OCR.sync.load()


def load_rust_aocr() -> RustAocr | None:
    return _OCR.asynchronous.load()


def supports_callback_adapter(*, asynchronous: bool = False) -> bool:
    binding = _OCR.asynchronous if asynchronous else _OCR.sync
    if binding.is_overridden():
        return False
    from litellm.rust_bridge import get_native_bridge
    from litellm.rust_bridge.loader import native_route_ready

    native: Final = get_native_bridge()
    return (
        native is not None
        and hasattr(native, "__python_callback_runtime__")
        and native_route_ready("ocr", frozenset({"callbacks"}))
    )


def dispatch_ocr(
    *,
    prepare: Callable[[], PreparedNativeCall[NativeOCRRequest]],
    fallback: Callable[[], ResultT],
    adapt: Callable[[Mapping[str, object]], ResultT],
    model: str,
    provider: str,
    eligible: bool = True,
    request_format: str | None = None,
) -> ResultT:
    return _OCR.invoke(
        prepare=prepare,
        call=call_native,
        fallback=fallback,
        adapt=adapt,
        error_context=BridgeErrorContext(provider=provider, model=model),
        eligible=eligible,
    )


async def adispatch_ocr(
    *,
    prepare: Callable[[], PreparedNativeCall[NativeOCRRequest]],
    fallback: Callable[[], Awaitable[ResultT]],
    adapt: Callable[[Mapping[str, object]], ResultT],
    model: str,
    provider: str,
    eligible: bool = True,
    request_format: str | None = None,
) -> ResultT:
    return await _OCR.ainvoke(
        prepare=prepare,
        call=call_native,
        fallback=fallback,
        adapt=adapt,
        error_context=BridgeErrorContext(provider=provider, model=model),
        eligible=eligible,
    )
