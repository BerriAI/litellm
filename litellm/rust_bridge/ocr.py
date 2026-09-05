"""Thin Python wrapper for the native Rust OCR bridge."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Final, TypeVar

from . import configuration as _configuration
from .bindings import UNCHANGED, Unchanged
from .protocols import RustAocr, RustOcr, RustRouteDecline
from .request import NativeOCRRequest, PreparedNativeCall, call_native
from .runtime import (
    BridgeErrorContext,
    EndpointBinding,
    EndpointDispatch,
    assess_route,
)

rust_ocr_enabled = _configuration.rust_ocr_enabled
rust = _configuration.rust
ResultT = TypeVar("ResultT")


_OCR: Final[EndpointDispatch[RustOcr, RustAocr]] = EndpointDispatch.native(
    route="ocr",
    sync=lambda native: native.ocr,
    asynchronous=lambda native: native.aocr,
    enabled=_configuration.rust_ocr_enabled,
)


_PREFLIGHT: Final[EndpointBinding[RustRouteDecline]] = EndpointBinding.native(
    route="ocr",
    select=lambda native: native.ocr_decline,
    enabled=_configuration.rust_ocr_enabled,
)


def set_rust_ocr(
    *,
    ocr: RustOcr | None | Unchanged = UNCHANGED,
    aocr: RustAocr | None | Unchanged = UNCHANGED,
    decline: RustRouteDecline | None | Unchanged = UNCHANGED,
) -> None:
    if not isinstance(decline, Unchanged):
        if decline is None:
            _PREFLIGHT.reset()
        else:
            _PREFLIGHT.override(decline)
    if not isinstance(ocr, Unchanged):
        if ocr is None:
            _OCR.sync.reset()
        else:
            _OCR.sync.override(ocr)
    if not isinstance(aocr, Unchanged):
        if aocr is None:
            _OCR.asynchronous.reset()
        else:
            _OCR.asynchronous.override(aocr)


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
        preflight=lambda: assess_route(_PREFLIGHT, model, provider, request_format=request_format),
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
        preflight=lambda: assess_route(_PREFLIGHT, model, provider, request_format=request_format),
    )
