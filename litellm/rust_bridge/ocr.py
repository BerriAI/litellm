"""Thin Python wrapper for the native Rust OCR bridge."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Final, TypeVar

from . import configuration as _configuration
from .protocols import RustAocr, RustOcr, RustRouteDecline
from .request import NativeOCRRequest, PreparedNativeCall, call_native
from .runtime import (
    BridgeErrorContext,
    EndpointBinding,
    EndpointDispatch,
    assess_route,
)

ResultT = TypeVar("ResultT")


_OCR: Final[EndpointDispatch[RustOcr, RustAocr]] = EndpointDispatch.native(
    route="ocr",
    sync=lambda native: native.ocr,
    asynchronous=lambda native: native.aocr,
    enabled=_configuration.rust_enabled,
)


_PREFLIGHT: Final[EndpointBinding[RustRouteDecline]] = EndpointBinding.native(
    route="ocr",
    select=lambda native: native.ocr_decline,
    enabled=_configuration.rust_enabled,
)


def load_rust_ocr() -> RustOcr | None:
    return _OCR.sync.load()


def load_rust_aocr() -> RustAocr | None:
    return _OCR.asynchronous.load()


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
