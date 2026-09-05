"""Thin Python wrapper for the native Rust OCR bridge."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Final, TypeVar

from . import configuration as _configuration
from .bindings import UNCHANGED, Unchanged
from .protocols import RustAocr, RustOcr
from .request import NativeOCRRequest, PreparedNativeCall, call_native
from .runtime import (
    BridgeErrorContext,
    EndpointDispatch,
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


def set_rust_ocr(
    *,
    ocr: RustOcr | None | Unchanged = UNCHANGED,
    aocr: RustAocr | None | Unchanged = UNCHANGED,
) -> None:
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


def dispatch_ocr(
    *,
    prepare: Callable[[], PreparedNativeCall[NativeOCRRequest]],
    fallback: Callable[[], ResultT],
    adapt: Callable[[Mapping[str, object]], ResultT],
    model: str,
    provider: str,
    eligible: bool,
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
    eligible: bool,
) -> ResultT:
    return await _OCR.ainvoke(
        prepare=prepare,
        call=call_native,
        fallback=fallback,
        adapt=adapt,
        error_context=BridgeErrorContext(provider=provider, model=model),
        eligible=eligible,
    )
