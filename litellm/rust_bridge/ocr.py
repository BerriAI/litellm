"""Thin Python wrapper for the native Rust OCR bridge."""

from __future__ import annotations

from typing import Final

import httpx

from litellm.rust_bridge import configuration as _configuration
from litellm.rust_bridge.bindings import UNCHANGED, Unchanged
from litellm.rust_bridge.protocols import RustAocr, RustOcr
from litellm.rust_bridge.runtime import (
    BridgeErrorContext,
    EndpointDispatch,
    NativeErrorPolicy,
    always_enabled,
    async_none,
    identity,
)
from litellm.rust_bridge.timeouts import timeout_to_seconds as _timeout_to_seconds

rust_ocr_enabled = _configuration.rust_ocr_enabled
rust = _configuration.rust


_OCR: Final[EndpointDispatch[RustOcr, RustAocr]] = EndpointDispatch.native(
    route="ocr",
    sync=lambda native: native.ocr,
    asynchronous=lambda native: native.aocr,
    enabled=always_enabled,
    error_policy=NativeErrorPolicy.PROPAGATE,
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


def ocr(
    *,
    model: str,
    document: dict[str, object],
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str | None,
    extra_headers: dict[str, object] | None,
    optional_params: dict[str, object],
    timeout: float | httpx.Timeout | None,
) -> dict[str, object] | None:
    return _OCR.invoke(
        prepare=lambda: _timeout_to_seconds(timeout),
        call=lambda rust_ocr, timeout_seconds: rust_ocr(
            model=model,
            document=document,
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            optional_params=optional_params,
            timeout_seconds=timeout_seconds,
        ),
        fallback=lambda: None,
        adapt=identity,
        error_context=BridgeErrorContext(provider=custom_llm_provider or "", model=model),
    )


async def aocr(
    *,
    model: str,
    document: dict[str, object],
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str | None,
    extra_headers: dict[str, object] | None,
    optional_params: dict[str, object],
    timeout: float | httpx.Timeout | None,
) -> dict[str, object] | None:
    return await _OCR.ainvoke(
        prepare=lambda: _timeout_to_seconds(timeout),
        call=lambda rust_aocr, timeout_seconds: rust_aocr(
            model=model,
            document=document,
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            optional_params=optional_params,
            timeout_seconds=timeout_seconds,
        ),
        fallback=async_none,
        adapt=identity,
        error_context=BridgeErrorContext(provider=custom_llm_provider or "", model=model),
    )
