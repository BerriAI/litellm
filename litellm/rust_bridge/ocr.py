"""Thin Python wrapper for the native Rust OCR bridge."""

from __future__ import annotations

from typing import Final

import httpx

from litellm.rust_bridge import configuration as _configuration
from litellm.rust_bridge.protocols import RustAocr, RustOcr
from litellm.rust_bridge.runtime import (
    BridgeErrorContext,
    EndpointDispatch,
    NativeErrorPolicy,
    async_none,
    identity,
)
from litellm.rust_bridge.timeouts import timeout_to_seconds as _timeout_to_seconds

_OCR: Final[EndpointDispatch[RustOcr, RustAocr]] = EndpointDispatch.native(
    route="ocr",
    sync=lambda native: native.ocr,
    asynchronous=lambda native: native.aocr,
    enabled=_configuration.rust_enabled,
    error_policy=NativeErrorPolicy.PROPAGATE,
)


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
