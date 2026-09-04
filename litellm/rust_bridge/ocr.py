"""Thin Python wrapper for the native Rust OCR bridge."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Final, Protocol

import httpx

from litellm.rust_bridge import configuration as _configuration
from litellm.rust_bridge.bindings import UNCHANGED, Unchanged
from litellm.rust_bridge.runtime import (
    BridgeErrorContext,
    RustEndpoint,
    async_none,
    identity,
)
from litellm.rust_bridge.timeouts import timeout_to_seconds as _timeout_to_seconds

rust_ocr_enabled = _configuration.rust_ocr_enabled
rust = _configuration.rust


@dataclass(frozen=True, slots=True)
class RustOCRRequest:
    model: str
    document: dict[str, object]
    api_key: str | None
    api_base: str | None
    custom_llm_provider: str | None
    extra_headers: dict[str, object] | None
    optional_params: dict[str, object]
    timeout: float | httpx.Timeout | None


class RustOcr(Protocol):
    def __call__(
        self,
        model: str,
        document: dict[str, object],
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: dict[str, object] | None,
        optional_params: dict[str, object],
        timeout_seconds: float | None,
    ) -> dict[str, object]:
        raise NotImplementedError


class RustAocr(Protocol):
    def __call__(
        self,
        model: str,
        document: dict[str, object],
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: dict[str, object] | None,
        optional_params: dict[str, object],
        timeout_seconds: float | None,
    ) -> Awaitable[dict[str, object]]:
        raise NotImplementedError


_OCR: Final[RustEndpoint[RustOcr, RustAocr]] = RustEndpoint.native(
    route="ocr",
    sync="ocr",
    asynchronous="aocr",
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


def ocr(
    *,
    prepare: Callable[[], RustOCRRequest],
    model: str,
    provider: str,
    request_override: bool | None,
    eligible: bool,
) -> Mapping[str, object] | None:
    return _OCR.invoke(
        call=lambda rust_ocr: _call_ocr(rust_ocr, prepare()),
        fallback=lambda: None,
        adapt=identity,
        context=BridgeErrorContext(provider=provider, model=model),
        request_override=request_override,
        eligible=eligible,
    )


async def aocr(
    *,
    prepare: Callable[[], RustOCRRequest],
    model: str,
    provider: str,
    request_override: bool | None,
    eligible: bool,
) -> Mapping[str, object] | None:
    return await _OCR.ainvoke(
        call=lambda rust_aocr: _call_aocr(rust_aocr, prepare()),
        fallback=async_none,
        adapt=identity,
        context=BridgeErrorContext(provider=provider, model=model),
        request_override=request_override,
        eligible=eligible,
    )


def _call_ocr(rust_ocr: RustOcr, request: RustOCRRequest) -> Mapping[str, object]:
    return rust_ocr(
        model=request.model,
        document=request.document,
        api_key=request.api_key,
        api_base=request.api_base,
        custom_llm_provider=request.custom_llm_provider,
        extra_headers=request.extra_headers,
        optional_params=request.optional_params,
        timeout_seconds=_timeout_to_seconds(request.timeout),
    )


def _call_aocr(rust_aocr: RustAocr, request: RustOCRRequest) -> Awaitable[Mapping[str, object]]:
    return rust_aocr(
        model=request.model,
        document=request.document,
        api_key=request.api_key,
        api_base=request.api_base,
        custom_llm_provider=request.custom_llm_provider,
        extra_headers=request.extra_headers,
        optional_params=request.optional_params,
        timeout_seconds=_timeout_to_seconds(request.timeout),
    )
