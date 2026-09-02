"""Thin Python wrapper for the native Rust OCR bridge."""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Final, Protocol

import httpx

from litellm.rust_bridge import configuration as _configuration
from litellm.rust_bridge.bindings import UNSET, NativeBinding, Unset
from litellm.rust_bridge.runtime import (
    BridgeErrorContext,
    FallbackMode,
    ainvoke,
    async_none,
    identity,
    invoke,
)
from litellm.rust_bridge.timeouts import timeout_to_seconds as _timeout_to_seconds

rust_ocr_enabled = _configuration.rust_ocr_enabled
use_litellm_rust = _configuration.use_litellm_rust


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


_OCR: Final = NativeBinding[RustOcr]("ocr")
_AOCR: Final = NativeBinding[RustAocr]("aocr")


def set_rust_ocr(
    *,
    ocr: RustOcr | None | Unset = UNSET,
    aocr: RustAocr | None | Unset = UNSET,
) -> None:
    _OCR.update(ocr)
    _AOCR.update(aocr)


def load_rust_ocr() -> RustOcr | None:
    return _OCR.load()


def load_rust_aocr() -> RustAocr | None:
    return _AOCR.load()


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
    rust_ocr: Final = load_rust_ocr()
    native_call: Final = (
        None
        if rust_ocr is None
        else lambda: rust_ocr(
            model=model,
            document=document,
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            optional_params=optional_params,
            timeout_seconds=_timeout_to_seconds(timeout),
        )
    )
    return invoke(
        native_call=native_call,
        fallback=lambda: None,
        adapt=identity,
        mode=FallbackMode.PYTHON,
        context=BridgeErrorContext(route="ocr", provider=custom_llm_provider or "", model=model),
    ).value


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
    rust_aocr: Final = load_rust_aocr()
    native_call: Final = (
        None
        if rust_aocr is None
        else lambda: rust_aocr(
            model=model,
            document=document,
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            optional_params=optional_params,
            timeout_seconds=_timeout_to_seconds(timeout),
        )
    )
    return (
        await ainvoke(
            native_call=native_call,
            fallback=async_none,
            adapt=identity,
            mode=FallbackMode.PYTHON,
            context=BridgeErrorContext(route="ocr", provider=custom_llm_provider or "", model=model),
        )
    ).value
