"""Thin Python wrapper for the native Rust OCR bridge."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Final, Protocol, cast  # noqa: TID251  # native extension exposes dynamically typed callables

import httpx

from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.timeouts import timeout_to_seconds as _timeout_to_seconds


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


def _as_ocr(value: object) -> RustOcr | None:
    return cast(RustOcr, value) if callable(value) else None


def _as_aocr(value: object) -> RustAocr | None:
    return cast(RustAocr, value) if callable(value) else None


_OCR: Final = NativeBinding("ocr", validate=_as_ocr)
_AOCR: Final = NativeBinding("aocr", validate=_as_aocr)


def load_rust_ocr() -> RustOcr | None:
    return _OCR.load()


def load_rust_aocr() -> RustAocr | None:
    return _AOCR.load()


def _public_ocr(arguments: dict[str, object]) -> object:
    import litellm

    public_ocr: Final = cast(  # cast-ok: the decorated public OCR callable accepts the retained argument bag
        Callable[..., object], litellm.ocr
    )
    return public_ocr(**arguments)


async def _public_aocr(arguments: dict[str, object]) -> object:
    import litellm

    public_aocr: Final = cast(  # cast-ok: the decorated public OCR callable accepts the retained argument bag
        Callable[..., Awaitable[object]], litellm.aocr
    )
    return await public_aocr(**arguments)


def ocr(
    arguments: dict[str, object] | None = None,
    *,
    model: str | None = None,
    document: dict[str, object] | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: dict[str, object] | None = None,
    optional_params: dict[str, object] | None = None,
    timeout: float | httpx.Timeout | None = None,
) -> object:
    if arguments is not None:
        return _public_ocr(arguments)
    if model is None or document is None or optional_params is None:
        raise TypeError("Native OCR requires model, document, and optional_params")
    rust_ocr: Final = load_rust_ocr()
    if rust_ocr is None:
        return None
    return rust_ocr(
        model=model,
        document=document,
        api_key=api_key,
        api_base=api_base,
        custom_llm_provider=custom_llm_provider,
        extra_headers=extra_headers,
        optional_params=optional_params,
        timeout_seconds=_timeout_to_seconds(timeout),
    )


async def aocr(
    arguments: dict[str, object] | None = None,
    *,
    model: str | None = None,
    document: dict[str, object] | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: dict[str, object] | None = None,
    optional_params: dict[str, object] | None = None,
    timeout: float | httpx.Timeout | None = None,
) -> object:
    if arguments is not None:
        return await _public_aocr(arguments)
    if model is None or document is None or optional_params is None:
        raise TypeError("Native OCR requires model, document, and optional_params")
    rust_aocr: Final = load_rust_aocr()
    if rust_aocr is None:
        return None
    return await rust_aocr(
        model=model,
        document=document,
        api_key=api_key,
        api_base=api_base,
        custom_llm_provider=custom_llm_provider,
        extra_headers=extra_headers,
        optional_params=optional_params,
        timeout_seconds=_timeout_to_seconds(timeout),
    )
