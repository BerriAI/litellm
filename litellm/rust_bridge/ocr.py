"""Thin Python wrapper for the native Rust OCR bridge."""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Final, Protocol, cast  # noqa: TID251  # native extension exposes dynamically typed callables

from litellm.rust_bridge.bindings import NativeBinding


class RustOcr(Protocol):
    def __call__(self, request: dict[str, object]) -> dict[str, object]:
        raise NotImplementedError


class RustAocr(Protocol):
    def __call__(self, request: dict[str, object]) -> Awaitable[dict[str, object]]:
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


def ocr(*, request: dict[str, object]) -> dict[str, object] | None:
    rust_ocr: Final = load_rust_ocr()
    return None if rust_ocr is None else rust_ocr(request=request)


async def aocr(*, request: dict[str, object]) -> dict[str, object] | None:
    rust_aocr: Final = load_rust_aocr()
    return None if rust_aocr is None else await rust_aocr(request=request)
