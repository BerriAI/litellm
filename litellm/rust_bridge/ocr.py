"""Thin Python wrapper for the native Rust OCR bridge."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

import httpx

from litellm.rust_bridge import configuration as _configuration
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.runtime import (
    BridgeErrorContext,
    RustAttempt,
    aattempt_binding,
    attempt_binding,
    identity,
)
from litellm.rust_bridge.timeouts import timeout_to_seconds as _timeout_to_seconds

rust_ocr_enabled = _configuration.rust_ocr_enabled
rust = _configuration.rust


@runtime_checkable
class RustOcr(Protocol):
    def __call__(
        self,
        model: str,
        document: Mapping[str, object],
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: Mapping[str, object] | None,
        optional_params: Mapping[str, object],
        timeout_seconds: float | None,
    ) -> Mapping[str, object]:
        raise NotImplementedError


@runtime_checkable
class RustAocr(Protocol):
    def __call__(
        self,
        model: str,
        document: Mapping[str, object],
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: Mapping[str, object] | None,
        optional_params: Mapping[str, object],
        timeout_seconds: float | None,
    ) -> Awaitable[Mapping[str, object]]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class RustOCRRequest:
    model: str
    document: Mapping[str, object]
    api_key: str | None
    api_base: str | None
    custom_llm_provider: str | None
    extra_headers: Mapping[str, object] | None
    optional_params: Mapping[str, object]
    timeout: float | httpx.Timeout | None


class _Unset:
    pass


_UNSET: Final[_Unset] = _Unset()


def _validate_ocr(value: object) -> RustOcr | None:
    return value if isinstance(value, RustOcr) else None


def _validate_aocr(value: object) -> RustAocr | None:
    return value if isinstance(value, RustAocr) else None


_OCR: Final = NativeBinding[RustOcr]("ocr", validate=_validate_ocr)
_AOCR: Final = NativeBinding[RustAocr]("aocr", validate=_validate_aocr)


def set_rust_ocr(
    *,
    ocr: RustOcr | None | _Unset = _UNSET,
    aocr: RustAocr | None | _Unset = _UNSET,
) -> None:
    if not isinstance(ocr, _Unset):
        if ocr is None:
            _OCR.reset()
        else:
            _OCR.override(ocr)
    if not isinstance(aocr, _Unset):
        if aocr is None:
            _AOCR.reset()
        else:
            _AOCR.override(aocr)


def load_rust_ocr() -> RustOcr | None:
    return _OCR.load()


def load_rust_aocr() -> RustAocr | None:
    return _AOCR.load()


def attempt_ocr(
    *,
    prepare_request: Callable[[], RustOCRRequest],
    context: BridgeErrorContext,
) -> RustAttempt[Mapping[str, object]]:
    return attempt_binding(
        binding=_OCR,
        native_call=lambda rust_ocr: _call_ocr(rust_ocr, prepare_request()),
        adapt=identity,
        context=context,
    )


async def attempt_aocr(
    *,
    prepare_request: Callable[[], RustOCRRequest],
    context: BridgeErrorContext,
) -> RustAttempt[Mapping[str, object]]:
    return await aattempt_binding(
        binding=_AOCR,
        native_call=lambda rust_aocr: _call_aocr(rust_aocr, prepare_request()),
        adapt=identity,
        context=context,
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
