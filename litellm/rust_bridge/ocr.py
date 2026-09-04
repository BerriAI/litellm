"""Thin Python wrapper for the native Rust OCR bridge."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

import httpx

from litellm.rust_bridge import configuration as _configuration
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.runtime import (
    RustAttempt,
    RustDeclined,
    RustUnavailable,
    acomplete,
    complete,
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


@runtime_checkable
class RustOcrDecline(Protocol):
    def __call__(
        self,
        model: str,
        custom_llm_provider: str | None,
        optional_params: Mapping[str, object],
    ) -> str | None:
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


@dataclass(frozen=True, slots=True)
class RustOCRCandidate:
    model: str
    custom_llm_provider: str | None
    optional_params: Mapping[str, object]


class _Unset:
    pass


_UNSET: Final[_Unset] = _Unset()


def _validate_ocr(value: object) -> RustOcr | None:
    return value if isinstance(value, RustOcr) else None


def _validate_aocr(value: object) -> RustAocr | None:
    return value if isinstance(value, RustAocr) else None


def _validate_decline(value: object) -> RustOcrDecline | None:
    return value if isinstance(value, RustOcrDecline) else None


_OCR: Final = NativeBinding[RustOcr]("ocr", validate=_validate_ocr)
_AOCR: Final = NativeBinding[RustAocr]("aocr", validate=_validate_aocr)
_OCR_DECLINE: Final = NativeBinding[RustOcrDecline]("ocr_decline", validate=_validate_decline)


def _accept_injected_ocr(
    model: str,
    custom_llm_provider: str | None,
    optional_params: Mapping[str, object],
) -> None:
    return None


def set_rust_ocr(
    *,
    ocr: RustOcr | None | _Unset = _UNSET,
    aocr: RustAocr | None | _Unset = _UNSET,
    decline: RustOcrDecline | None | _Unset = _UNSET,
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
    if not isinstance(decline, _Unset):
        if decline is None:
            _OCR_DECLINE.reset()
        else:
            _OCR_DECLINE.override(decline)
    elif not isinstance(ocr, _Unset) or not isinstance(aocr, _Unset):
        if ocr is None and aocr is None:
            _OCR_DECLINE.reset()
        else:
            _OCR_DECLINE.override(_accept_injected_ocr)


def load_rust_ocr() -> RustOcr | None:
    return _OCR.load()


def load_rust_aocr() -> RustAocr | None:
    return _AOCR.load()


def attempt_ocr(
    *,
    candidate: RustOCRCandidate,
    prepare_request: Callable[[], RustOCRRequest],
) -> RustAttempt[Mapping[str, object]]:
    rust_ocr: Final = _OCR.load()
    decline: Final = _OCR_DECLINE.load()
    if rust_ocr is None or decline is None:
        return RustUnavailable()
    reason: Final = decline(
        model=candidate.model,
        custom_llm_provider=candidate.custom_llm_provider,
        optional_params=candidate.optional_params,
    )
    if reason is not None:
        return RustDeclined(reason)
    request: Final = prepare_request()
    return complete(native_call=lambda: _call_ocr(rust_ocr, request), adapt=identity)


async def attempt_aocr(
    *,
    candidate: RustOCRCandidate,
    prepare_request: Callable[[], RustOCRRequest],
) -> RustAttempt[Mapping[str, object]]:
    rust_aocr: Final = _AOCR.load()
    decline: Final = _OCR_DECLINE.load()
    if rust_aocr is None or decline is None:
        return RustUnavailable()
    reason: Final = decline(
        model=candidate.model,
        custom_llm_provider=candidate.custom_llm_provider,
        optional_params=candidate.optional_params,
    )
    if reason is not None:
        return RustDeclined(reason)
    request: Final = prepare_request()
    return await acomplete(native_call=lambda: _call_aocr(rust_aocr, request), adapt=identity)


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
