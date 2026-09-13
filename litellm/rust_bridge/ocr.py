"""Thin Python wrapper for the native Rust OCR bridge."""

from __future__ import annotations

from collections.abc import Awaitable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Protocol, cast  # noqa: TID251  # native extension exposes dynamically typed callables

import httpx

from litellm.llms.base_llm.ocr.transformation import PROVIDER_NATIVE_RESPONSE_KEY, OCRResponse
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.timeouts import timeout_to_seconds as _timeout_to_seconds


@dataclass(frozen=True, slots=True)
class LiteLLMOcrRequest:
    model: str
    document: Mapping[str, object]
    api_key: str | None
    api_base: str | None
    timeout: float | httpx.Timeout | None
    custom_llm_provider: str | None
    extra_headers: dict[str, object] | None
    kwargs: Mapping[str, object]
    input_sources: Mapping[str, str] | None = None


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
        input_sources: dict[str, str],
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
        input_sources: dict[str, str],
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


def _response(response: Mapping[str, object]) -> OCRResponse:
    provider_native_response: Final = response.get(PROVIDER_NATIVE_RESPONSE_KEY)
    normalized: Final = OCRResponse.model_validate(
        MappingProxyType({key: value for key, value in response.items() if key != PROVIDER_NATIVE_RESPONSE_KEY})
    )
    if isinstance(provider_native_response, Mapping):
        normalized.set_provider_native_response(provider_native_response)
    return normalized


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
    input_sources: Mapping[str, str] | None = None,
) -> dict[str, object] | None:
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
        input_sources=dict(input_sources or {}),  # mutable-ok: native boundary requires a concrete dict
        timeout_seconds=_timeout_to_seconds(timeout),
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
    input_sources: Mapping[str, str] | None = None,
) -> dict[str, object] | None:
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
        input_sources=dict(input_sources or {}),  # mutable-ok: native boundary requires a concrete dict
        timeout_seconds=_timeout_to_seconds(timeout),
    )
