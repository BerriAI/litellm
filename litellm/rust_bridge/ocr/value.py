"""Native OCR bindings and response adaptation."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, cast  # noqa: TID251  # native extension exposes dynamically typed callables

import httpx

from litellm.llms.base_llm.ocr.transformation import PROVIDER_NATIVE_RESPONSE_KEY, OCRResponse
from litellm.rust_bridge.configuration import RouteName
from litellm.rust_bridge.ocr.types import RustAocr, RustOcr
from litellm.rust_bridge.route import NativeRoute
from litellm.rust_bridge.timeouts import timeout_to_seconds as _timeout_to_seconds


def _as_ocr(value: object) -> RustOcr | None:
    return cast(RustOcr, value) if callable(value) else None


def _as_aocr(value: object) -> RustAocr | None:
    return cast(RustAocr, value) if callable(value) else None


ROUTE: Final = NativeRoute(RouteName.OCR)
_OCR: Final = ROUTE.bind("ocr", validate=_as_ocr)
_AOCR: Final = ROUTE.bind("aocr", validate=_as_aocr)


def load_rust_ocr() -> RustOcr | None:
    return ROUTE.select(_OCR)


def load_rust_aocr() -> RustAocr | None:
    return ROUTE.select(_AOCR)


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
