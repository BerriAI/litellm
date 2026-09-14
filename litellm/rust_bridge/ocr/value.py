"""Native OCR bindings and response adaptation."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, cast  # noqa: TID251  # native extension exposes dynamically typed callables

import httpx

from litellm.llms.base_llm.ocr.transformation import PROVIDER_NATIVE_RESPONSE_KEY, OCRResponse
from litellm.rust_bridge.ocr.definition import COMPONENT
from litellm.rust_bridge.ocr.types import RustAocr, RustOcr
from litellm.rust_bridge.runtime import BridgeErrorContext, ainvoke, invoke
from litellm.rust_bridge.timeouts import timeout_to_seconds as _timeout_to_seconds


def _as_ocr(value: object) -> RustOcr | None:
    return cast(RustOcr, value) if callable(value) else None


def _as_aocr(value: object) -> RustAocr | None:
    return cast(RustAocr, value) if callable(value) else None


_OCR: Final = COMPONENT.bind("ocr", validate=_as_ocr)
_AOCR: Final = COMPONENT.bind("aocr", validate=_as_aocr)


def load_rust_ocr() -> RustOcr | None:
    return COMPONENT.resolve().select(_OCR)


def load_rust_aocr() -> RustAocr | None:
    return COMPONENT.resolve().select(_AOCR)


def adapt_response(response: Mapping[str, object]) -> OCRResponse:
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
    execution: Final = COMPONENT.resolve()
    rust_ocr: Final = execution.select(_OCR)
    return invoke(
        execution=execution,
        native_call=(
            lambda: rust_ocr(
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
        )
        if rust_ocr is not None
        else None,
        python_fallback=lambda: None,
        adapt=lambda value: value,
        context=BridgeErrorContext(route=COMPONENT.name.value, provider=custom_llm_provider or "", model=model),
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
    execution: Final = COMPONENT.resolve()
    rust_aocr: Final = execution.select(_AOCR)
    async def python_fallback() -> None:
        return None

    return await ainvoke(
        execution=execution,
        native_call=(
            lambda: rust_aocr(
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
        )
        if rust_aocr is not None
        else None,
        python_fallback=python_fallback,
        adapt=lambda value: value,
        context=BridgeErrorContext(route=COMPONENT.name.value, provider=custom_llm_provider or "", model=model),
    )
