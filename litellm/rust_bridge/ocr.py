"""Thin Python wrapper for the native Rust OCR bridge."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Final, Protocol, TypeVar, cast  # noqa: TID251  # runtime typing constructs

import httpx

from . import configuration as _configuration
from .bindings import UNCHANGED, Unchanged
from .callbacks import OneShotCallbackHandle
from .runtime import (
    BridgeErrorContext,
    EndpointDispatch,
)  # cast-ok: generic classmethod cannot preserve the route Protocol parameters
from .timeouts import timeout_to_seconds as _timeout_to_seconds

rust_ocr_enabled = _configuration.rust_ocr_enabled
rust = _configuration.rust
ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class NativeOCRRequest:
    model: str
    document: dict[str, object]
    api_key: str | None
    api_base: str | None
    custom_llm_provider: str | None
    extra_headers: dict[str, object] | None
    optional_params: dict[str, object]
    timeout: float | httpx.Timeout | None
    litellm_call_id: str | None = None
    callback_adapter: OneShotCallbackHandle | None = None


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
        litellm_call_id: str | None,
        callback_adapter: OneShotCallbackHandle | None,
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
        litellm_call_id: str | None,
        callback_adapter: OneShotCallbackHandle | None,
    ) -> Awaitable[dict[str, object]]:
        raise NotImplementedError


_OCR: Final = cast(  # cast-ok: generic classmethod loses the route Protocol parameters
    EndpointDispatch[RustOcr, RustAocr],
    EndpointDispatch.native(
        route="ocr",
        sync="ocr",
        asynchronous="aocr",
        enabled=_configuration.rust_ocr_enabled,
    ),
)


def set_rust_ocr(
    *,
    sync: RustOcr | None | Unchanged = UNCHANGED,
    asynchronous: RustAocr | None | Unchanged = UNCHANGED,
) -> None:
    if not isinstance(sync, Unchanged):
        if sync is None:
            _OCR.sync.reset()
        else:
            _OCR.sync.override(sync)
    if not isinstance(asynchronous, Unchanged):
        if asynchronous is None:
            _OCR.asynchronous.reset()
        else:
            _OCR.asynchronous.override(asynchronous)


def supports_callback_adapter(*, asynchronous: bool = False) -> bool:
    binding = _OCR.asynchronous if asynchronous else _OCR.sync
    if binding.is_overridden():
        return False
    from litellm.rust_bridge import get_native_bridge
    from litellm.rust_bridge.loader import native_route_ready

    native: Final = get_native_bridge()
    return (
        native is not None
        and hasattr(native, "__python_callback_runtime__")
        and native_route_ready("ocr", frozenset({"callbacks"}))
    )


def dispatch_ocr(
    *,
    prepare: Callable[[], NativeOCRRequest],
    fallback: Callable[[], ResultT],
    adapt: Callable[[Mapping[str, object]], ResultT],
    model: str,
    provider: str,
    request_override: bool | None,
    eligible: bool,
) -> ResultT:
    return _OCR.invoke(
        prepare=prepare,
        call=_call_ocr,
        fallback=fallback,
        adapt=adapt,
        context=BridgeErrorContext(provider=provider, model=model),
        request_override=request_override,
        eligible=eligible,
    )


async def adispatch_ocr(
    *,
    prepare: Callable[[], NativeOCRRequest],
    fallback: Callable[[], Awaitable[ResultT]],
    adapt: Callable[[Mapping[str, object]], ResultT],
    model: str,
    provider: str,
    request_override: bool | None,
    eligible: bool,
) -> ResultT:
    return await _OCR.ainvoke(
        prepare=prepare,
        call=_call_aocr,
        fallback=fallback,
        adapt=adapt,
        context=BridgeErrorContext(provider=provider, model=model),
        request_override=request_override,
        eligible=eligible,
    )


def _call_ocr(rust_ocr: RustOcr, request: NativeOCRRequest) -> Mapping[str, object]:
    return rust_ocr(
        model=request.model,
        document=request.document,
        api_key=request.api_key,
        api_base=request.api_base,
        custom_llm_provider=request.custom_llm_provider,
        extra_headers=request.extra_headers,
        optional_params=request.optional_params,
        timeout_seconds=_timeout_to_seconds(request.timeout),
        litellm_call_id=request.litellm_call_id,
        callback_adapter=request.callback_adapter,
    )


def _call_aocr(rust_aocr: RustAocr, request: NativeOCRRequest) -> Awaitable[Mapping[str, object]]:
    return rust_aocr(
        model=request.model,
        document=request.document,
        api_key=request.api_key,
        api_base=request.api_base,
        custom_llm_provider=request.custom_llm_provider,
        extra_headers=request.extra_headers,
        optional_params=request.optional_params,
        timeout_seconds=_timeout_to_seconds(request.timeout),
        litellm_call_id=request.litellm_call_id,
        callback_adapter=request.callback_adapter,
    )
