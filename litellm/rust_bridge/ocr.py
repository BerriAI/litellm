"""Thin Python wrapper for the native Rust OCR bridge."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Final, Protocol, TypeVar, cast

import httpx

from . import configuration as _configuration
from .bindings import UNCHANGED, Unchanged
from .callback_adapters import ProviderLogging, ProviderLoggingAdapter, ProviderPreCall
from .callbacks import CallbackDecision
from .protocols import RustAocr, RustOcr, RustRouteDecline
from .request import (
    NativeOCRRequest,
    NativeRequestContext,
    NativeRequestOptions,
    PreparedNativeCall,
    call_native,
    provider_connection_params,
)
from .runtime import (
    BridgeErrorContext,
    EndpointBinding,
    EndpointDispatch,
    assess_route,
)
from .timeouts import timeout_to_seconds

rust_ocr_enabled = _configuration.rust_ocr_enabled
rust = _configuration.rust
ResultT = TypeVar("ResultT")


_OCR: Final[EndpointDispatch[RustOcr, RustAocr]] = EndpointDispatch.native(
    route="ocr",
    sync=lambda native: native.ocr,
    asynchronous=lambda native: native.aocr,
    enabled=_configuration.rust_ocr_enabled,
)


_PREFLIGHT: Final[EndpointBinding[RustRouteDecline]] = EndpointBinding.native(
    route="ocr",
    select=lambda native: native.ocr_decline,
    enabled=_configuration.rust_ocr_enabled,
)


def set_rust_ocr(
    *,
    ocr: RustOcr | None | Unchanged = UNCHANGED,
    aocr: RustAocr | None | Unchanged = UNCHANGED,
    decline: RustRouteDecline | None | Unchanged = UNCHANGED,
) -> None:
    if not isinstance(decline, Unchanged):
        if decline is None:
            _PREFLIGHT.reset()
        else:
            _PREFLIGHT.override(decline)
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


def load_rust_ocr() -> RustOcr | None:
    return _OCR.sync.load()


def load_rust_aocr() -> RustAocr | None:
    return _OCR.asynchronous.load()


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
    prepare: Callable[[], PreparedNativeCall[NativeOCRRequest]],
    fallback: Callable[[], ResultT],
    adapt: Callable[[Mapping[str, object]], ResultT],
    model: str,
    provider: str,
    eligible: bool = True,
    request_format: str | None = None,
) -> ResultT:
    return _OCR.invoke(
        prepare=prepare,
        call=call_native,
        fallback=fallback,
        adapt=adapt,
        error_context=BridgeErrorContext(provider=provider, model=model),
        eligible=eligible,
        preflight=lambda: assess_route(_PREFLIGHT, model, provider, request_format=request_format),
    )


async def adispatch_ocr(
    *,
    prepare: Callable[[], PreparedNativeCall[NativeOCRRequest]],
    fallback: Callable[[], Awaitable[ResultT]],
    adapt: Callable[[Mapping[str, object]], ResultT],
    model: str,
    provider: str,
    eligible: bool = True,
    request_format: str | None = None,
) -> ResultT:
    return await _OCR.ainvoke(
        prepare=prepare,
        call=call_native,
        fallback=fallback,
        adapt=adapt,
        error_context=BridgeErrorContext(provider=provider, model=model),
        eligible=eligible,
        preflight=lambda: assess_route(_PREFLIGHT, model, provider, request_format=request_format),
    )


def eligible(document: Mapping[str, object], kwargs: Mapping[str, object]) -> bool:
    import litellm

    return (
        isinstance(document, dict)
        and document.get("type") in ("document_url", "image_url")
        and litellm.secret_manager_client is None
        and kwargs.get("azure_credential") is None
        and kwargs.get("extra_query") is None
    )


def prepare_call(
    *,
    model: str,
    document: Mapping[str, object],
    api_key: str | None,
    api_base: str | None,
    timeout: float | httpx.Timeout | None,
    custom_llm_provider: str | None,
    extra_headers: Mapping[str, object] | None,
    kwargs: Mapping[str, object],
    asynchronous: bool,
) -> PreparedNativeCall[NativeOCRRequest]:
    import litellm
    from litellm.constants import request_timeout

    call_id: Final = kwargs.get("litellm_call_id")
    connection: Final = provider_connection_params(kwargs)
    connection.update(sdk_api_key=litellm.api_key, sdk_api_base=litellm.api_base)
    if litellm.enable_azure_ad_token_refresh:
        connection.setdefault("enable_azure_ad_token_refresh", True)
    return PreparedNativeCall(
        request=NativeOCRRequest(
            model=model,
            document=dict(document),  # mutable-ok: PyO3 extracts a concrete document dictionary
            optional_params=dict(kwargs),  # mutable-ok: PyO3 filters raw SDK keyword arguments before serialization
            options=NativeRequestOptions(
                api_key=api_key,
                api_base=api_base,
                custom_llm_provider=custom_llm_provider,
                extra_headers=extra_headers,
                timeout_seconds=timeout_to_seconds(timeout or request_timeout),
                provider_connection=connection,
            ),
        ),
        context=NativeRequestContext(litellm_call_id=call_id if isinstance(call_id, str) else None),
        callback_adapter=(
            OCRLoggingAdapter(
                cast(  # cast-ok: SDK decorator supplies the logging interface
                    OCRLogging, kwargs["litellm_logging_obj"]
                ),
                "OCR document processing",
                api_key,
                kwargs,
            )
            if supports_callback_adapter(asynchronous=asynchronous)
            else None
        ),
        auth_provider=kwargs.get("azure_ad_token_provider"),
    )


class OCRLogging(ProviderLogging, Protocol):
    def update_from_kwargs(
        self,
        *,
        kwargs: dict[str, object],  # mutable-ok: matches the logging implementation contract
        model: str,
        optional_params: dict[str, object],  # mutable-ok: matches the logging implementation contract
        litellm_params: dict[str, object],  # mutable-ok: matches the logging implementation contract
        custom_llm_provider: str,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class OCRLoggingAdapter(ProviderLoggingAdapter):
    logging_obj: OCRLogging
    kwargs: Mapping[str, object]

    def pre_call(self, payload: object, /) -> CallbackDecision:
        event: Final = ProviderPreCall.model_validate(payload)
        self.logging_obj.update_from_kwargs(
            kwargs=dict(self.kwargs),  # mutable-ok: logging updates its owned SDK argument dictionary
            model=event.model,
            optional_params={  # mutable-ok: logging stores mutable provider parameters
                key: value for key, value in event.request.items() if key not in ("model", "document")
            },
            litellm_params={  # mutable-ok: logging stores mutable call identity parameters
                "litellm_call_id": event.call_id,
                "api_base": event.api_base,
            },
            custom_llm_provider=event.provider,
        )
        return ProviderLoggingAdapter.pre_call(self, payload)
