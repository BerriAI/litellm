from __future__ import annotations

from collections.abc import Awaitable, Mapping
from typing import Final, Protocol, cast  # noqa: TID251  # native extension callables have no Python stubs

import litellm
from litellm.constants import request_timeout
from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.configuration import rust_enabled
from litellm.rust_bridge.ocr import (
    LiteLLMOcrRequest,
    input_sources,
    map_error,
    optional_params,
    provider,
    supported,
)
from litellm.rust_bridge.timeouts import timeout_to_seconds
from litellm.secret_managers.main import get_secret_str


class NativeOcrLifecycle(Protocol):
    def __call__(self, kwargs: Mapping[str, object], asynchronous: bool) -> OCRResponse | Awaitable[OCRResponse]: ...


class ExceptionMapper(Protocol):
    def __call__(
        self,
        *,
        model: str,
        custom_llm_provider: str | None,
        original_exception: Exception,
        completion_kwargs: dict[str, object],
        extra_kwargs: dict[str, object],
    ) -> Exception: ...


def _binding(value: object) -> NativeOcrLifecycle | None:
    if not callable(value):
        return None
    return cast(NativeOcrLifecycle, value)  # cast-ok: callable validated at the native binding boundary


NATIVE_OCR_LIFECYCLE: Final = NativeBinding("_ocr_lifecycle", validate=_binding)


def select(request: LiteLLMOcrRequest) -> NativeOcrLifecycle | None:
    if not rust_enabled() or not supported(request):
        return None
    if litellm.cache is not None or request.kwargs.get("caching") or request.kwargs.get("aocr"):
        return None
    return NATIVE_OCR_LIFECYCLE.load()


def arguments(request: LiteLLMOcrRequest) -> Mapping[str, object]:
    return {  # mutable-ok: PyO3 receives an owned kwargs dict while preserving nested object identity
        **request.kwargs,
        "model": request.model,
        "document": request.document,
        "api_key": request.api_key,
        "api_base": request.api_base,
        "timeout": request.timeout,
        "custom_llm_provider": request.custom_llm_provider,
        "extra_headers": request.extra_headers,
    }


def wire_request(request: LiteLLMOcrRequest) -> Mapping[str, object]:
    from litellm.ocr.main import convert_file_document_to_url_document

    document: Final = (
        convert_file_document_to_url_document(
            dict(request.document)  # mutable-ok: legacy file conversion requires a concrete document dict
        )
        if request.document.get("type") == "file"
        else request.document
    )
    options: Final = optional_params(request, get_secret_str)
    return {  # mutable-ok: serde conversion requires a concrete Python dict
        "model": request.model,
        "document": document,
        "api_key": (request.api_key or get_secret_str("MISTRAL_API_KEY"))
        if provider(request) == "mistral"
        else request.api_key,
        "api_base": request.api_base,
        "custom_llm_provider": request.custom_llm_provider,
        "extra_headers": request.extra_headers,
        "optional_params": dict(options),  # mutable-ok: serde conversion requires a concrete options dict
        "input_sources": dict(  # mutable-ok: serde conversion requires concrete source metadata
            input_sources(request, options)
        ),
        "timeout_seconds": timeout_to_seconds(request.timeout or request_timeout),
    }


def map_failure(error: Exception, request: LiteLLMOcrRequest) -> Exception:
    mapped: Final = map_error(error, request)
    request_provider: Final = provider(request)
    mapper: Final = cast(  # cast-ok: bounded adapter for the legacy public exception mapper
        ExceptionMapper, litellm.exception_type
    )
    try:
        return mapper(
            model=request.model.removeprefix(f"{request_provider}/"),
            custom_llm_provider=request_provider,
            original_exception=mapped,
            completion_kwargs=dict(arguments(request)),  # mutable-ok: exception mapper requires owned kwargs
            extra_kwargs=dict(request.kwargs),  # mutable-ok: exception mapper requires owned kwargs
        )
    except Exception as public_error:
        return public_error
