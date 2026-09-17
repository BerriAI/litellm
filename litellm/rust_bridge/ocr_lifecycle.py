from __future__ import annotations

from collections.abc import Awaitable, Mapping, Sequence
from typing import Final, Protocol, cast  # noqa: TID251  # validates dynamically loaded native callables

import httpx

import litellm
from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.ocr import LiteLLMOcrRequest


class NativeOcrLifecycle(Protocol):
    def __call__(
        self,
        request: LiteLLMOcrRequest,
        args: Sequence[object],
        kwargs: Mapping[str, object],
        asynchronous: bool,
    ) -> OCRResponse | Awaitable[OCRResponse]: ...


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
    return cast("NativeOcrLifecycle", value)  # cast-ok: callable validated at the native binding boundary


NATIVE_OCR_LIFECYCLE: Final = NativeBinding("_ocr_lifecycle", validate=_binding)


def select(request: LiteLLMOcrRequest) -> NativeOcrLifecycle | None:
    if request.kwargs.get("aocr"):
        return None
    return NATIVE_OCR_LIFECYCLE.load()


def arguments(request: LiteLLMOcrRequest) -> Mapping[str, object]:
    return request.kwargs


def map_failure(error: Exception, request: LiteLLMOcrRequest, request_provider: str) -> Exception:
    model: Final = request.model.removeprefix(f"{request_provider}/")
    if getattr(error, "ocr_request_format_error", False):
        return litellm.UnsupportedParamsError(
            message=f"Invalid `req_format`: {request.kwargs.get('req_format')!r}. Expected 'native' or 'litellm'.",
            model=model,
            llm_provider=request_provider,
        )
    mapper: Final = cast(  # cast-ok: bounded adapter for the legacy public exception mapper
        ExceptionMapper, litellm.exception_type
    )
    try:
        return mapper(
            model=model,
            custom_llm_provider=request_provider,
            original_exception=error,
            completion_kwargs=dict(arguments(request)),  # mutable-ok: exception mapper requires owned kwargs
            extra_kwargs=dict(request.kwargs),  # mutable-ok: exception mapper requires owned kwargs
        )
    except Exception as public_error:
        response: Final = getattr(error, "response", None)
        if isinstance(response, httpx.Response):
            public_error.response = response
            public_error.status_code = response.status_code
        public_error.__context__ = error
        return public_error
