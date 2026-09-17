from __future__ import annotations

from collections.abc import Awaitable, Mapping, Sequence
from typing import Final, Protocol, cast  # noqa: TID251  # validates dynamically loaded native callables

import httpx
from pydantic import TypeAdapter, ValidationError

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

_UPSTREAM_ARGS: Final = TypeAdapter(tuple[int, str])
_UPSTREAM_HEADERS: Final = TypeAdapter(list[tuple[str, str]])


class UpstreamFailure(Exception):
    def __init__(self, response: httpx.Response, cause: Exception) -> None:
        super().__init__(str(cause))
        self.message: Final = str(cause)
        self.response: Final = response
        self.status_code: Final = response.status_code
        self.__cause__ = cause


def _upstream_failure(error: Exception) -> Exception:
    try:
        status, body = _UPSTREAM_ARGS.validate_python(error.args)
        headers: Final = _UPSTREAM_HEADERS.validate_python(getattr(error, "headers", None))
    except ValidationError:
        return error
    return UpstreamFailure(httpx.Response(status, content=body.encode(), headers=headers), error)


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
    original: Final = _upstream_failure(error)
    try:
        return mapper(
            model=model,
            custom_llm_provider=request_provider,
            original_exception=original,
            completion_kwargs=dict(arguments(request)),  # mutable-ok: exception mapper requires owned kwargs
            extra_kwargs=dict(request.kwargs),  # mutable-ok: exception mapper requires owned kwargs
        )
    except Exception as public_error:
        if isinstance(original, UpstreamFailure):
            public_error.response = original.response
            public_error.status_code = original.status_code
        public_error.__context__ = error
        return public_error
