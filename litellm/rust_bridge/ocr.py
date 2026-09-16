from __future__ import annotations

from collections.abc import Coroutine, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Protocol, cast  # noqa: TID251  # validates dynamically loaded native callables

import httpx
from pydantic import TypeAdapter

import litellm
from litellm.llms.base_llm.ocr.transformation import PROVIDER_NATIVE_RESPONSE_KEY, OCRResponse
from litellm.rust_bridge.bindings import NativeBinding


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


def _bind_request(
    model: str,
    document: Mapping[str, object],
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: dict[str, object] | None = None,
    **kwargs: object,  # kwargs-ok: public OCR accepts provider-specific options
) -> LiteLLMOcrRequest:
    return LiteLLMOcrRequest(model, document, api_key, api_base, timeout, custom_llm_provider, extra_headers, kwargs)


def bind_request(name: str, args: tuple[object, ...], kwargs: dict[str, object]) -> LiteLLMOcrRequest:
    try:
        return _bind_request(*args, **kwargs)  # pyright: ignore[reportArgumentType]  # Python binds arguments before native validation
    except TypeError as error:
        raise TypeError(str(error).replace("_bind_request()", f"{name}()")) from None


class RustOcr(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> OCRResponse: ...


class RustAocr(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> Coroutine[object, object, OCRResponse]: ...


def _as_ocr(value: object) -> RustOcr | None:
    return cast(RustOcr, value) if callable(value) else None


def _as_aocr(value: object) -> RustAocr | None:
    return cast(RustAocr, value) if callable(value) else None


NATIVE_OCR: Final = NativeBinding("ocr", validate=_as_ocr)
NATIVE_AOCR: Final = NativeBinding("aocr", validate=_as_aocr)
_NATIVE_RESPONSE: Final = TypeAdapter(Mapping[str, object])


def build_response(response: Mapping[str, object]) -> OCRResponse:
    provider_native_response: Final = response.get(PROVIDER_NATIVE_RESPONSE_KEY)
    normalized: Final = OCRResponse.model_validate(
        MappingProxyType({key: value for key, value in response.items() if key != PROVIDER_NATIVE_RESPONSE_KEY})
    )
    if isinstance(provider_native_response, Mapping):
        normalized.set_provider_native_response(_NATIVE_RESPONSE.validate_python(provider_native_response))
    return normalized


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


def map_failure(error: Exception, request: LiteLLMOcrRequest, request_provider: str) -> Exception:
    mapper: Final = cast(
        ExceptionMapper, litellm.exception_type
    )  # cast-ok: bounded adapter for the public exception mapper
    try:
        return mapper(
            model=request.model.removeprefix(f"{request_provider}/"),
            custom_llm_provider=request_provider,
            original_exception=error,
            completion_kwargs=dict(request.kwargs),  # mutable-ok: exception mapper requires owned kwargs
            extra_kwargs=dict(request.kwargs),  # mutable-ok: exception mapper requires owned kwargs
        )
    except Exception as public_error:
        public_error.__context__ = error
        return public_error
