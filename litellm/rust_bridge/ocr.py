from __future__ import annotations

from collections.abc import Coroutine, Mapping
from types import MappingProxyType
from typing import Final, Protocol, cast  # noqa: TID251  # validates dynamically loaded native callables

from pydantic import TypeAdapter

import litellm
from litellm.llms.base_llm.ocr.transformation import PROVIDER_NATIVE_RESPONSE_KEY, OCRResponse
from litellm.rust_bridge.bindings import NativeBinding


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


def map_failure(error: Exception, model: str, provider: str, kwargs: Mapping[str, object]) -> Exception:
    mapper: Final = cast(
        ExceptionMapper, litellm.exception_type
    )  # cast-ok: bounded adapter for the public exception mapper
    try:
        return mapper(
            model=model,
            custom_llm_provider=provider,
            original_exception=error,
            completion_kwargs=dict(kwargs),  # mutable-ok: exception mapper requires owned kwargs
            extra_kwargs=dict(kwargs),  # mutable-ok: exception mapper requires owned kwargs
        )
    except Exception as public_error:
        public_error.__context__ = error
        return public_error
