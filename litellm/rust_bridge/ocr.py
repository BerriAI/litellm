from __future__ import annotations

from collections.abc import Callable, Coroutine, Mapping
from types import MappingProxyType
from typing import Final, Protocol, cast  # noqa: TID251  # validates dynamically loaded native callables

from pydantic import TypeAdapter

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


def read_document(reader: Callable[[], object]) -> bytes:
    value: Final = reader()
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, bytes):
        return value
    raise TypeError(f"OCR file read must return bytes or str, got {type(value)}")


def build_response(response: Mapping[str, object]) -> OCRResponse:
    provider_native_response: Final = response.get(PROVIDER_NATIVE_RESPONSE_KEY)
    normalized: Final = OCRResponse.model_validate(
        MappingProxyType({key: value for key, value in response.items() if key != PROVIDER_NATIVE_RESPONSE_KEY})
    )
    if isinstance(provider_native_response, Mapping):
        normalized.set_provider_native_response(_NATIVE_RESPONSE.validate_python(provider_native_response))
    return normalized
