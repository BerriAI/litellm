from __future__ import annotations

from collections.abc import Awaitable, Mapping
from typing import Final, Protocol, cast  # noqa: TID251  # validates dynamically loaded native callables

from litellm.llms.base_llm.ocr.transformation import DocumentType, OCRResponse
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.public_call import NativeCall


class NativeOcr(Protocol):
    def __call__(
        self,
        request: NativeCall,
    ) -> OCRResponse: ...


class NativeAocr(Protocol):
    def __call__(
        self,
        request: NativeCall,
    ) -> Awaitable[OCRResponse]: ...


def _ocr_binding(value: object) -> NativeOcr | None:
    if not callable(value):
        return None
    return cast("NativeOcr", value)  # cast-ok: callable validated at the native binding boundary


def _aocr_binding(value: object) -> NativeAocr | None:
    if not callable(value):
        return None
    return cast("NativeAocr", value)  # cast-ok: callable validated at the native binding boundary


class NativeOcrHealthCheckDocument(Protocol):
    def __call__(self, model: str, custom_llm_provider: str | None) -> DocumentType: ...


class NativeOcrPassthroughResponse(Protocol):
    def __call__(self, model: str, endpoint: str, body: bytes) -> Mapping[str, object] | None: ...


def _health_check_document_binding(value: object) -> NativeOcrHealthCheckDocument | None:
    if not callable(value):
        return None
    return cast("NativeOcrHealthCheckDocument", value)  # cast-ok: callable validated at the native binding boundary


def _passthrough_response_binding(value: object) -> NativeOcrPassthroughResponse | None:
    if not callable(value):
        return None
    return cast("NativeOcrPassthroughResponse", value)  # cast-ok: callable validated at the native binding boundary


NATIVE_OCR: Final = NativeBinding("ocr", validate=_ocr_binding)
NATIVE_AOCR: Final = NativeBinding("aocr", validate=_aocr_binding)
NATIVE_OCR_HEALTH_CHECK_DOCUMENT: Final = NativeBinding(
    "ocr_health_check_document", validate=_health_check_document_binding
)
NATIVE_OCR_PASSTHROUGH_RESPONSE: Final = NativeBinding(
    "ocr_passthrough_response", validate=_passthrough_response_binding
)
