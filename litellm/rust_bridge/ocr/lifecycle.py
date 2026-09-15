from __future__ import annotations

from collections.abc import Awaitable
from typing import Final, Protocol, cast  # noqa: TID251  # validates dynamically loaded native callables

from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.rust_bridge.bindings import BINDING_UNSET, BindingUnset
from litellm.rust_bridge.ocr.definition import COMPONENT
from litellm.rust_bridge.ocr.host import HOST
from litellm.rust_bridge.ocr.types import LiteLLMOcrRequest
from litellm.rust_bridge.route import ComponentExecution


class NativeOcr(Protocol):
    def __call__(
        self,
        request: LiteLLMOcrRequest,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        host: object,
    ) -> OCRResponse: ...


class NativeAocr(Protocol):
    def __call__(
        self,
        request: LiteLLMOcrRequest,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        host: object,
    ) -> Awaitable[OCRResponse]: ...


def _ocr_binding(value: object) -> NativeOcr | None:
    if not callable(value):
        return None
    return cast("NativeOcr", value)  # cast-ok: callable validated at the native binding boundary


def _aocr_binding(value: object) -> NativeAocr | None:
    if not callable(value):
        return None
    return cast("NativeAocr", value)  # cast-ok: callable validated at the native binding boundary


OCR: Final = COMPONENT.bind("ocr", validate=_ocr_binding)
AOCR: Final = COMPONENT.bind("aocr", validate=_aocr_binding)


class OcrLifecycleBindings:
    def override(self, value: object) -> None:
        OCR.override(_ocr_binding(value))
        AOCR.override(_aocr_binding(value))

    def reset(self) -> None:
        OCR.reset()
        AOCR.reset()


NATIVE_OCR_LIFECYCLE: Final = OcrLifecycleBindings()


def set_rust_ocr(
    *,
    ocr: NativeOcr | None | BindingUnset = BINDING_UNSET,
    aocr: NativeAocr | None | BindingUnset = BINDING_UNSET,
) -> None:
    OCR.configure(ocr)
    AOCR.configure(aocr)


def select_ocr(request: LiteLLMOcrRequest, execution: ComponentExecution) -> NativeOcr | None:
    return execution.select(OCR)


def select_aocr(request: LiteLLMOcrRequest, execution: ComponentExecution) -> NativeAocr | None:
    return execution.select(AOCR)


def map_failure(error: Exception, request: LiteLLMOcrRequest, request_provider: str) -> Exception:
    return HOST.map_failure(error, request, request_provider)
