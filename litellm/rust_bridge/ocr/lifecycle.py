from __future__ import annotations

from typing import Final, cast  # noqa: TID251  # validates dynamically loaded native callables

from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.rust_bridge.ocr.definition import COMPONENT
from litellm.rust_bridge.ocr.host import HOST
from litellm.rust_bridge.ocr.types import LiteLLMOcrRequest
from litellm.rust_bridge.route import ComponentExecution, NativeLifecycle

NativeOcrLifecycle = NativeLifecycle[LiteLLMOcrRequest, OCRResponse]


def _binding(value: object) -> NativeOcrLifecycle | None:
    if not callable(value):
        return None
    return cast("NativeOcrLifecycle", value)  # cast-ok: callable validated at the native binding boundary


LIFECYCLE: Final = COMPONENT.bind("_ocr_lifecycle", validate=_binding)
NATIVE_OCR_LIFECYCLE: Final = LIFECYCLE


def select(request: LiteLLMOcrRequest, execution: ComponentExecution) -> NativeOcrLifecycle | None:
    if request.kwargs.get("aocr"):
        return None
    return execution.select(NATIVE_OCR_LIFECYCLE)


def map_failure(error: Exception, request: LiteLLMOcrRequest, request_provider: str) -> Exception:
    return HOST.map_failure(error, request, request_provider)
