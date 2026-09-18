from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Final, Protocol, cast  # noqa: TID251  # validates dynamically loaded native callables

import httpx

from litellm.llms.base_llm.ocr.transformation import OCRResponse
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
    input_sources: Mapping[str, str] | None = None


@dataclass(frozen=True, slots=True)
class OcrHostHelpers:
    """The Python callables the native OCR route answers with: the public response and
    exception constructors, and the timeout conversion for the caller's raw keyword."""

    response: Callable[[Mapping[str, object]], OCRResponse]
    map_failure: Callable[[Exception, LiteLLMOcrRequest, str], Exception]
    timeout_to_seconds: Callable[[float | httpx.Timeout | None], float | None]


class NativeOcr(Protocol):
    def __call__(
        self,
        request: LiteLLMOcrRequest,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
        helpers: OcrHostHelpers,
    ) -> OCRResponse: ...


class NativeAocr(Protocol):
    def __call__(
        self,
        request: LiteLLMOcrRequest,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
        helpers: OcrHostHelpers,
    ) -> Awaitable[OCRResponse]: ...


def _ocr_binding(value: object) -> NativeOcr | None:
    if not callable(value):
        return None
    return cast("NativeOcr", value)  # cast-ok: callable validated at the native binding boundary


def _aocr_binding(value: object) -> NativeAocr | None:
    if not callable(value):
        return None
    return cast("NativeAocr", value)  # cast-ok: callable validated at the native binding boundary


NATIVE_OCR: Final = NativeBinding("ocr", validate=_ocr_binding)
NATIVE_AOCR: Final = NativeBinding("aocr", validate=_aocr_binding)
