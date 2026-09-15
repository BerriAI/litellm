from __future__ import annotations

from collections.abc import Mapping
from typing import Final, Protocol

import litellm
from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.rust_bridge.ocr.types import LiteLLMOcrRequest
from litellm.rust_bridge.ocr.value import adapt_response
from litellm.types.utils import CustomPricingLiteLLMParams


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


class OcrLifecycleHost:
    def response(self, response: Mapping[str, object]) -> OCRResponse:
        return adapt_response(response)

    def custom_pricing_fields(self) -> tuple[str, ...]:
        return tuple(CustomPricingLiteLLMParams.model_fields)

    def map_failure(
        self,
        error: Exception,
        request: LiteLLMOcrRequest,
        request_provider: str,
    ) -> Exception:
        mapper: Final[ExceptionMapper] = litellm.exception_type  # pyright: ignore[reportAssignmentType]  # legacy public mapper is callable
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


HOST: Final = OcrLifecycleHost()
