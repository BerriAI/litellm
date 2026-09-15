from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Protocol, cast  # noqa: TID251  # public exception mapper is dynamically typed

import litellm
from litellm.llms.base_llm.ocr.transformation import PROVIDER_NATIVE_RESPONSE_KEY, OCRResponse
from litellm.rust_bridge.ocr.types import LiteLLMOcrRequest
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
        provider_native_response: Final = response.get(PROVIDER_NATIVE_RESPONSE_KEY)
        normalized: Final = OCRResponse.model_validate(
            MappingProxyType(
                {  # mutable-ok: immediately frozen before model validation
                    key: value for key, value in response.items() if key != PROVIDER_NATIVE_RESPONSE_KEY
                }
            )
        )
        if isinstance(provider_native_response, Mapping):
            native_mapping: Final = cast(  # cast-ok: Mapping runtime check precedes narrowing
                Mapping[str, object], provider_native_response
            )
            normalized.set_provider_native_response(
                dict(native_mapping)  # mutable-ok: response API retains an owned provider payload
            )
        return normalized

    def custom_pricing_fields(self) -> tuple[str, ...]:
        return tuple(CustomPricingLiteLLMParams.model_fields)

    def map_failure(
        self,
        error: Exception,
        request: LiteLLMOcrRequest,
        request_provider: str,
    ) -> Exception:
        mapper: Final = cast(  # cast-ok: legacy public mapper is callable
            ExceptionMapper,
            litellm.exception_type,  # pyright: ignore[reportUnknownMemberType]  # dynamically exported mapper
        )
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
