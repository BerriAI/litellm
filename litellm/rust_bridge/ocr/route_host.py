from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter

import litellm
from litellm.llms.base_llm.ocr.transformation import PROVIDER_NATIVE_RESPONSE_KEY, OCRResponse
from litellm.ocr.main import resolve_error_provider
from litellm.rust_bridge import failures
from litellm.rust_bridge.failures import UpstreamFailure
from litellm.rust_bridge.ocr.entrypoints import LiteLLMOcrRequest

__all__ = ("UpstreamFailure", "arguments", "map_failure", "response")

_RESPONSE_ADAPTER: Final = TypeAdapter(dict[str, object])


def response(value: Mapping[str, object]) -> OCRResponse:
    provider_native_response: Final = value.get(PROVIDER_NATIVE_RESPONSE_KEY)
    normalized: Final = OCRResponse.model_validate(
        MappingProxyType({key: item for key, item in value.items() if key != PROVIDER_NATIVE_RESPONSE_KEY})
    )
    if isinstance(provider_native_response, Mapping):
        normalized.set_provider_native_response(_RESPONSE_ADAPTER.validate_python(provider_native_response))
    return normalized


def arguments(request: LiteLLMOcrRequest) -> Mapping[str, object]:
    return request.kwargs


def map_failure(error: Exception, request: LiteLLMOcrRequest, projected_provider: str) -> Exception:
    request_provider: Final = projected_provider or resolve_error_provider(request.model, request.custom_llm_provider)
    if getattr(error, "ocr_request_format_error", False):
        return litellm.UnsupportedParamsError(
            message=f"Invalid `req_format`: {request.kwargs.get('req_format')!r}. Expected one of litellm, native.",
            model=request.model.removeprefix(f"{request_provider}/") if request_provider else request.model,
            llm_provider=request_provider or "",
        )
    invalid_provider: Final = getattr(error, "ocr_invalid_provider", None)
    if isinstance(invalid_provider, str):
        return _invalid_provider_failure(invalid_provider, request)
    return failures.map_native_failure(error, request.model, request_provider, arguments(request), request.api_base)


def _invalid_provider_failure(provider: str, request: LiteLLMOcrRequest) -> Exception:
    """Python rejects a LiteLLM provider without OCR support differently from an unknown provider prefix."""
    if provider in litellm.provider_list:
        return failures.map_failure(
            ValueError(f"OCR is not supported for provider: {provider}"), request.model, None, arguments(request)
        )
    return litellm.BadRequestError(
        message=(
            "LLM Provider NOT provided. Pass in the LLM provider you are trying to call. "
            f"You passed model={request.model}"
        ),
        model=request.model,
        llm_provider="",
    )
