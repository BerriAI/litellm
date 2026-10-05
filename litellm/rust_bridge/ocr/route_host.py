from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import httpx
from pydantic import TypeAdapter

import litellm
from litellm.llms.base_llm.ocr.transformation import PROVIDER_NATIVE_RESPONSE_KEY, OCRResponse
from litellm.router_utils.add_retry_fallback_headers import get_hidden_params_dict, set_hidden_params_dict
from litellm.rust_bridge import failures
from litellm.rust_bridge.failures import UpstreamFailure
from litellm.rust_bridge.ocr.entrypoints import LiteLLMOcrRequest
from litellm.rust_bridge.transport import hidden_params

__all__ = ("UpstreamFailure", "arguments", "map_failure", "response")

_RESPONSE_ADAPTER: Final = TypeAdapter(dict[str, object])


def response(value: Mapping[str, object], headers: httpx.Headers | None = None, status: int = 200) -> OCRResponse:
    provider_native_response: Final = value.get(PROVIDER_NATIVE_RESPONSE_KEY)
    normalized: Final = OCRResponse.model_validate(
        MappingProxyType({key: item for key, item in value.items() if key != PROVIDER_NATIVE_RESPONSE_KEY})
    )
    if isinstance(provider_native_response, Mapping):
        normalized.set_provider_native_response(_RESPONSE_ADAPTER.validate_python(provider_native_response))
    if headers is not None:
        set_hidden_params_dict(normalized, {**get_hidden_params_dict(normalized), **hidden_params(headers, status)})
    return normalized


def arguments(request: LiteLLMOcrRequest) -> Mapping[str, object]:
    return request.kwargs


def map_failure(error: Exception, request: LiteLLMOcrRequest, request_provider: str) -> Exception:
    if getattr(error, "ocr_request_format_error", False):
        return litellm.UnsupportedParamsError(
            message=f"Invalid `req_format`: {request.kwargs.get('req_format')!r}. Expected 'native' or 'litellm'.",
            model=request.model.removeprefix(f"{request_provider}/"),
            llm_provider=request_provider,
        )
    return failures.map_native_failure(error, request.model, request_provider, arguments(request), request.api_base)
