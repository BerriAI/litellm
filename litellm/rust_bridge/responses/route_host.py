from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import httpx

import litellm
from litellm import get_llm_provider
from litellm.router_utils.add_retry_fallback_headers import get_hidden_params_dict, set_hidden_params_dict
from litellm.rust_bridge import failures
from litellm.rust_bridge.public_call import inference_decline_reason
from litellm.rust_bridge.responses.entrypoints import LiteLLMResponsesRequest
from litellm.rust_bridge.transport import hidden_params
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams, ResponsesAPIResponse

PARAMETERS: Final = tuple(ResponsesAPIOptionalRequestParams.__annotations__)


def connection_defaults(_provider: str) -> tuple[str | None, str | None]:
    return litellm.api_key or litellm.openai_key, litellm.api_base


def response(
    value: Mapping[str, object], headers: httpx.Headers | None = None, status: int = 200
) -> ResponsesAPIResponse:
    built: Final = ResponsesAPIResponse.model_validate(value)
    if headers is not None:
        set_hidden_params_dict(built, {**get_hidden_params_dict(built), **hidden_params(headers, status)})
    return built


def arguments(request: LiteLLMResponsesRequest) -> Mapping[str, object]:
    return request.kwargs


def map_failure(error: Exception, request: LiteLLMResponsesRequest) -> Exception:
    provider: Final = request.custom_llm_provider or "openai"
    return failures.map_native_failure(error, request.model, provider, arguments(request), request.api_base)


def decline_reason(request: LiteLLMResponsesRequest) -> str | None:
    if request.custom_llm_provider is None and "/" not in request.model:
        try:
            _, provider, _, _ = get_llm_provider(model=request.model)
        except litellm.exceptions.BadRequestError:
            return "native Responses could not resolve the provider"
        if provider != "openai":
            return "native HTTP responses provider"
    return inference_decline_reason(PARAMETERS, {**request.parameters, **request.kwargs})
