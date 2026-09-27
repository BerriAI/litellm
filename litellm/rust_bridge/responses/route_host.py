from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from litellm import get_llm_provider
from litellm.rust_bridge import failures
from litellm.rust_bridge.public_call import inference_decline_reason
from litellm.rust_bridge.responses.entrypoints import LiteLLMResponsesRequest
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams, ResponsesAPIResponse

PARAMETERS: Final = tuple(ResponsesAPIOptionalRequestParams.__annotations__)


def response(value: Mapping[str, object]) -> ResponsesAPIResponse:
    return ResponsesAPIResponse.model_validate(value)


def arguments(request: LiteLLMResponsesRequest) -> Mapping[str, object]:
    return request.kwargs


def map_failure(error: Exception, request: LiteLLMResponsesRequest) -> Exception:
    provider: Final = request.custom_llm_provider or "openai"
    return failures.map_native_failure(error, request.model, provider, arguments(request), request.api_base)


def decline_reason(request: LiteLLMResponsesRequest) -> str | None:
    if request.custom_llm_provider is None and "/" not in request.model:
        try:
            _, provider, _, _ = get_llm_provider(model=request.model)
        except Exception:  # noqa: BLE001  # unresolved models stay on the existing Python dispatch path
            return "native Responses could not resolve the provider"
        if provider != "openai":
            return "native HTTP responses provider"
    return inference_decline_reason(PARAMETERS, {**request.parameters, **request.kwargs})
