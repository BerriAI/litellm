from __future__ import annotations

from collections.abc import Mapping

from litellm.rust_bridge import failures
from litellm.rust_bridge.responses.entrypoints import LiteLLMResponsesRequest
from litellm.types.llms.openai import ResponsesAPIResponse


def response(value: Mapping[str, object]) -> ResponsesAPIResponse:
    return ResponsesAPIResponse.model_validate(value)


def arguments(request: LiteLLMResponsesRequest) -> Mapping[str, object]:
    return request.kwargs


def map_failure(error: Exception, request: LiteLLMResponsesRequest, request_provider: str) -> Exception:
    return failures.map_failure(error, request.model, request_provider, arguments(request))
