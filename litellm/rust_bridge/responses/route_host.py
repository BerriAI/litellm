from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import litellm
from litellm import get_llm_provider
from litellm.rust_bridge import failures
from litellm.rust_bridge.public_call import inference_decline_reason, optional_str
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams, ResponsesAPIResponse

PARAMETERS: Final = tuple(ResponsesAPIOptionalRequestParams.__annotations__)


def connection_defaults(_provider: str) -> tuple[str | None, str | None]:
    return litellm.api_key or litellm.openai_key, litellm.api_base


def response(value: Mapping[str, object]) -> ResponsesAPIResponse:
    return ResponsesAPIResponse.model_validate(value)


def map_failure(error: Exception, request: Mapping[str, object]) -> Exception:
    provider: Final = optional_str(request.get("custom_llm_provider")) or "openai"
    return failures.map_native_failure(
        error,
        str(request["model"]),
        provider,
        request,
        optional_str(request.get("api_base")) or optional_str(request.get("base_url")),
    )


def decline_reason(request: Mapping[str, object]) -> str | None:
    if optional_str(request.get("custom_llm_provider")) is None and "/" not in str(request["model"]):
        try:
            _, provider, _, _ = get_llm_provider(model=str(request["model"]))
        except litellm.exceptions.BadRequestError:
            return "native Responses could not resolve the provider"
        if provider != "openai":
            return "native HTTP responses provider"
    return inference_decline_reason(PARAMETERS, request)
