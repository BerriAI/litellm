from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import litellm
from litellm import get_llm_provider
from litellm.rust_bridge.public_call import optional_str, unsupported_argument
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams, ResponsesAPIResponse

PARAMETERS: Final = tuple(ResponsesAPIOptionalRequestParams.__annotations__)


def connection_defaults(_provider: str) -> tuple[str | None, str | None]:
    return litellm.api_key or litellm.openai_key, litellm.api_base


def response(value: Mapping[str, object]) -> ResponsesAPIResponse:
    return ResponsesAPIResponse.model_validate(value)


def arguments(request: Mapping[str, object]) -> Mapping[str, object]:
    return request


def unsupported_request(request: Mapping[str, object]) -> str | None:
    """Why the native Responses route cannot serve this call, reported as a request failure before any work."""
    model: Final = str(request["model"])
    provider: Final = optional_str(request.get("custom_llm_provider"))
    if provider is None and "/" not in model:
        try:
            _, resolved, _, _ = get_llm_provider(model=model)
        except litellm.exceptions.BadRequestError:
            return "native Responses could not resolve the provider"
        if resolved != "openai":
            return "native HTTP responses provider"
    elif (provider or model.partition("/")[0]) != "openai" or "/" in model.removeprefix("openai/"):
        return "native HTTP responses provider"
    return unsupported_argument(PARAMETERS, request)
