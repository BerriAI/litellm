from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import litellm
from litellm.rust_bridge import failures
from litellm.rust_bridge.public_call import known_params, optional_str
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams, ResponsesAPIResponse

PARAMETERS: Final = tuple(ResponsesAPIOptionalRequestParams.__annotations__)


def body_params(request: Mapping[str, object]) -> Mapping[str, object]:
    return known_params(PARAMETERS, request)


def connection_defaults(provider: str) -> tuple[str | None, str | None]:
    provider_key: Final = litellm.openai_key if provider == "openai" else None
    return litellm.api_key or provider_key, litellm.api_base


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
