from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import litellm
from litellm.rust_bridge.public_call import unsupported_argument
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams, ResponsesAPIResponse

PARAMETERS: Final = tuple(ResponsesAPIOptionalRequestParams.__annotations__)


def connection_defaults(_provider: str) -> tuple[str | None, str | None]:
    return litellm.api_key or litellm.openai_key, litellm.api_base


def response(value: Mapping[str, object]) -> ResponsesAPIResponse:
    return ResponsesAPIResponse.model_validate(value)


def arguments(request: Mapping[str, object]) -> Mapping[str, object]:
    return request


def unsupported_request(request: Mapping[str, object]) -> str | None:
    """An argument the native Responses route does not carry yet, reported before any work.

    Goes away with the kwarg passthrough; the provider and streaming checks already live in Rust."""
    return unsupported_argument(PARAMETERS, request)
