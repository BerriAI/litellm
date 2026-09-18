from __future__ import annotations

from collections.abc import Mapping
from typing import cast  # noqa: TID251  # narrows the normalized native payload to the public TypedDict

from litellm.rust_bridge import failures
from litellm.rust_bridge.messages.entrypoints import LiteLLMMessagesRequest
from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicMessagesResponse


def response(value: Mapping[str, object]) -> AnthropicMessagesResponse:
    return cast(  # cast-ok: AnthropicMessagesResponse is a TypedDict over the normalized native payload
        AnthropicMessagesResponse,
        dict(value),  # mutable-ok: the public Messages response is a TypedDict the caller may annotate in place
    )


def arguments(request: LiteLLMMessagesRequest) -> Mapping[str, object]:
    return request.kwargs


def map_failure(error: Exception, request: LiteLLMMessagesRequest, request_provider: str) -> Exception:
    return failures.map_failure(error, request.model, request_provider, arguments(request))
