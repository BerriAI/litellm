from __future__ import annotations

from collections.abc import Mapping

from litellm.rust_bridge import failures
from litellm.rust_bridge.chat_completions.entrypoints import LiteLLMChatCompletionsRequest
from litellm.types.utils import ModelResponse


def response(value: Mapping[str, object]) -> ModelResponse:
    return ModelResponse(**value)


def arguments(request: LiteLLMChatCompletionsRequest) -> Mapping[str, object]:
    return request.kwargs


def map_failure(error: Exception, request: LiteLLMChatCompletionsRequest, request_provider: str) -> Exception:
    return failures.map_failure(error, request.model, request_provider, arguments(request))
