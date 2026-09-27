from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from litellm.constants import OPENAI_CHAT_COMPLETION_PARAMS
from litellm.rust_bridge import failures
from litellm.rust_bridge.chat_completions.entrypoints import LiteLLMChatCompletionsRequest
from litellm.rust_bridge.public_call import inference_decline_reason
from litellm.types.utils import ModelResponse

_TRANSPORT_PARAMETERS: Final = frozenset(
    {
        "api_base",
        "api_key",
        "api_version",
        "deployment_id",
        "organization",
        "base_url",
        "default_headers",
        "timeout",
        "request_timeout",
        "max_retries",
        "extra_headers",
    }
)
PARAMETERS: Final = tuple(name for name in OPENAI_CHAT_COMPLETION_PARAMS if name not in _TRANSPORT_PARAMETERS)


def response(value: Mapping[str, object]) -> ModelResponse:
    return ModelResponse(**value)


def arguments(request: LiteLLMChatCompletionsRequest) -> Mapping[str, object]:
    return request.kwargs


def map_failure(error: Exception, request: LiteLLMChatCompletionsRequest) -> Exception:
    provider: Final = request.custom_llm_provider or request.model.partition("/")[0]
    return failures.map_native_failure(error, request.model, provider, arguments(request), request.api_base)


def decline_reason(request: LiteLLMChatCompletionsRequest) -> str | None:
    return inference_decline_reason(PARAMETERS, {**request.parameters, **request.kwargs})
