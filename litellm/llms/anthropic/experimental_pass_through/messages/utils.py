from functools import lru_cache
from typing import Any, Dict, FrozenSet, List, cast, get_type_hints

from litellm.types.llms.anthropic import AnthropicMessagesRequestOptionalParams
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)


@lru_cache(maxsize=1)
def _anthropic_messages_optional_param_keys() -> FrozenSet[str]:
    """
    Valid AnthropicMessagesRequestOptionalParams keys.

    ``typing.get_type_hints`` is ~80us/call and this TypedDict is static, so
    resolving it once per process instead of once per request removes a fixed
    full-pass cost from the /v1/messages request-parse path.
    """
    return frozenset(get_type_hints(AnthropicMessagesRequestOptionalParams).keys())


class AnthropicMessagesRequestUtils:
    @staticmethod
    def get_requested_anthropic_messages_optional_param(
        params: Dict[str, Any],
        *,
        model: str | None = None,
        drop_params: bool = False,
        custom_llm_provider: str | None = None,
    ) -> AnthropicMessagesRequestOptionalParams:
        """
        Filter parameters to only include those defined in AnthropicMessagesRequestOptionalParams.

        Args:
            params: Dictionary of parameters to filter
            model: Resolved model id; when set, unsupported params may be dropped
            drop_params: Per-request drop_params flag (also respects litellm.drop_params)
            custom_llm_provider: Routed provider; fast mode is gated to direct Anthropic

        Returns:
            AnthropicMessagesRequestOptionalParams instance with only the valid parameters
        """
        valid_keys = _anthropic_messages_optional_param_keys()
        filtered_params = {
            k: v for k, v in params.items() if k in valid_keys and v is not None
        }
        if model is not None:
            from litellm.llms.anthropic.chat.transformation import AnthropicConfig

            AnthropicConfig._maybe_drop_speed_param(
                model=model,
                optional_params=filtered_params,
                drop_params=drop_params,
                custom_llm_provider=custom_llm_provider,
            )
        return cast(AnthropicMessagesRequestOptionalParams, filtered_params)


def mock_response(
    model: str,
    messages: List[Dict],
    max_tokens: int,
    mock_response: str = "Hi! My name is Claude.",
    **kwargs,
) -> AnthropicMessagesResponse:
    """
    Mock response for Anthropic messages
    """
    from litellm.exceptions import (
        ContextWindowExceededError,
        InternalServerError,
        RateLimitError,
    )

    if mock_response == "litellm.InternalServerError":
        raise InternalServerError(
            message="this is a mock internal server error",
            llm_provider="anthropic",
            model=model,
        )
    elif mock_response == "litellm.ContextWindowExceededError":
        raise ContextWindowExceededError(
            message="this is a mock context window exceeded error",
            llm_provider="anthropic",
            model=model,
        )
    elif mock_response == "litellm.RateLimitError":
        raise RateLimitError(
            message="this is a mock rate limit error",
            llm_provider="anthropic",
            model=model,
        )
    return AnthropicMessagesResponse(
        **{
            "content": [{"text": mock_response, "type": "text"}],
            "id": "msg_013Zva2CMHLNnXjNJJKqJ2EF",
            "model": "claude-sonnet-4-20250514",
            "role": "assistant",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "type": "message",
            "usage": {"input_tokens": 2095, "output_tokens": 503},
        }
    )
