from collections.abc import Mapping
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Final, cast, get_type_hints

from litellm.types.llms.anthropic import AnthropicMessagesRequestOptionalParams
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)

if TYPE_CHECKING:
    from litellm.exceptions import ContentPolicyViolationError


def get_safeguard_refusal_stop_details(response: object) -> Mapping[str, Any] | None:
    """
    Return the ``stop_details`` of an Anthropic Messages response refused by a
    safeguard (``stop_reason: "refusal"`` carrying ``stop_details``:
    https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback),
    or None for any other response, a plain refusal without ``stop_details`` included.
    """
    if not isinstance(response, dict) or response.get("stop_reason") != "refusal":
        return None
    stop_details: Final = response.get("stop_details")
    return stop_details if isinstance(stop_details, dict) else None


def safeguard_refusal_error(model: str, stop_details: Mapping[str, object]) -> "ContentPolicyViolationError":
    """The exception a safeguard-refused Anthropic response converts into so the
    content-policy fallback chain can re-dispatch it."""
    from litellm.exceptions import ContentPolicyViolationError

    return ContentPolicyViolationError(
        message=f"Anthropic safeguard refusal (category: {stop_details.get('category')}).",
        model=model,
        llm_provider="anthropic",
    )


@lru_cache(maxsize=1)
def _anthropic_messages_optional_param_keys() -> frozenset[str]:
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
        params: dict[str, Any],
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
        valid_keys: Final = _anthropic_messages_optional_param_keys()
        filtered_params: Final = {k: v for k, v in params.items() if k in valid_keys and v is not None}
        if model is not None:
            from litellm.llms.anthropic.chat.transformation import AnthropicConfig
            from litellm.llms.anthropic.common_utils import AnthropicModelInfo

            AnthropicConfig._maybe_drop_speed_param(
                model=model,
                optional_params=filtered_params,
                drop_params=drop_params,
                custom_llm_provider=custom_llm_provider,
            )
            for param in ("temperature", "top_p", "top_k"):
                if param in filtered_params:
                    AnthropicModelInfo._apply_sampling_param(  # pyright: ignore[reportPrivateUsage]  # same gating the /chat/completions path applies; forking it would drift
                        optional_params=filtered_params,
                        model=model,
                        param=param,
                        value=filtered_params.pop(param),
                        drop_params=drop_params,
                        output_key=param,
                    )
        return cast(AnthropicMessagesRequestOptionalParams, filtered_params)


def mock_response(
    model: str,
    messages: list[dict],
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
        content=[{"text": mock_response, "type": "text"}],
        id="msg_013Zva2CMHLNnXjNJJKqJ2EF",
        model="claude-sonnet-4-20250514",
        role="assistant",
        stop_reason="end_turn",
        stop_sequence=None,
        type="message",
        usage={"input_tokens": 2095, "output_tokens": 503},
    )
