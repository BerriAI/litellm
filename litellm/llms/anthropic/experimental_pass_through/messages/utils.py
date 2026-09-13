from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Final, cast, get_type_hints

from litellm.types.llms.anthropic import (
    AnthropicMessagesRequestOptionalParams,
    AnthropicStopDetails,
)
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)
from litellm.types.llms.openai import ChatCompletionSystemMessage

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


def refusal_stop_details(explanation: str | None) -> AnthropicStopDetails:
    """The ``stop_details`` object accompanying a translated ``stop_reason: "refusal"``."""
    return AnthropicStopDetails(type="refusal", category=None, explanation=explanation)


def _mapping_field(container: object, key: str) -> object | None:
    """One key of a raw provider payload, or None when the payload is not a mapping."""
    if not isinstance(container, Mapping):
        return None
    return cast(Mapping[str, object], container).get(key)  # cast-ok: raw payload, callers re-check every value


def _mapping_str_field(container: object, key: str) -> str | None:
    value: Final = _mapping_field(container, key)
    return value if isinstance(value, str) and value else None


def openai_chat_refusal_text(message_or_delta: object) -> str | None:
    """
    Refusal text carried by an OpenAI Chat Completions message or streaming delta,
    read from ``refusal`` or from the ``provider_specific_fields`` LiteLLM parks it
    in, or None when the turn is not a refusal.
    """
    refusal: Final = getattr(message_or_delta, "refusal", None)
    if isinstance(refusal, str) and refusal:
        return refusal
    return _mapping_str_field(getattr(message_or_delta, "provider_specific_fields", None), "refusal")


def _responses_message_refusal_text(item: object) -> str | None:
    from openai.types.responses import ResponseOutputMessage, ResponseOutputRefusal

    if isinstance(item, ResponseOutputMessage):
        return next(
            (part.refusal for part in item.content if isinstance(part, ResponseOutputRefusal) and part.refusal),
            None,
        )
    raw_parts: Final = _mapping_field(item, "content")
    if _mapping_str_field(item, "type") != "message" or not isinstance(raw_parts, Sequence):
        return None
    return next(
        (
            refusal
            for part in cast(Sequence[object], raw_parts)  # cast-ok: members re-validated below
            if _mapping_str_field(part, "type") == "refusal"
            and isinstance(refusal := _mapping_str_field(part, "refusal"), str)
        ),
        None,
    )


def responses_output_refusal_text(output: Iterable[object]) -> str | None:
    """
    Refusal text carried by an OpenAI Responses ``output`` list, in typed
    (``ResponseOutputRefusal``) or raw-dictionary shape, or None when none of the
    output messages refused.
    """
    return next(
        (text for item in output if (text := _responses_message_refusal_text(item)) is not None),
        None,
    )


def safeguard_refusal_error(model: str, stop_details: Mapping[str, object]) -> "ContentPolicyViolationError":
    """The exception a safeguard-refused Anthropic response converts into so the
    content-policy fallback chain can re-dispatch it."""
    from litellm.exceptions import ContentPolicyViolationError

    return ContentPolicyViolationError(
        message=f"Anthropic safeguard refusal (category: {stop_details.get('category')}).",
        model=model,
        llm_provider="anthropic",
    )


def anthropic_system_to_openai_message(system: object) -> ChatCompletionSystemMessage | None:
    """
    Return the Anthropic Messages top-level ``system`` (a string or a list of text
    blocks) as an OpenAI-style system message, or None when the request has none.
    """
    if not isinstance(system, (str, list)) or not system:
        return None
    return ChatCompletionSystemMessage(role="system", content=system)


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
