from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Final, cast, get_type_hints

from pydantic import JsonValue

from litellm.exceptions import BadRequestError
from litellm.integrations.anthropic_cache_control_hook import AnthropicCacheControlHook
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.get_llm_provider_logic import resolve_model_and_provider_for_metadata
from litellm.llms.anthropic.common_utils import (
    AnthropicModelInfo,
    flatten_unencrypted_web_search_results_in_anthropic_messages,
    sanitize_tool_use_ids_in_anthropic_messages,
    strip_empty_content_blocks_from_anthropic_messages,
)
from litellm.llms.anthropic.pass_through.convertible_blocks import (
    drop_non_string_system_texts,
    drop_non_string_texts,
    reject_emptied_messages,
    reject_non_string_system_text,
    reject_non_string_user_text,
)
from litellm.llms.anthropic.pass_through.messages.api_versions import serves_anthropic_api_versions
from litellm.llms.anthropic.pass_through.utils import is_reasoning_auto_summary_enabled
from litellm.llms.base_llm.anthropic_messages.transformation import BaseAnthropicMessagesConfig
from litellm.types.llms.anthropic import (
    AnthropicMessagesRequestOptionalParams,
    AnthropicStopDetails,
)
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)
from litellm.types.llms.openai import ChatCompletionSystemMessage
from litellm.types.utils import LlmProviders, LlmProvidersSet
from litellm.utils import ProviderConfigManager

if TYPE_CHECKING:
    from litellm.exceptions import ContentPolicyViolationError


INCOMPLETE_STREAM_ERROR_MESSAGE: Final = (
    "Provider stream ended before emitting a message_stop event; "
    "the response is incomplete and any partial content (e.g. tool_use input JSON) may be truncated."
)


def get_safeguard_refusal_stop_details(response: object) -> Mapping[str, object] | None:
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
    return container.get(key)


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


def normalize_native_messages(
    messages: Sequence[object], *, rewrites_tool_use_ids: bool = True
) -> list[dict[str, object]]:
    """Apply the block rewrites every /v1/messages request gets before it reaches a provider."""
    stripped: Final = strip_empty_content_blocks_from_anthropic_messages(messages)
    return flatten_unencrypted_web_search_results_in_anthropic_messages(
        sanitize_tool_use_ids_in_anthropic_messages(stripped) if rewrites_tool_use_ids else stripped
    )


def provider_messages_config(model: str, custom_llm_provider: str | None) -> BaseAnthropicMessagesConfig | None:
    return (
        ProviderConfigManager.get_provider_anthropic_messages_config(
            model=model,
            provider=LlmProviders(custom_llm_provider),
        )
        if custom_llm_provider is not None and custom_llm_provider in LlmProvidersSet
        else None
    )


_HOSTS_SERVING_CLAUDE_UNDER_ANY_DEPLOYMENT_NAME: Final = frozenset(
    {LlmProviders.BEDROCK.value, LlmProviders.AZURE_AI.value}
)


def renames_tool_use_ids_up_front(model: str, custom_llm_provider: str | None, api_base: str | None) -> bool:
    try:
        resolved_model, provider = resolve_model_and_provider_for_metadata(model, custom_llm_provider, api_base)
    except BadRequestError:
        return True
    if provider == LlmProviders.ANTHROPIC.value:
        return serves_anthropic_api_versions(AnthropicModelInfo.get_api_base(api_base))
    return provider in _HOSTS_SERVING_CLAUDE_UNDER_ANY_DEPLOYMENT_NAME or (
        provider_messages_config(resolved_model, provider) is not None
    )


def native_messages_to_send(
    messages: Sequence[object], drop_params: object, *, rewrites_tool_use_ids: bool = True
) -> list[dict[str, object]]:  # mutable-ok: cache-control injection and token counters take list[dict]
    return list(
        drop_non_string_texts(
            normalize_native_messages(messages, rewrites_tool_use_ids=rewrites_tool_use_ids), drop_params
        )
    )


def cleaned_native_messages(
    messages: Sequence[object],
    drop_params: object,
    *,
    model: str,
    custom_llm_provider: str | None,
    api_base: str | None,
) -> list[dict[str, object]]:  # mutable-ok: maybe_inject_cache_control takes these messages as list[dict]
    reject_non_string_user_text(messages, drop_params, model, custom_llm_provider)
    cleaned: Final = native_messages_to_send(
        messages,
        drop_params,
        rewrites_tool_use_ids=renames_tool_use_ids_up_front(model, custom_llm_provider, api_base),
    )
    reject_emptied_messages(messages, cleaned, model, custom_llm_provider)
    return cleaned


def cleaned_native_system(
    system: str | list[dict[str, JsonValue]] | None,  # mutable-ok: AnthropicMessagesRequest.system is typed list[dict]
    drop_params: object,
    *,
    model: str,
    custom_llm_provider: str | None,
) -> str | list[dict[str, JsonValue]] | None:  # mutable-ok: AnthropicMessagesRequest.system is typed list[dict]
    reject_non_string_system_text(system, drop_params, model, custom_llm_provider)
    return drop_non_string_system_texts(system, drop_params)


def prepare_native_messages(
    messages: list[dict[str, JsonValue]],
    system: str | list[dict[str, JsonValue]] | None,
    kwargs: dict[str, object],
    *,
    model: str,
    custom_llm_provider: str | None = None,
    tools: list[dict[str, JsonValue]] | None = None,
    api_base: str | None = None,
    presanitized: bool = False,
) -> tuple[list[dict[str, JsonValue]], str | list[dict[str, JsonValue]] | None]:
    from litellm.responses.bridge_check import needs_empty_user_turn_back

    normalized: Final = (
        messages
        if presanitized
        else cleaned_native_messages(
            messages,
            kwargs.get("drop_params"),
            model=model,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
        )
    )
    cleaned_system: Final = (
        system
        if presanitized
        else cleaned_native_system(
            system, kwargs.get("drop_params"), model=model, custom_llm_provider=custom_llm_provider
        )
    )
    restored: Final = (
        [*normalized, {"role": "user", "content": ""}]
        if needs_empty_user_turn_back(
            messages[-1] if messages else None,
            normalized,
            model,
            custom_llm_provider,
            {**kwargs, "api_base": api_base},
        )
        else normalized
    )
    return cast(  # cast-ok: legacy normalizers and injection preserve the JSON message and system shapes
        tuple[list[dict[str, JsonValue]], str | list[dict[str, JsonValue]] | None],
        AnthropicCacheControlHook.maybe_inject_cache_control(
            restored,
            cleaned_system,
            kwargs,
            model=model,
            custom_llm_provider=custom_llm_provider,
            tools=tools,
            api_base=api_base,
        ),
    )


@lru_cache(maxsize=1)
def anthropic_messages_optional_param_keys() -> frozenset[str]:
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
        params: Mapping[str, object],
        *,
        model: str | None = None,
        drop_params: bool = False,
        custom_llm_provider: str | None = None,
        additional_drop_params: Iterable[str] = (),
    ) -> AnthropicMessagesRequestOptionalParams:
        """
        Filter parameters to only include those defined in AnthropicMessagesRequestOptionalParams.

        Args:
            params: Dictionary of parameters to filter
            model: Resolved model id; when set, unsupported params may be dropped
            drop_params: Per-request drop_params flag (also respects litellm.drop_params)
            custom_llm_provider: Routed provider; fast mode is gated to direct Anthropic
            additional_drop_params: Paths the deployment drops. A listed top-level key is left out
                before the unsupported-param checks run, so it never raises

        Returns:
            AnthropicMessagesRequestOptionalParams instance with only the valid parameters
        """
        valid_keys: Final = anthropic_messages_optional_param_keys().difference(additional_drop_params)
        filtered_params: Final = {k: v for k, v in params.items() if k in valid_keys and v is not None}
        if model is not None:
            from litellm.llms.anthropic.chat.transformation import AnthropicConfig
            from litellm.llms.anthropic.common_utils import AnthropicModelInfo

            AnthropicConfig.maybe_drop_speed_param(
                model=model,
                optional_params=filtered_params,
                drop_params=drop_params,
                custom_llm_provider=custom_llm_provider,
            )
            AnthropicConfig.maybe_drop_service_tier_param(
                model=model,
                optional_params=filtered_params,
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
        if is_reasoning_auto_summary_enabled():
            thinking_param: Final = filtered_params.get("thinking")
            thinking_mapping: Final = as_str_mapping(thinking_param)
            if (
                isinstance(thinking_param, dict)
                and thinking_mapping is not None
                and thinking_mapping.get("type") != "disabled"
            ):
                return cast(
                    AnthropicMessagesRequestOptionalParams,
                    {**filtered_params, "thinking": {**thinking_mapping, "display": "summarized"}},
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
