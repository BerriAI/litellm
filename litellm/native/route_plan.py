"""Which native API format a request reaches its provider in, decided once per request by the client-facing surface."""

from dataclasses import dataclass
from typing import Final, TypeAlias

from litellm.llms.base_llm.anthropic_messages.transformation import BaseAnthropicMessagesConfig
from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager, get_model_info

__all__ = (
    "NativeChat",
    "NativeMessages",
    "NativeResponses",
    "RoutePlan",
    "native_messages_config",
    "plan_chat_route",
    "plan_messages_route",
    "plan_responses_route",
)

_MESSAGES_TO_RESPONSES_PROVIDERS: Final = frozenset({"openai"})
_PROVIDER_VALUES: Final = frozenset(provider.value for provider in LlmProviders)


@dataclass(frozen=True, slots=True)
class NativeMessages:
    config: BaseAnthropicMessagesConfig


@dataclass(frozen=True, slots=True)
class NativeResponses:
    pass


@dataclass(frozen=True, slots=True)
class NativeChat:
    pass


RoutePlan: TypeAlias = NativeMessages | NativeResponses | NativeChat


def _deployment_passes_through_anthropic_messages(model_info: object) -> bool:
    """Whether the deployment opted into forwarding /v1/messages untranslated.

    The opt-in is ``model_info.supported_endpoints`` containing ``"/v1/messages"``,
    declared per deployment in config.yaml and plumbed here as ``kwargs["model_info"]``
    by the router.
    """
    if not isinstance(model_info, dict):
        return False
    supported_endpoints: Final = model_info.get("supported_endpoints")
    return isinstance(supported_endpoints, (list, tuple)) and "/v1/messages" in supported_endpoints


def _deployment_supports_cache_control_ttl(model_info: object) -> bool:
    return isinstance(model_info, dict) and model_info.get("cache_control_ttl") is True


def native_messages_config(
    model: str, custom_llm_provider: str | None, model_info: object, api_base: str | None
) -> BaseAnthropicMessagesConfig | None:
    provider_config: Final = (
        ProviderConfigManager.get_provider_anthropic_messages_config(
            model=model,
            provider=LlmProviders(custom_llm_provider),
        )
        if custom_llm_provider is not None and custom_llm_provider in _PROVIDER_VALUES
        else None
    )
    native_config: Final = (
        provider_config if provider_config is not None and provider_config.serves_api_base(model, api_base) else None
    )
    if native_config is not None or not _deployment_passes_through_anthropic_messages(model_info):
        return native_config
    from litellm.llms.openai_like.messages.transformation import (
        OpenAILikeAnthropicMessagesConfig,
    )

    return OpenAILikeAnthropicMessagesConfig(cache_control_ttl=_deployment_supports_cache_control_ttl(model_info))


def _bridges_to_responses_api(model: str, custom_llm_provider: str, deployment_model_info: object) -> bool:
    from litellm.responses.bridge_check import responses_api_bridge_check, str_keyed_mapping

    model_info, _ = responses_api_bridge_check(
        model=model,
        custom_llm_provider=custom_llm_provider,
        deployment_model_info=str_keyed_mapping(deployment_model_info),
    )
    return model_info.get("mode") == "responses"


def _responses_mode_is_lost_by_prefix_strip(
    requested_model: str, resolved_model: str, custom_llm_provider: str, deployment_model_info: object
) -> bool:
    """Whether a Responses-only deployment stops looking like one once its provider prefix is stripped.

    ``litellm.completion`` re-derives the Responses bridge from the stripped id alone, so a
    deployment id such as ``perplexity/perplexity/sonar`` (mode ``responses``) is shadowed by the
    chat entry ``perplexity/sonar`` and would otherwise be sent to chat/completions.
    """
    if requested_model == resolved_model:
        return False
    return _bridges_to_responses_api(
        requested_model, custom_llm_provider, deployment_model_info
    ) and not _bridges_to_responses_api(resolved_model, custom_llm_provider, deployment_model_info)


def _model_may_reason(model: str, custom_llm_provider: str) -> bool:
    try:
        return (
            get_model_info(model=model, custom_llm_provider=custom_llm_provider).get("supports_reasoning") is not False
        )
    except Exception:  # noqa: BLE001  # an unmapped model is assumed to reason, as the chat bridge assumed
        return True


def _thinking_needs_responses(custom_llm_provider: str | None, resolved_model: str, thinking_enabled: bool) -> bool:
    """OpenAI returns reasoning text only on the Responses API, so enabled thinking goes there even when chat is forced."""
    return (
        thinking_enabled
        and custom_llm_provider is not None
        and custom_llm_provider in _MESSAGES_TO_RESPONSES_PROVIDERS
        and _model_may_reason(resolved_model, custom_llm_provider)
    )


def _messages_bridge_to_responses(
    custom_llm_provider: str | None,
    requested_model: str,
    resolved_model: str,
    deployment_model_info: object,
    force_chat_completions: bool,
    thinking_enabled: bool,
) -> bool:
    if force_chat_completions:
        return _thinking_needs_responses(custom_llm_provider, resolved_model, thinking_enabled)
    if custom_llm_provider in _MESSAGES_TO_RESPONSES_PROVIDERS:
        return True
    if custom_llm_provider is None:
        return False
    return _responses_mode_is_lost_by_prefix_strip(
        requested_model, resolved_model, custom_llm_provider, deployment_model_info
    )


def plan_messages_route(
    *,
    requested_model: str,
    resolved_model: str,
    custom_llm_provider: str | None,
    model_info: object,
    api_base: str | None,
    compaction_requested: bool,
    force_chat_completions: bool,
    thinking_enabled: bool,
) -> RoutePlan:
    """``force_chat_completions`` is ``litellm.use_chat_completions_url_for_anthropic_messages``."""
    config: Final = native_messages_config(resolved_model, custom_llm_provider, model_info, api_base)
    if config is not None:
        return NativeMessages(config)
    if not compaction_requested and _messages_bridge_to_responses(
        custom_llm_provider, requested_model, resolved_model, model_info, force_chat_completions, thinking_enabled
    ):
        return NativeResponses()
    return NativeChat()


def plan_responses_route(
    responses_api_provider_config: BaseResponsesAPIConfig | None, use_chat_completions_api: bool
) -> NativeResponses | NativeChat:
    if responses_api_provider_config is None or use_chat_completions_api:
        return NativeChat()
    return NativeResponses()


def plan_chat_route(responses_api_provider_config: BaseResponsesAPIConfig | None) -> NativeChat | NativeResponses:
    """For a deployment ``responses_api_bridge_check`` resolved to mode ``responses``."""
    return NativeChat() if responses_api_provider_config is None else NativeResponses()
