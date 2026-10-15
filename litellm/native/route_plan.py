"""Which native API format a request reaches its provider in, decided once per request by the client-facing surface."""

from dataclasses import dataclass
from typing import Final, TypeAlias

from litellm.llms.base_llm.anthropic_messages.transformation import BaseAnthropicMessagesConfig
from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

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
    model: str, custom_llm_provider: str | None, model_info: object
) -> BaseAnthropicMessagesConfig | None:
    provider_config: Final = (
        ProviderConfigManager.get_provider_anthropic_messages_config(
            model=model,
            provider=LlmProviders(custom_llm_provider),
        )
        if custom_llm_provider is not None and custom_llm_provider in _PROVIDER_VALUES
        else None
    )
    if provider_config is not None or not _deployment_passes_through_anthropic_messages(model_info):
        return provider_config
    from litellm.llms.openai_like.messages.transformation import (
        OpenAILikeAnthropicMessagesConfig,
    )

    return OpenAILikeAnthropicMessagesConfig(cache_control_ttl=_deployment_supports_cache_control_ttl(model_info))


def _bridges_to_responses_api(model: str, custom_llm_provider: str) -> bool:
    from litellm.responses.bridge_check import responses_api_bridge_check

    model_info, _ = responses_api_bridge_check(model=model, custom_llm_provider=custom_llm_provider)
    return model_info.get("mode") == "responses"


def _responses_mode_is_lost_by_prefix_strip(
    requested_model: str, resolved_model: str, custom_llm_provider: str
) -> bool:
    """Whether a Responses-only deployment stops looking like one once its provider prefix is stripped.

    ``litellm.completion`` re-derives the Responses bridge from the stripped id alone, so a
    deployment id such as ``perplexity/perplexity/sonar`` (mode ``responses``) is shadowed by the
    chat entry ``perplexity/sonar`` and would otherwise be sent to chat/completions.
    """
    if requested_model == resolved_model:
        return False
    return _bridges_to_responses_api(requested_model, custom_llm_provider) and not _bridges_to_responses_api(
        resolved_model, custom_llm_provider
    )


def _messages_bridge_to_responses(
    custom_llm_provider: str | None, requested_model: str, resolved_model: str, force_chat_completions: bool
) -> bool:
    if force_chat_completions:
        return False
    if custom_llm_provider in _MESSAGES_TO_RESPONSES_PROVIDERS:
        return True
    if custom_llm_provider is None:
        return False
    return _responses_mode_is_lost_by_prefix_strip(requested_model, resolved_model, custom_llm_provider)


def plan_messages_route(
    *,
    requested_model: str,
    resolved_model: str,
    custom_llm_provider: str | None,
    model_info: object,
    compaction_requested: bool,
    force_chat_completions: bool,
) -> RoutePlan:
    """``force_chat_completions`` is ``litellm.use_chat_completions_url_for_anthropic_messages``."""
    config: Final = native_messages_config(resolved_model, custom_llm_provider, model_info)
    if config is not None:
        return NativeMessages(config)
    if not compaction_requested and _messages_bridge_to_responses(
        custom_llm_provider, requested_model, resolved_model, force_chat_completions
    ):
        return NativeResponses()
    return NativeChat()


def plan_responses_route(
    responses_api_provider_config: BaseResponsesAPIConfig | None, use_chat_completions_api: bool
) -> NativeResponses | NativeChat:
    if responses_api_provider_config is None or use_chat_completions_api:
        return NativeChat()
    return NativeResponses()


def plan_chat_route(responses_mode_model: bool, skip_responses_api_bridge: bool) -> NativeChat | NativeResponses:
    """``responses_mode_model`` is whether ``responses_api_bridge_check`` resolved the deployment to mode ``responses``."""
    if responses_mode_model and not skip_responses_api_bridge:
        return NativeResponses()
    return NativeChat()
