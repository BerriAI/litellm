"""
Apodex Anthropic Messages — native passthrough for the core models only.

Apodex implements the Anthropic protocol itself at POST /v1/messages and serves
the core models there, so the payload is forwarded untranslated and
Anthropic-only features such as `thinking` and `cache_control` survive. The Deep
Research tiers are not served on that path, so `ProviderConfigManager` hands back
no config for them and they fall back to LiteLLM's Anthropic-to-chat-completions
translation.

Ref: https://platform.apodex.ai/docs/anthropic-messages
"""

from litellm.llms.openai_like.messages.transformation import (
    OpenAILikeAnthropicMessagesConfig,
)

from ..common_utils import get_apodex_api_base, get_apodex_api_key


class ApodexAnthropicMessagesConfig(OpenAILikeAnthropicMessagesConfig):
    @property
    def custom_llm_provider(self) -> str | None:
        return "apodex"

    def should_strip_billing_metadata(self) -> bool:
        return True

    def validate_anthropic_messages_environment(
        self,
        headers: dict[str, str],  # mutable-ok: matches the base-class signature
        model: str,
        messages: list[object],  # mutable-ok: matches the base-class signature
        optional_params: dict,  # mutable-ok: matches the base-class signature
        litellm_params: dict,  # mutable-ok: matches the base-class signature
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> tuple[dict[str, str], str | None]:  # mutable-ok: matches the base-class signature
        """Fill in the Apodex credentials and base URL.

        The returned api_base is what the handler hands to get_complete_url, so
        resolving it here is enough to reach the native endpoint.
        """
        return super().validate_anthropic_messages_environment(
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=get_apodex_api_key(api_key),
            api_base=get_apodex_api_base(api_base),
        )
