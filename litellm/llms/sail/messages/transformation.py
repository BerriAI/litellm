from collections.abc import Mapping
from typing import Final

from litellm.llms.openai_like.messages.transformation import JSONProviderAnthropicMessagesConfig
from litellm.llms.sail.common_utils import json_body, params_with_completion_window, without_keys
from litellm.types.router import GenericLiteLLMParams


class SailAnthropicMessagesConfig(JSONProviderAnthropicMessagesConfig):
    def get_supported_anthropic_messages_params(self, model: str) -> list:  # mutable-ok: matches the provider interface
        return [*super().get_supported_anthropic_messages_params(model), "metadata"]

    def request_metadata(self, metadata: Mapping[str, object] | None) -> Mapping[str, object] | None:
        if metadata is None:
            return None
        return {key: value for key, value in metadata.items() if isinstance(value, str)} or None

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: list[dict],  # mutable-ok: matches the native Messages provider interface
        anthropic_messages_optional_request_params: dict,  # mutable-ok: matches the provider interface
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: matches the native Messages provider interface
    ) -> dict:  # mutable-ok: HTTP bodies and the provider interface require plain dicts
        normalized: Final = params_with_completion_window(
            anthropic_messages_optional_request_params,
            model=model,
            drop_params=bool(litellm_params.drop_params),
            allow_drop_window=False,
        )
        return super().transform_anthropic_messages_request(
            model=model,
            messages=messages,
            anthropic_messages_optional_request_params=json_body(without_keys(normalized, frozenset({"service_tier"}))),
            litellm_params=litellm_params,
            headers=headers,
        )
