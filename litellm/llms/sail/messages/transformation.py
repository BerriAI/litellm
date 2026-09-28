from collections.abc import Mapping
from typing import Final

from litellm.litellm_core_utils.core_helpers import normalize_drop_params
from litellm.llms.openai_like.messages.transformation import (
    JSONProviderAnthropicMessagesConfig,
)
from litellm.llms.sail.common_utils import params_with_completion_window, without_keys
from litellm.types.router import GenericLiteLLMParams


class SailAnthropicMessagesConfig(JSONProviderAnthropicMessagesConfig):
    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: list[dict],
        anthropic_messages_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> dict:
        normalized: Final = params_with_completion_window(
            anthropic_messages_optional_request_params,
            model=model,
            drop_params=normalize_drop_params(litellm_params.get("drop_params")) is True,
        )
        metadata: Final = normalized.get("metadata")
        parent_params: Final = (
            {**normalized, "metadata": dict(metadata)} if isinstance(metadata, Mapping) else dict(normalized)
        )
        return super().transform_anthropic_messages_request(
            model=model,
            messages=messages,
            anthropic_messages_optional_request_params=dict(without_keys(parent_params, frozenset({"service_tier"}))),
            litellm_params=litellm_params,
            headers=headers,
        )
