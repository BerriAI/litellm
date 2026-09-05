"""
OpenCode Anthropic-wire chat config.

The gateway serves part of its catalogue over the Anthropic Messages wire
format. Those models still have to be reachable through ``litellm.completion()``
with OpenAI-shaped input and a ``ModelResponse`` back, so they route through
``AnthropicConfig`` (the same translation the first-party ``anthropic`` provider
uses) rather than the raw ``/v1/messages`` passthrough handler, which does no
translation in either direction.

Only auth differs from the base: OpenCode authenticates with ``x-api-key`` and
its own key, never ANTHROPIC_API_KEY.
"""

from typing import Final

from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.llms.opencode.common_utils import (
    cost_map_max_output_tokens,
    resolve_opencode_api_key,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.router import GenericLiteLLMParams


class OpenCodeAnthropicConfig(AnthropicConfig):
    """Anthropic wire format over the OpenCode gateway, via the chat path."""

    def __init__(self, surface: str = "zen") -> None:
        super().__init__()
        self.surface: Final = surface

    @property
    def custom_llm_provider(self) -> str | None:
        return f"opencode_{self.surface}"

    def should_strip_billing_metadata(self) -> bool:
        """OpenCode is a third-party gateway, not the first-party Anthropic API;
        x-anthropic-billing-header client attribution must not leak to it."""
        return True

    def map_openai_params(
        self,
        non_default_params: dict,  # mutable-ok: signature must match AnthropicConfig
        optional_params: dict,  # mutable-ok: signature must match AnthropicConfig
        model: str,
        drop_params: bool,
    ) -> dict:  # mutable-ok: signature must match AnthropicConfig
        """Default ``max_tokens`` from the cost map when the caller omitted it.

        The Anthropic wire format requires ``max_tokens`` while
        ``litellm.completion()`` does not, so the base class substitutes a flat
        DEFAULT_ANTHROPIC_CHAT_MAX_TOKENS. That lookup uses the bare model name,
        which misses the surface-qualified cost-map entry
        (``opencode_zen/claude-sonnet-4``), so the model's real
        ``max_output_tokens`` is resolved here instead.
        """
        mapped: Final = super().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
        )
        caller_set_max_tokens: Final = (
            non_default_params.get("max_tokens") is not None
            or non_default_params.get("max_completion_tokens") is not None
        )
        if caller_set_max_tokens:
            return mapped
        cost_map_max_tokens: Final = cost_map_max_output_tokens(surface=self.surface, model=model)
        if cost_map_max_tokens is None:
            return mapped
        return {  # mutable-ok: request params stay mutable for the base handler
            **mapped,
            "max_tokens": cost_map_max_tokens,
        }

    def validate_environment(
        self,
        headers: dict,  # mutable-ok: signature must match AnthropicConfig
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: signature must match AnthropicConfig
        optional_params: dict,  # mutable-ok: signature must match AnthropicConfig
        litellm_params: dict | GenericLiteLLMParams,  # mutable-ok: signature must match AnthropicConfig
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:  # mutable-ok: signature must match AnthropicConfig
        """Resolve the OpenCode key, then let the base class build the headers.

        The base class resolves a missing key from ANTHROPIC_API_KEY, so the key
        is resolved here and a missing one is rejected before that fallback can
        send a first-party Anthropic credential to opencode.ai.
        """
        resolved_key: Final = resolve_opencode_api_key(self.surface, api_key)
        if resolved_key is None:
            raise ValueError(
                f"OpenCode API key is required. Set OPENCODE_{self.surface.upper()}_API_KEY "
                f"or OPENCODE_API_KEY, or pass api_key."
            )
        params: Final = (
            litellm_params if isinstance(litellm_params, dict) else litellm_params.model_dump(exclude_none=True)
        )
        return super().validate_environment(
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=params,
            api_key=resolved_key,
            api_base=api_base,
        )
