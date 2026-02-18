"""
Claude Code - Prompt Cache Injection Guardrail

Automatically injects cache_control: {type: ephemeral} into system messages
so Anthropic can cache the prompt prefix, reducing costs on repeated calls.

Only runs when the request targets an Anthropic API model.  Uses the existing
AnthropicCacheControlHook._safe_insert_cache_control_in_message utility so
the injection logic is not duplicated.
"""

from typing import TYPE_CHECKING, Literal, Optional

from litellm._logging import verbose_proxy_logger
from litellm.integrations.anthropic_cache_control_hook import AnthropicCacheControlHook
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.llms.openai import ChatCompletionCachedContent
from litellm.types.utils import GenericGuardrailAPIInputs
from litellm.utils import get_llm_provider

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


def _is_anthropic_model(model: Optional[str]) -> bool:
    """Return True when the model resolves to the Anthropic provider."""
    if not model:
        return False
    try:
        _, custom_llm_provider, _, _ = get_llm_provider(model=model)
        return custom_llm_provider == "anthropic"
    except Exception:
        return False


class ClaudeCodePromptCacheGuardrail(CustomGuardrail):
    """
    Guardrail that injects Anthropic prompt-caching headers into system messages.

    Targets only Anthropic API models — the provider is detected from the
    model string so this guardrail is safe to apply globally without
    restricting other providers.
    """

    def __init__(self, **kwargs):
        if "supported_event_hooks" not in kwargs:
            kwargs["supported_event_hooks"] = [GuardrailEventHooks.pre_call]
        super().__init__(**kwargs)
        verbose_proxy_logger.debug("ClaudeCodePromptCacheGuardrail initialized")

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        if input_type != "request":
            return inputs

        model: Optional[str] = request_data.get("model") or inputs.get("model")  # type: ignore[assignment]
        if not _is_anthropic_model(model):
            verbose_proxy_logger.debug(
                f"ClaudeCodePromptCacheGuardrail: skipping non-Anthropic model '{model}'"
            )
            return inputs

        messages = request_data.get("messages") or []
        if not messages:
            return inputs

        control = ChatCompletionCachedContent(type="ephemeral")
        modified = []
        for msg in messages:
            if msg.get("role") == "system":
                msg = AnthropicCacheControlHook._safe_insert_cache_control_in_message(
                    message=msg,  # type: ignore[arg-type]
                    control=control,
                )
            modified.append(msg)

        request_data["messages"] = modified
        verbose_proxy_logger.debug(
            "ClaudeCodePromptCacheGuardrail: injected cache_control into system messages"
        )
        return inputs
