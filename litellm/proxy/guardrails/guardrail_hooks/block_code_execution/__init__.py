"""Block Code Execution guardrail: blocks or masks fenced code blocks by language."""

from typing import TYPE_CHECKING, Literal, cast

from litellm.types.guardrails import (GuardrailEventHooks,
                                      SupportedGuardrailIntegrations)

from .block_code_execution import BlockCodeExecutionGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams

# Default: run on both request and response (and during_call is supported too)
DEFAULT_EVENT_HOOKS = [
    GuardrailEventHooks.pre_call.value,
    GuardrailEventHooks.post_call.value,
]


def initialize_guardrail(
    litellm_params: "LitellmParams",
    guardrail: "Guardrail",
) -> BlockCodeExecutionGuardrail:
    """Initialize the Block Code Execution guardrail from config."""
    import litellm

    guardrail_name = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError(
            "Block Code Execution guardrail requires a guardrail_name"
        )

    blocked_languages = getattr(litellm_params, "blocked_languages", None)
    action = cast(
        Literal["block", "mask"], getattr(litellm_params, "action", "block")
    )
    confidence_threshold = float(
        getattr(litellm_params, "confidence_threshold", 0.7)
    )
    mode = getattr(litellm_params, "mode", None)
    event_hook = mode if mode is not None else DEFAULT_EVENT_HOOKS

    instance = BlockCodeExecutionGuardrail(
        guardrail_name=guardrail_name,
        blocked_languages=blocked_languages,
        action=action,
        confidence_threshold=confidence_threshold,
        event_hook=event_hook,
        default_on=getattr(litellm_params, "default_on", False),
    )
    litellm.logging_callback_manager.add_litellm_callback(instance)
    return instance


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.BLOCK_CODE_EXECUTION.value: initialize_guardrail,
}

guardrail_class_registry = {
    SupportedGuardrailIntegrations.BLOCK_CODE_EXECUTION.value: BlockCodeExecutionGuardrail,
}

__all__ = [
    "BlockCodeExecutionGuardrail",
    "initialize_guardrail",
]
