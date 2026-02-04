"""Custom code guardrail integration for LiteLLM.

This module allows users to write custom guardrail logic using Python-like code
that runs in a sandboxed environment with access to LiteLLM-provided primitives.
"""

from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .custom_code_guardrail import CustomCodeGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(
    litellm_params: "LitellmParams", guardrail: "Guardrail"
) -> CustomCodeGuardrail:
    """
    Initialize a custom code guardrail.

    Args:
        litellm_params: Configuration parameters including the custom code
        guardrail: The guardrail configuration dict

    Returns:
        CustomCodeGuardrail instance
    """
    import litellm

    guardrail_name = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError("Custom code guardrail requires a guardrail_name")

    # Get the custom code from litellm_params
    custom_code = getattr(litellm_params, "custom_code", None)
    if not custom_code:
        raise ValueError(
            "Custom code guardrail requires 'custom_code' in litellm_params"
        )

    custom_code_guardrail = CustomCodeGuardrail(
        guardrail_name=guardrail_name,
        custom_code=custom_code,
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )

    litellm.logging_callback_manager.add_litellm_callback(custom_code_guardrail)
    return custom_code_guardrail


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.CUSTOM_CODE.value: initialize_guardrail,
}

guardrail_class_registry = {
    SupportedGuardrailIntegrations.CUSTOM_CODE.value: CustomCodeGuardrail,
}

__all__ = [
    "CustomCodeGuardrail",
    "initialize_guardrail",
]
