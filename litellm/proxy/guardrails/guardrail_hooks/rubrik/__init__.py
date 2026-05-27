"""Rubrik guardrail integration for LiteLLM."""

from typing import TYPE_CHECKING

from litellm.integrations.rubrik import RubrikLogger
from litellm.types.guardrails import SupportedGuardrailIntegrations

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(
    litellm_params: "LitellmParams", guardrail: "Guardrail"
) -> RubrikLogger:
    import litellm

    rubrik_callback = RubrikLogger(
        api_key=litellm_params.api_key,
        api_base=litellm_params.api_base,
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )

    litellm.logging_callback_manager.add_litellm_callback(rubrik_callback)
    return rubrik_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.RUBRIK.value: initialize_guardrail,
}

guardrail_class_registry = {
    SupportedGuardrailIntegrations.RUBRIK.value: RubrikLogger,
}
