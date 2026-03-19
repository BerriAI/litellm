"""
OpenGuardrails Native Integration for LiteLLM

Full-featured guardrail integration supporting:
- Input/Output detection (19 risk categories + prompt injection)
- Sensitive data anonymization with restoration
- Private model switching (automatic routing to data-safe models)
- Tool call anomaly detection
- Ban policy enforcement
"""

from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .openguardrails import OpenGuardrailsGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(
    litellm_params: "LitellmParams",
    guardrail: "Guardrail",
):
    import litellm

    callback = OpenGuardrailsGuardrail(
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
        guardrail_name=guardrail.get("guardrail_name", "openguardrails"),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
        private_model_name=getattr(litellm_params, "private_model_name", None),
    )

    litellm.logging_callback_manager.add_litellm_callback(callback)
    return callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.OPENGUARDRAILS.value: initialize_guardrail,
}

guardrail_class_registry = {
    SupportedGuardrailIntegrations.OPENGUARDRAILS.value: OpenGuardrailsGuardrail,
}
