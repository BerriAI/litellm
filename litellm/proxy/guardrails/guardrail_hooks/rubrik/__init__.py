from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from litellm.integrations.rubrik import RubrikLogger

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    _rubrik_callback = RubrikLogger(
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
    )

    litellm.logging_callback_manager.add_litellm_callback(_rubrik_callback)
    return _rubrik_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.RUBRIK.value: initialize_guardrail,
}

guardrail_class_registry = {
    SupportedGuardrailIntegrations.RUBRIK.value: RubrikLogger,
}
