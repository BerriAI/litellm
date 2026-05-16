from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .atr import ATRGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(
    litellm_params: "LitellmParams",
    guardrail: "Guardrail",
):
    import litellm

    _cb = ATRGuardrail(
        rules_path=litellm_params.rules_path,
        severity_threshold=litellm_params.severity_threshold,
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(_cb)

    return _cb


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.ATR.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.ATR.value: ATRGuardrail,
}
