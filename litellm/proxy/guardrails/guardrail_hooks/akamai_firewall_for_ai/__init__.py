from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .akamai_firewall_for_ai import AkamaiFirewallForAIGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    _akamai_callback = AkamaiFirewallForAIGuardrail(
        api_key=litellm_params.api_key,
        api_base=litellm_params.api_base,
        fai_configuration_id=litellm_params.get("fai_configuration_id"),
        user_application_id=litellm_params.get("user_application_id"),
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(_akamai_callback)

    return _akamai_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.AKAMAI_FIREWALL_FOR_AI.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.AKAMAI_FIREWALL_FOR_AI.value: AkamaiFirewallForAIGuardrail,
}
