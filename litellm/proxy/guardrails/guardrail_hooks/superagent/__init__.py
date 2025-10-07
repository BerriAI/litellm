from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .superagent import SuperAgentGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    _superagent_callback = SuperAgentGuardrail(
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(_superagent_callback)

    return _superagent_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.SUPERAGENT.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.SUPERAGENT.value: SuperAgentGuardrail,
}
