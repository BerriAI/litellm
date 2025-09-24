from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .guardrails_ai import GuardrailsAI

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    if litellm_params.guard_name is None:
        raise Exception(
            "GuardrailsAIException - Please pass the Guardrails AI guard name via 'litellm_params::guard_name'"
        )

    _guardrails_ai_callback = GuardrailsAI(
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
        guard_name=litellm_params.guard_name,
        guardrails_ai_api_input_format=getattr(
            litellm_params, "guardrails_ai_api_input_format", "llmOutput"
        ),
    )
    litellm.logging_callback_manager.add_litellm_callback(_guardrails_ai_callback)

    return _guardrails_ai_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.GUARDRAILS_AI.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.GUARDRAILS_AI.value: GuardrailsAI,
}
