from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .noma import NomaGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    _noma_callback = NomaGuardrail(
        guardrail_name=guardrail.get("guardrail_name", ""),
        api_key=litellm_params.api_key,
        api_base=litellm_params.api_base,
        application_id=litellm_params.application_id,
        monitor_mode=litellm_params.monitor_mode,
        block_failures=litellm_params.block_failures,
        anonymize_input=litellm_params.anonymize_input,
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(_noma_callback)

    return _noma_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.NOMA.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.NOMA.value: NomaGuardrail,
}
