from typing import TYPE_CHECKING, Final

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .beyondguard import BeyondGuardGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail") -> BeyondGuardGuardrail:
    import litellm

    callback: Final = BeyondGuardGuardrail(
        api_key=litellm_params.api_key,
        api_base=litellm_params.api_base,
        beyondguard_endpoint_id=litellm_params.beyondguard_endpoint_id,
        beyondguard_output_endpoint_id=litellm_params.beyondguard_output_endpoint_id,
        beyondguard_output_api_key=litellm_params.beyondguard_output_api_key,
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
        timeout=litellm_params.timeout,
    )
    litellm.logging_callback_manager.add_litellm_callback(callback)
    return callback


guardrail_initializer_registry: Final = {
    SupportedGuardrailIntegrations.BEYONDGUARD.value: initialize_guardrail,
}

guardrail_class_registry: Final = {
    SupportedGuardrailIntegrations.BEYONDGUARD.value: BeyondGuardGuardrail,
}
