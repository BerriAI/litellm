from typing import TYPE_CHECKING

from litellm.types.guardrails import GuardrailEventHooks, SupportedGuardrailIntegrations

from .crowdstrike_aidr import CrowdStrikeAIDRHandler

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    guardrail_name = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError("CrowdStrike AIDR guardrail name is required")

    _crowdstrike_aidr_callback = CrowdStrikeAIDRHandler(
        guardrail_name=guardrail_name,
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
        # Exclude during_call to prevent duplicate input events
        event_hook=[
            GuardrailEventHooks.pre_call.value,
            GuardrailEventHooks.post_call.value,
        ],
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(_crowdstrike_aidr_callback)

    return _crowdstrike_aidr_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.CROWDSTRIKE_AIDR.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.CROWDSTRIKE_AIDR.value: CrowdStrikeAIDRHandler,
}
