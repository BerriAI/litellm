from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .deepkeep import DeepKeepGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    _deepkeep_guardrail_callback = DeepKeepGuardrail(
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
        firewall_id=getattr(litellm_params, "deepkeep_firewall_id", None),
        unreachable_fallback=getattr(litellm_params, "unreachable_fallback", "fail_closed"),
        extra_headers=getattr(litellm_params, "extra_headers", None),
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )

    litellm.logging_callback_manager.add_litellm_callback(_deepkeep_guardrail_callback)
    return _deepkeep_guardrail_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.DEEPKEEP.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.DEEPKEEP.value: DeepKeepGuardrail,
}
