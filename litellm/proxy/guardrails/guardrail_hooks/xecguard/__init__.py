from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .xecguard import XecGuardGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    _xecguard_callback = XecGuardGuardrail(
        api_key=getattr(litellm_params, "api_key", None),
        api_base=getattr(litellm_params, "api_base", None),
        model=getattr(litellm_params, "model", "xecguard_v2") or "xecguard_v2",
        policy_names=getattr(litellm_params, "policy_names", None),
        grounding_enabled=getattr(litellm_params, "grounding_enabled", False) or False,
        grounding_strictness=getattr(litellm_params, "grounding_strictness", "BALANCED") or "BALANCED",
        grounding_documents=getattr(litellm_params, "grounding_documents", None),
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )

    litellm.logging_callback_manager.add_litellm_callback(_xecguard_callback)
    return _xecguard_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.XECGUARD.value: initialize_guardrail,
}

guardrail_class_registry = {
    SupportedGuardrailIntegrations.XECGUARD.value: XecGuardGuardrail,
}
