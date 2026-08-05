from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .thirdlaw import ThirdlawGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    _thirdlaw_callback = ThirdlawGuardrail(
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
        headers=getattr(litellm_params, "headers", None),
        guardrail_timeout=getattr(litellm_params, "guardrail_timeout", 60),
        additional_headers=getattr(litellm_params, "additional_headers", None),
        additional_provider_specific_params=getattr(
            litellm_params, "additional_provider_specific_params", {}
        ),
        unreachable_fallback=getattr(
            litellm_params, "unreachable_fallback", "fail_closed"
        ),
        extra_headers=getattr(litellm_params, "extra_headers", None),
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(_thirdlaw_callback)
    return _thirdlaw_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.THIRDLAW.value: initialize_guardrail,
}

guardrail_class_registry = {
    SupportedGuardrailIntegrations.THIRDLAW.value: ThirdlawGuardrail,
}
