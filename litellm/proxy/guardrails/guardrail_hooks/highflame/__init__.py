from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .highflame import HighflameGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    _highflame_callback = HighflameGuardrail(
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on or False,
        capabilities=getattr(litellm_params, "capabilities", None),
        application=litellm_params.application,
        shield_mode=getattr(litellm_params, "shield_mode", "enforce") or "enforce",
        token_url=getattr(litellm_params, "token_url", None),
        metadata=litellm_params.metadata,
    )
    litellm.logging_callback_manager.add_litellm_callback(_highflame_callback)

    return _highflame_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.HIGHFLAME.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.HIGHFLAME.value: HighflameGuardrail,
}
