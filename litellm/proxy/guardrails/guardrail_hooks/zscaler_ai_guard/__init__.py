from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .zscaler_ai_guard import ZscalerAIGuard

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    _zscaler_ai_guard_callback = ZscalerAIGuard(
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(_zscaler_ai_guard_callback)

    return _zscaler_ai_guard_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.ZSCALER_AI_GUARD.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.ZSCALER_AI_GUARD.value: ZscalerAIGuard,
}
