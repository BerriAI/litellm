from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .lasso import LassoGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    _lasso_callback = LassoGuardrail(
        guardrail_name=guardrail.get("guardrail_name", ""),
        api_key=litellm_params.api_key,
        api_base=litellm_params.api_base,
        user_id=litellm_params.lasso_user_id,
        conversation_id=litellm_params.lasso_conversation_id,
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(_lasso_callback)

    return _lasso_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.LASSO.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.LASSO.value: LassoGuardrail,
}
