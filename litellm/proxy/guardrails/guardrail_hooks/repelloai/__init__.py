from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .repelloai import RepelloAIGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    _repelloai_callback = RepelloAIGuardrail(
        guardrail_name=guardrail.get("guardrail_name", ""),
        api_key=litellm_params.api_key,
        api_base=litellm_params.api_base,
        asset_id=getattr(litellm_params, "asset_id", None),
        unreachable_fallback=getattr(
            litellm_params, "unreachable_fallback", "fail_closed"
        ),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(_repelloai_callback)

    return _repelloai_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.REPELLOAI.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.REPELLOAI.value: RepelloAIGuardrail,
}
