from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .hiddenlayer import HiddenlayerGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    api_id = litellm_params.api_id if hasattr(litellm_params, "api_id") else None
    auth_url = litellm_params.auth_url if hasattr(litellm_params, "auth_url") else None

    _hiddenlayer_callback = HiddenlayerGuardrail(
        api_base=litellm_params.api_base,
        api_id=api_id,
        api_key=litellm_params.api_key,
        auth_url=auth_url,
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )

    litellm.logging_callback_manager.add_litellm_callback(_hiddenlayer_callback)
    return _hiddenlayer_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.HIDDENLAYER.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.HIDDENLAYER.value: HiddenlayerGuardrail,
}
