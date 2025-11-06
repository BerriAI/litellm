from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .pointguardai import PointGuardAIGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    if not litellm_params.api_key:
        raise ValueError("PointGuardAI: api_key is required")
    if not litellm_params.api_base:
        raise ValueError("PointGuardAI: api_base is required")
    if not litellm_params.api_email:
        raise ValueError("PointGuardAI: api_email is required")
    if not litellm_params.org_code:
        raise ValueError("PointGuardAI: org_code is required")
    if not litellm_params.policy_config_name:
        raise ValueError("PointGuardAI: policy_config_name is required")

    guardrail_name = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError("PointGuardAI: guardrail_name is required")

    pointguardai_guardrail = PointGuardAIGuardrail(
        guardrail_name=guardrail_name,
        **{
            **litellm_params.model_dump(exclude_none=True),
            "api_key": litellm_params.api_key,
            "api_base": litellm_params.api_base,
            "api_email": litellm_params.api_email,
            "org_code": litellm_params.org_code,
            "policy_config_name": litellm_params.policy_config_name,
            "model_provider_name": getattr(litellm_params, "model_provider_name", None),
            "model_name": getattr(litellm_params, "model_name", None),
            "default_on": litellm_params.default_on,
            "event_hook": litellm_params.mode,
        },
    )

    litellm.logging_callback_manager.add_litellm_callback(pointguardai_guardrail)
    return pointguardai_guardrail


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.POINTGUARDAI.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.POINTGUARDAI.value: PointGuardAIGuardrail,
}


__all__ = ["PointGuardAIGuardrail", "initialize_guardrail", "guardrail_initializer_registry", "guardrail_class_registry"]
