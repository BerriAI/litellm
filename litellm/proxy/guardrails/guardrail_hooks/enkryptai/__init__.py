from .enkryptai import EnkryptAIGuardrails

__all__ = ["EnkryptAIGuardrails"]


from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    _enkryptai_callback = EnkryptAIGuardrails(
        guardrail_name=guardrail.get("guardrail_name", ""),
        api_key=litellm_params.api_key,
        api_base=litellm_params.api_base,
        policy_name=litellm_params.policy_name,
        deployment_name=litellm_params.deployment_name,
        detectors=litellm_params.detectors,
        block_on_violation=litellm_params.block_on_violation,
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(_enkryptai_callback)

    return _enkryptai_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.ENKRYPTAI.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.ENKRYPTAI.value: EnkryptAIGuardrails,
}

