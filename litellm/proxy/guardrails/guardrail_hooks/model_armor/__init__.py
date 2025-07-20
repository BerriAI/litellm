from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .model_armor import ModelArmorGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm
    from litellm.proxy.guardrails.guardrail_hooks.model_armor import (
        ModelArmorGuardrail,
    )

    _model_armor_callback = ModelArmorGuardrail(
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        template_id=litellm_params.template_id,
        project_id=litellm_params.project_id,
        location=litellm_params.location,
        credentials=litellm_params.credentials,
        api_endpoint=litellm_params.api_endpoint,
        default_on=litellm_params.default_on,
        mask_request_content=litellm_params.mask_request_content,
        mask_response_content=litellm_params.mask_response_content,
        fail_on_error=litellm_params.fail_on_error,
    )
    litellm.logging_callback_manager.add_litellm_callback(_model_armor_callback)

    return _model_armor_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.MODEL_ARMOR.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.MODEL_ARMOR.value: ModelArmorGuardrail,
}