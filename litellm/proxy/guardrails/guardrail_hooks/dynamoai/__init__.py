from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .dynamoai import DynamoAIGuardrails

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm
    _dynamoai_callback = DynamoAIGuardrails(

    _dynamoai_callback = DynamoAIGuardrails(
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(_dynamoai_callback)

    return _dynamoai_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.DYNAMOAI.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.DYNAMOAI.value: DynamoAIGuardrails,
}
