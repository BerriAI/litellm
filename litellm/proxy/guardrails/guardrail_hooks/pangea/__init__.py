from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .pangea import PangeaHandler

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    guardrail_name = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError("Pangea guardrail name is required")

    _pangea_callback = PangeaHandler(
        guardrail_name=guardrail_name,
        pangea_input_recipe=litellm_params.pangea_input_recipe,
        pangea_output_recipe=litellm_params.pangea_output_recipe,
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(_pangea_callback)

    return _pangea_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.PANGEA.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.PANGEA.value: PangeaHandler,
}
