from typing import TYPE_CHECKING, Union

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .prompt_shield import AzureContentSafetyPromptShieldGuardrail
from .text_moderation import AzureContentSafetyTextModerationGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    if not litellm_params.api_key:
        raise ValueError("Azure Content Safety: api_key is required")
    if not litellm_params.api_base:
        raise ValueError("Azure Content Safety: api_base is required")

    azure_guardrail = litellm_params.guardrail.split("/")[1]

    guardrail_name = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError("Azure Content Safety: guardrail_name is required")

    if azure_guardrail == "prompt_shield":
        azure_content_safety_guardrail: Union[
            AzureContentSafetyPromptShieldGuardrail,
            AzureContentSafetyTextModerationGuardrail,
        ] = AzureContentSafetyPromptShieldGuardrail(
            guardrail_name=guardrail_name,
            **{
                **litellm_params.model_dump(exclude_none=True),
                "api_key": litellm_params.api_key,
                "api_base": litellm_params.api_base,
                "default_on": litellm_params.default_on,
                "event_hook": litellm_params.mode,
            },
        )
    elif azure_guardrail == "text_moderations":
        azure_content_safety_guardrail = AzureContentSafetyTextModerationGuardrail(
            guardrail_name=guardrail_name,
            **{
                **litellm_params.model_dump(exclude_none=True),
                "api_key": litellm_params.api_key,
                "api_base": litellm_params.api_base,
                "default_on": litellm_params.default_on,
                "event_hook": litellm_params.mode,
            },
        )
    else:
        raise ValueError(
            f"Azure Content Safety: {azure_guardrail} is not a valid guardrail"
        )

    litellm.logging_callback_manager.add_litellm_callback(
        azure_content_safety_guardrail
    )
    return azure_content_safety_guardrail


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.AZURE_PROMPT_SHIELD.value: initialize_guardrail,
    SupportedGuardrailIntegrations.AZURE_TEXT_MODERATIONS.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.AZURE_PROMPT_SHIELD.value: AzureContentSafetyPromptShieldGuardrail,
    SupportedGuardrailIntegrations.AZURE_TEXT_MODERATIONS.value: AzureContentSafetyTextModerationGuardrail,
}
