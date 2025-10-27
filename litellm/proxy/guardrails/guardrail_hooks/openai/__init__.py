from typing import TYPE_CHECKING

import litellm
from litellm.proxy.guardrails.guardrail_hooks.openai.moderations import (
    OpenAIModerationGuardrail,
)
from litellm.types.guardrails import SupportedGuardrailIntegrations

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    guardrail_name = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError("OpenAI Moderation: guardrail_name is required")
    
    openai_moderation_guardrail = OpenAIModerationGuardrail(
        guardrail_name=guardrail_name,
        **{
            **litellm_params.model_dump(exclude_none=True),
            "api_key": litellm_params.api_key,
            "api_base": litellm_params.api_base,
            "default_on": litellm_params.default_on,
            "event_hook": litellm_params.mode,
            "model": litellm_params.model,
        },
    )

    litellm.logging_callback_manager.add_litellm_callback(
        openai_moderation_guardrail
    )

    return openai_moderation_guardrail



guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.OPENAI_MODERATION.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.OPENAI_MODERATION.value: OpenAIModerationGuardrail,
}
