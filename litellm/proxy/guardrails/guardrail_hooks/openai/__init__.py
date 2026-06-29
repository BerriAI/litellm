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

    optional_params = getattr(litellm_params, "optional_params", None)

    openai_moderation_guardrail = OpenAIModerationGuardrail(
        guardrail_name=guardrail_name,
        **{
            **litellm_params.model_dump(exclude_none=True),
            "api_key": litellm_params.api_key,
            "api_base": litellm_params.api_base,
            "default_on": litellm_params.default_on,
            "event_hook": litellm_params.mode,
            "model": litellm_params.model,
            "streaming_end_of_stream_only": _get_config_value(
                litellm_params, optional_params, "streaming_end_of_stream_only"
            ),
            "streaming_sampling_rate": _get_config_value(litellm_params, optional_params, "streaming_sampling_rate"),
        },
    )

    litellm.logging_callback_manager.add_litellm_callback(openai_moderation_guardrail)

    return openai_moderation_guardrail


def _get_config_value(litellm_params, optional_params, attribute_name):
    if optional_params is not None:
        value = getattr(optional_params, attribute_name, None)
        if value is not None:
            return value
    return getattr(litellm_params, attribute_name, None)


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.OPENAI_MODERATION.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.OPENAI_MODERATION.value: OpenAIModerationGuardrail,
}
