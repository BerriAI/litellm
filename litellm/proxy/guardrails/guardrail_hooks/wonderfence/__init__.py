"""Alice WonderFence guardrail integration for LiteLLM."""

from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .wonderfence import (
    WonderFenceGuardrail,
    WonderFenceMissingSecrets,
)

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(
    litellm_params: "LitellmParams", guardrail: "Guardrail"
) -> WonderFenceGuardrail:
    import litellm

    guardrail_name = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError("WonderFence guardrail requires a guardrail_name")

    optional_params = getattr(litellm_params, "optional_params", None)

    wonderfence_guardrail = WonderFenceGuardrail(
        guardrail_name=guardrail_name,
        api_key=litellm_params.api_key or "",
        api_base=_get_config_value(litellm_params, optional_params, "api_base"),
        app_name=_get_config_value(litellm_params, optional_params, "app_name"),
        api_timeout=_get_config_value(litellm_params, optional_params, "api_timeout")
        or 10.0,
        platform=_get_config_value(litellm_params, optional_params, "platform"),
        retry_max=_get_config_value(litellm_params, optional_params, "retry_max"),
        retry_base_delay=_get_config_value(
            litellm_params, optional_params, "retry_base_delay"
        ),
        event_hook=litellm_params.mode,  # type: ignore[arg-type]
        default_on=litellm_params.default_on if litellm_params.default_on is not None else True,
    )

    litellm.logging_callback_manager.add_litellm_callback(wonderfence_guardrail)
    return wonderfence_guardrail


def _get_config_value(litellm_params, optional_params, attribute_name):
    if optional_params is not None:
        value = getattr(optional_params, attribute_name, None)
        if value is not None:
            return value
    return getattr(litellm_params, attribute_name, None)


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.WONDERFENCE.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.WONDERFENCE.value: WonderFenceGuardrail,
}


__all__ = [
    "WonderFenceGuardrail",
    "WonderFenceMissingSecrets",
    "initialize_guardrail",
]
