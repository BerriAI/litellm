"""
Pillar Security Guardrail Integration for LiteLLM
"""

from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .pillar import (
    PillarGuardrail,
    PillarGuardrailAPIError,
    PillarGuardrailMissingSecrets,
)

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    guardrail_name = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError("Pillar guardrail name is required")

    optional_params = getattr(litellm_params, "optional_params", None)

    _pillar_callback = PillarGuardrail(
        guardrail_name=guardrail_name,
        api_key=litellm_params.api_key,
        api_base=litellm_params.api_base,
        on_flagged_action=getattr(litellm_params, "on_flagged_action", "monitor"),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
        async_mode=_get_config_value(
            litellm_params, optional_params, "async_mode"
        ),
        persist_session=_get_config_value(
            litellm_params, optional_params, "persist_session"
        ),
        include_scanners=_get_config_value(
            litellm_params, optional_params, "include_scanners"
        ),
        include_evidence=_get_config_value(
            litellm_params, optional_params, "include_evidence"
        ),
    )
    litellm.logging_callback_manager.add_litellm_callback(_pillar_callback)

    return _pillar_callback


def _get_config_value(litellm_params, optional_params, attribute_name):
    """Return guardrail configuration value prioritising optional params when present."""

    if optional_params is not None:
        value = getattr(optional_params, attribute_name, None)
        if value is not None:
            return value
    return getattr(litellm_params, attribute_name, None)


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.PILLAR.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.PILLAR.value: PillarGuardrail,
}

__all__ = [
    "PillarGuardrail",
    "PillarGuardrailAPIError",
    "PillarGuardrailMissingSecrets",
]
