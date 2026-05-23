"""Sentinel LLM Fortress V2 guardrail integration for LiteLLM."""

from typing import TYPE_CHECKING, Optional

from .sentinel_fortress import (
    SentinelFortressGuardrail,
    SentinelFortressConfigError,
    SentinelFortressMissingDependency,
)

if TYPE_CHECKING:
    from litellm import Router
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(
    litellm_params: "LitellmParams",
    guardrail: "Guardrail",
    llm_router: Optional["Router"] = None,
) -> SentinelFortressGuardrail:
    """
    Initialize the Sentinel Fortress V2 guardrail.

    Args:
        litellm_params: Configuration parameters from the guardrail config
        guardrail: The guardrail definition
        llm_router: Optional LLM router for load balancing

    Returns:
        SentinelFortressGuardrail instance
    """
    import litellm

    guardrail_name = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError("Sentinel Fortress guardrail requires a guardrail_name")

    optional_params = getattr(litellm_params, "optional_params", None)

    sentinel_guardrail = SentinelFortressGuardrail(
        guardrail_name=guardrail_name,
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
        # Sentinel Fortress specific params
        enabled_detectors=_get_config_value(
            litellm_params, optional_params, "enabled_detectors"
        ),
        detector_config=_get_config_value(
            litellm_params, optional_params, "detector_config"
        ),
        on_flagged_action=_get_config_value(
            litellm_params, optional_params, "on_flagged_action"
        )
        or "block",
        pii_entities_config=_get_config_value(
            litellm_params, optional_params, "pii_entities_config"
        ),
        mask_request_content=litellm_params.mask_request_content or False,
        mask_response_content=litellm_params.mask_response_content or False,
        presidio_language=_get_config_value(
            litellm_params, optional_params, "presidio_language"
        )
        or "en",
        sentinel_config_path=_get_config_value(
            litellm_params, optional_params, "sentinel_config_path"
        ),
        violation_message_template=litellm_params.violation_message_template,
    )

    litellm.logging_callback_manager.add_litellm_callback(sentinel_guardrail)
    return sentinel_guardrail


def _get_config_value(litellm_params, optional_params, attribute_name):
    """Helper to get config value from either litellm_params or optional_params."""
    if optional_params is not None:
        value = getattr(optional_params, attribute_name, None)
        if value is not None:
            return value
    return getattr(litellm_params, attribute_name, None)


# Register the guardrail initializer
guardrail_initializer_registry = {
    "sentinel_fortress": initialize_guardrail,
}

# Register the guardrail class for the registry
guardrail_class_registry = {
    "sentinel_fortress": SentinelFortressGuardrail,
}

__all__ = [
    "SentinelFortressGuardrail",
    "SentinelFortressConfigError",
    "SentinelFortressMissingDependency",
    "initialize_guardrail",
    "guardrail_initializer_registry",
    "guardrail_class_registry",
]
