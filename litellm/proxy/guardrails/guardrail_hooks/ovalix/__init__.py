"""Ovalix guardrail hook: registration and initialization for the proxy."""

from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .ovalix import OvalixGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    """Create and register an Ovalix guardrail callback from proxy config."""
    import litellm

    tracker_api_base = getattr(litellm_params, "tracker_api_base", None)
    tracker_api_key = getattr(litellm_params, "tracker_api_key", None)
    application_id = getattr(litellm_params, "application_id", None)
    pre_checkpoint_id = getattr(litellm_params, "pre_checkpoint_id", None)
    post_checkpoint_id = getattr(litellm_params, "post_checkpoint_id", None)

    _ovalix_callback = OvalixGuardrail(
        guardrail_name=guardrail.get("guardrail_name", ""),
        tracker_api_base=tracker_api_base,
        tracker_api_key=tracker_api_key,
        application_id=application_id,
        pre_checkpoint_id=pre_checkpoint_id,
        post_checkpoint_id=post_checkpoint_id,
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(_ovalix_callback)

    return _ovalix_callback


# Registry of guardrail name -> initializer for proxy config loading.
guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.OVALIX.value: initialize_guardrail,
}

# Registry of guardrail name -> guardrail class (e.g. for apply_guardrail API).
guardrail_class_registry = {
    SupportedGuardrailIntegrations.OVALIX.value: OvalixGuardrail,
}
