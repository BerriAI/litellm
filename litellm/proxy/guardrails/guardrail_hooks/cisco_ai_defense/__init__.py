"""Cisco AI Defense Guardrail Integration for LiteLLM."""

from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .cisco_ai_defense import (
    CiscoAIDefenseGuardrail,
    CiscoAIDefenseGuardrailAPIError,
    CiscoAIDefenseGuardrailMissingSecrets,
)

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    guardrail_name = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError("Cisco AI Defense: guardrail_name is required")

    optional_params = getattr(litellm_params, "optional_params", None)

    _callback = CiscoAIDefenseGuardrail(
        guardrail_name=guardrail_name,
        api_key=litellm_params.api_key,
        api_base=litellm_params.api_base,
        inspection_type=_get_optional_value(
            litellm_params, optional_params, "inspection_type"
        ),
        inspect_path=_get_optional_value(
            litellm_params, optional_params, "inspect_path"
        ),
        enabled_rules=_get_optional_value(
            litellm_params, optional_params, "enabled_rules"
        ),
        integration_profile_id=_get_optional_value(
            litellm_params, optional_params, "integration_profile_id"
        ),
        integration_profile_version=_get_optional_value(
            litellm_params, optional_params, "integration_profile_version"
        ),
        integration_tenant_id=_get_optional_value(
            litellm_params, optional_params, "integration_tenant_id"
        ),
        integration_type=_get_optional_value(
            litellm_params, optional_params, "integration_type"
        ),
        on_flagged_action=_get_optional_value(
            litellm_params, optional_params, "on_flagged_action"
        ),
        fallback_on_error=_get_optional_value(
            litellm_params, optional_params, "fallback_on_error"
        ),
        timeout=_get_optional_value(litellm_params, optional_params, "timeout"),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on or False,
    )
    # Register on litellm.callbacks so the proxy's pre_call / during_call /
    # post_call guardrail dispatch picks us up (this is how every other
    # guardrail wires itself in).
    litellm.logging_callback_manager.add_litellm_callback(_callback)

    # ALSO register on litellm.success_callback so the MCP post-tool-call
    # dispatcher in litellm_logging.async_post_mcp_tool_call_hook reaches
    # us. That dispatcher only iterates ``litellm.success_callback`` (via
    # ``get_combined_callback_list(global_callbacks=litellm.success_callback)``)
    # — without this second registration, mcp-mode Cisco guardrails would
    # silently skip MCP response scanning even though the handler defines
    # ``async_post_mcp_tool_call_hook``. The other CustomLogger methods
    # inherited by CustomGuardrail are no-ops by default, so adding to
    # success_callback doesn't introduce side effects on chat completions.
    litellm.logging_callback_manager.add_litellm_success_callback(_callback)

    return _callback


def _get_optional_value(litellm_params, optional_params, attribute_name):
    """Resolve a Cisco-specific field with two-tier precedence.

    Lookup order:

    1. ``optional_params`` (nested, canonical Cisco location). Supports both
       the typed ``CiscoAIDefenseGuardrailConfigModelOptionalParams`` model
       and a plain ``dict``.
    2. Top-level ``litellm_params`` — but **only** when the field is in
       ``litellm_params.model_fields_set``, i.e. the user explicitly passed
       it at the flattened root (e.g. via the Admin UI / management API,
       which currently flattens Cisco fields onto ``LitellmParams``).

    The ``model_fields_set`` check is critical. Several shared field names
    (``on_flagged_action``, ``fallback_on_error``, ``timeout``) are also
    declared on sibling guardrail config models with their own defaults
    (e.g. GraySwan defaults ``on_flagged_action`` to ``"passthrough"``).
    Because all guardrail configs are mixed into ``LitellmParams`` via
    multiple inheritance, a bare ``getattr(litellm_params, attr)`` would
    silently inherit those sibling defaults even when the user never set
    the field. ``model_fields_set`` is the Pydantic v2-supplied set of
    fields that were actually passed at construction time, so it gives us
    a precise "user really set this" signal without false positives from
    MRO-inherited defaults.

    ``api_key`` / ``api_base`` / ``mode`` / ``default_on`` are read from
    ``litellm_params`` directly outside this helper because they ARE
    intentional top-level fields with no sibling-default ambiguity.
    """
    if optional_params is not None:
        if isinstance(optional_params, dict):
            if attribute_name in optional_params:
                return optional_params[attribute_name]
        else:
            nested_fields_set = getattr(optional_params, "model_fields_set", None)
            if nested_fields_set is None or attribute_name in nested_fields_set:
                value = getattr(optional_params, attribute_name, None)
                if value is not None:
                    return value

    # Fallback: honor a user-explicit top-level setting (flattened config).
    if litellm_params is None:
        return None
    fields_set = getattr(litellm_params, "model_fields_set", None)
    if fields_set is None or attribute_name not in fields_set:
        return None
    return getattr(litellm_params, attribute_name, None)


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.CISCO_AI_DEFENSE.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.CISCO_AI_DEFENSE.value: CiscoAIDefenseGuardrail,
}


__all__ = [
    "CiscoAIDefenseGuardrail",
    "CiscoAIDefenseGuardrailAPIError",
    "CiscoAIDefenseGuardrailMissingSecrets",
    "initialize_guardrail",
    "guardrail_initializer_registry",
    "guardrail_class_registry",
]
