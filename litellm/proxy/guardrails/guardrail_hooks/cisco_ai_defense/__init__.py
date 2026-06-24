"""Cisco AI Defense Guardrail Integration for LiteLLM."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from litellm.types.guardrails import SupportedGuardrailIntegrations
from litellm.types.proxy.guardrails.guardrail_hooks.cisco_ai_defense import (
    CiscoAIDefenseGuardrailConfigModelOptionalParams,
    CiscoAIDefenseRule,
)

from .cisco_ai_defense import (
    CiscoAIDefenseGuardrail,
    CiscoAIDefenseGuardrailAPIError,
    CiscoAIDefenseGuardrailMissingSecrets,
)

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(
    litellm_params: "LitellmParams", guardrail: "Guardrail"
) -> CiscoAIDefenseGuardrail:
    import litellm

    guardrail_name = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError("Cisco AI Defense: guardrail_name is required")

    optional_params = _resolve_optional_params(litellm_params)
    # fmt: off
    enabled_rules = _dump_rules(optional_params.enabled_rules)  # any-ok: Pydantic model_dump returns dict[str, Any] across the typed/untyped boundary
    # fmt: on

    _callback = CiscoAIDefenseGuardrail(
        guardrail_name=guardrail_name,
        api_key=litellm_params.api_key,
        api_base=litellm_params.api_base,
        inspection_type=optional_params.inspection_type,
        inspect_path=optional_params.inspect_path,
        enabled_rules=enabled_rules,  # any-ok: Pydantic dict[str, Any] propagates from _dump_rules across the typed/untyped boundary
        integration_profile_id=optional_params.integration_profile_id,
        integration_profile_version=optional_params.integration_profile_version,
        integration_tenant_id=optional_params.integration_tenant_id,
        integration_type=optional_params.integration_type,
        on_flagged_action=optional_params.on_flagged_action,
        fallback_on_error=optional_params.fallback_on_error,
        timeout=optional_params.timeout,
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on or False,
    )
    litellm.logging_callback_manager.add_litellm_callback(_callback)

    # MCP post-tool-call hooks are dispatched through success callbacks.
    litellm.logging_callback_manager.add_litellm_success_callback(_callback)

    return _callback


def _resolve_optional_params(
    litellm_params: "LitellmParams",
) -> CiscoAIDefenseGuardrailConfigModelOptionalParams:
    """Resolve Cisco optional params by merging explicitly-set flat fields on
    ``litellm_params`` with the nested ``optional_params``. Nested explicit
    values override flat ones (matching the original lookup priority); only
    fields the user actually set are forwarded so sibling guardrails' defaults
    inherited via MRO on ``LitellmParams`` are not picked up."""
    return CiscoAIDefenseGuardrailConfigModelOptionalParams.model_validate(
        _explicit_overrides(litellm_params)
    )


def _explicit_overrides(litellm_params: "LitellmParams") -> Dict[str, object]:
    """Collect Cisco optional-param overrides the user actually set, suitable
    for Pydantic validation. Crosses a typed/untyped boundary because
    ``model_dump`` and dynamic dict access on untyped sources return ``Any``."""
    cisco_fields = CiscoAIDefenseGuardrailConfigModelOptionalParams.model_fields.keys()
    flat_keys = litellm_params.model_fields_set & cisco_fields
    # fmt: off
    merged: Dict[str, object] = dict(litellm_params.model_dump(include=flat_keys))  # any-ok: Pydantic model_dump returns dict[str, Any] across the typed/untyped boundary
    nested = litellm_params.optional_params
    if isinstance(nested, CiscoAIDefenseGuardrailConfigModelOptionalParams):
        merged.update(nested.model_dump(exclude_unset=True, exclude_none=True))  # any-ok: Pydantic model_dump returns dict[str, Any] across the typed/untyped boundary
    elif isinstance(nested, dict):
        for key, value in nested.items():
            if key in cisco_fields:
                merged[key] = value  # any-ok: nested optional_params dict is untyped at the user-input boundary
    # fmt: on
    return merged


def _dump_rules(
    rules: Optional[List[CiscoAIDefenseRule]],
) -> Optional[List[Dict[str, Any]]]:
    if not rules:
        return None
    # fmt: off
    return [rule.model_dump() for rule in rules]  # any-ok: Pydantic model_dump returns dict[str, Any] across the typed/untyped boundary
    # fmt: on


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
