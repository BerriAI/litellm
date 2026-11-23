"""Epistemic Guard guardrail: deterministic post-call tool-receipt verification."""

from typing import TYPE_CHECKING, Final, Literal, cast

from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.types.guardrails import GuardrailEventHooks, SupportedGuardrailIntegrations

from .epistemic_guard import EpistemicGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams

DEFAULT_EVENT_HOOKS: Final = [
    GuardrailEventHooks.post_call.value,
]


def _get_param(
    litellm_params: "LitellmParams",
    guardrail: "Guardrail",
    key: str,
    default: object = None,
) -> object:
    """Get a param from litellm_params with fallback to raw guardrail litellm_params."""
    value: Final = getattr(litellm_params, key, default)
    if value is not None:
        return value
    raw: Final = as_str_mapping(guardrail.get("litellm_params"))
    if isinstance(raw, dict) and key in raw:
        return raw[key]
    return default


def initialize_guardrail(
    litellm_params: "LitellmParams",
    guardrail: "Guardrail",
) -> EpistemicGuardrail:
    """Initialize the Epistemic Guard guardrail from LiteLLM proxy config."""
    import litellm

    guardrail_name: Final = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError("Epistemic Guard guardrail requires a guardrail_name")

    on_fail: Final = cast(
        Literal["block", "rewrite", "log_only"],
        _get_param(litellm_params, guardrail, "on_fail", "block"),
    )
    verify_output_values: Final = bool(
        _get_param(litellm_params, guardrail, "verify_output_values", False)
    )
    allow_creative_context: Final = bool(
        _get_param(litellm_params, guardrail, "allow_creative_context", False)
    )
    raw_device: Final = _get_param(litellm_params, guardrail, "device", None)
    device: Final[str | None] = str(raw_device) if raw_device is not None else None
    mode: Final = _get_param(litellm_params, guardrail, "mode")
    event_hook: Final = cast(
        Literal["post_call"] | list[str] | None,
        mode if mode is not None else DEFAULT_EVENT_HOOKS,
    )

    instance: Final = EpistemicGuardrail(
        guardrail_name=guardrail_name,
        on_fail=on_fail,
        verify_output_values=verify_output_values,
        allow_creative_context=allow_creative_context,
        device=device,
        event_hook=event_hook,
        default_on=bool(_get_param(litellm_params, guardrail, "default_on", False)),
    )
    litellm.logging_callback_manager.add_litellm_callback(instance)
    return instance


guardrail_initializer_registry: Final = {
    SupportedGuardrailIntegrations.EPISTEMIC_GUARD.value: initialize_guardrail,
}

guardrail_class_registry: Final = {
    SupportedGuardrailIntegrations.EPISTEMIC_GUARD.value: EpistemicGuardrail,
}

__all__ = [
    "EpistemicGuardrail",
    "initialize_guardrail",
]
