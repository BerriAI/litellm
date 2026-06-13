"""Bastion Prompt Protection guardrail registration for the LiteLLM proxy.

Discovered automatically by ``get_guardrail_initializer_from_hooks()`` via the
``guardrail_initializer_registry`` / ``guardrail_class_registry`` dicts below.
"""

from typing import TYPE_CHECKING, Any, List, Optional, Union, cast

from litellm.types.guardrails import (
    GuardrailEventHooks,
    SupportedGuardrailIntegrations,
)

from .bastion import DEFAULT_VIOLATION_MESSAGE, BastionGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams

# Bastion screens user input by default; add "post_call" to also screen responses.
DEFAULT_EVENT_HOOKS = [GuardrailEventHooks.pre_call.value]


def _get_param(
    litellm_params: "LitellmParams",
    guardrail: "Guardrail",
    key: str,
    default: Any = None,
) -> Any:
    """Read a param from LitellmParams, falling back to the raw guardrail dict."""
    value = getattr(litellm_params, key, default)
    if value is not None:
        return value
    raw = guardrail.get("litellm_params")
    if isinstance(raw, dict) and key in raw:
        return raw[key]
    return default


def initialize_guardrail(
    litellm_params: "LitellmParams",
    guardrail: "Guardrail",
) -> BastionGuardrail:
    """Initialize the Bastion guardrail from proxy config."""
    import litellm

    guardrail_name = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError("Bastion guardrail requires a guardrail_name")

    preset = cast(str, _get_param(litellm_params, guardrail, "preset", "tiny"))

    threshold_raw = _get_param(litellm_params, guardrail, "threshold")
    threshold: Optional[float] = (
        float(threshold_raw) if threshold_raw is not None else None
    )

    violation_message = cast(
        str,
        _get_param(
            litellm_params, guardrail, "violation_message", DEFAULT_VIOLATION_MESSAGE
        ),
    )

    # `mode` is a first-class field on LitellmParams (read it directly, like the
    # other guardrail initializers do) rather than via the raw-dict fallback.
    mode = getattr(litellm_params, "mode", None)
    event_hook = cast(
        Optional[Union[str, List[str]]],
        mode if mode is not None else DEFAULT_EVENT_HOOKS,
    )

    instance = BastionGuardrail(
        guardrail_name=guardrail_name,
        preset=preset,
        threshold=threshold,
        violation_message=violation_message,
        event_hook=event_hook,
        default_on=bool(_get_param(litellm_params, guardrail, "default_on", False)),
    )
    litellm.logging_callback_manager.add_litellm_callback(instance)
    return instance


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.BASTION.value: initialize_guardrail,
}

guardrail_class_registry = {
    SupportedGuardrailIntegrations.BASTION.value: BastionGuardrail,
}

__all__ = [
    "BastionGuardrail",
    "initialize_guardrail",
]
