from __future__ import annotations

from typing import TYPE_CHECKING

from litellm.types.guardrails import (
    GuardrailEventHooks,
    Mode,
    SupportedGuardrailIntegrations,
)

from .headroom import HeadroomGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def _coerce_event_hook(
    mode: str | list[str] | Mode,
) -> GuardrailEventHooks | list[GuardrailEventHooks] | Mode:
    if isinstance(mode, Mode):
        return mode
    if isinstance(mode, list):
        return [GuardrailEventHooks(item) for item in mode]
    return GuardrailEventHooks(mode)


def initialize_guardrail(litellm_params: LitellmParams, guardrail: Guardrail) -> HeadroomGuardrail:
    import litellm

    _callback = HeadroomGuardrail(
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
        model=litellm_params.model,
        guardrail_name=guardrail["guardrail_name"],
        event_hook=_coerce_event_hook(litellm_params.mode),
        default_on=litellm_params.default_on or False,
    )
    litellm.logging_callback_manager.add_litellm_callback(  # pyright: ignore[reportUnknownMemberType]
        _callback
    )
    return _callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.HEADROOM.value: initialize_guardrail,
}

guardrail_class_registry = {
    SupportedGuardrailIntegrations.HEADROOM.value: HeadroomGuardrail,
}
