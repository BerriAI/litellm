from __future__ import annotations

from typing import TYPE_CHECKING, Final

from litellm.types.guardrails import (
    GuardrailEventHooks,
    Mode,
    SupportedGuardrailIntegrations,
)

from .agentguards import AgentGuardsGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def _coerce_event_hook(
    mode: str | list[str] | Mode,  # mutable-ok: LitellmParams.mode is declared this way
) -> GuardrailEventHooks | list[GuardrailEventHooks] | Mode:  # mutable-ok: CustomGuardrail's event_hook type
    if isinstance(mode, Mode):
        return mode
    if isinstance(mode, list):
        return [GuardrailEventHooks(item) for item in mode]
    return GuardrailEventHooks(mode)


def initialize_guardrail(litellm_params: LitellmParams, guardrail: Guardrail) -> AgentGuardsGuardrail:
    import litellm

    _agentguards_callback: Final = AgentGuardsGuardrail(
        api_key=litellm_params.api_key,
        api_base=litellm_params.api_base,
        agentguards_use_case=litellm_params.agentguards_use_case,
        unreachable_fallback=litellm_params.unreachable_fallback,
        guardrail_name=guardrail["guardrail_name"],
        event_hook=_coerce_event_hook(litellm_params.mode),
        default_on=litellm_params.default_on or False,
    )

    litellm.logging_callback_manager.add_litellm_callback(  # pyright: ignore[reportUnknownMemberType]  # callback manager is untyped
        _agentguards_callback
    )
    return _agentguards_callback


guardrail_initializer_registry: Final = {
    SupportedGuardrailIntegrations.AGENTGUARDS.value: initialize_guardrail,
}

guardrail_class_registry: Final = {
    SupportedGuardrailIntegrations.AGENTGUARDS.value: AgentGuardsGuardrail,
}
