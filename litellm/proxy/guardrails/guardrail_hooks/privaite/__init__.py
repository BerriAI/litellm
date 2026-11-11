from __future__ import annotations

from typing import TYPE_CHECKING, Final

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .privaite import PrivaiteGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: LitellmParams, guardrail: Guardrail) -> PrivaiteGuardrail:
    import litellm

    _privaite_callback: Final = PrivaiteGuardrail(
        guardrail_name=guardrail["guardrail_name"],
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
        preset=litellm_params.preset,
        languages=litellm_params.languages,
        deanonymize=litellm_params.deanonymize,
        block_entities=litellm_params.block_entities,
    )
    litellm.logging_callback_manager.add_litellm_callback(_privaite_callback)
    return _privaite_callback


guardrail_initializer_registry: Final = {
    SupportedGuardrailIntegrations.PRIVAITE.value: initialize_guardrail,
}

guardrail_class_registry: Final = {
    SupportedGuardrailIntegrations.PRIVAITE.value: PrivaiteGuardrail,
}
