from typing import TYPE_CHECKING, Union

from litellm.types.guardrails import (
    GuardrailEventHooks,
    Mode,
    SupportedGuardrailIntegrations,
)

from .repelloai import RepelloAIGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def _event_hook_from_mode(
    mode: str | list[str] | Mode,
) -> Union[GuardrailEventHooks, list[GuardrailEventHooks], Mode]:
    if isinstance(mode, Mode):
        return mode
    if isinstance(mode, list):
        return [GuardrailEventHooks(item) for item in mode]
    return GuardrailEventHooks(mode)


def initialize_guardrail(
    litellm_params: "LitellmParams", guardrail: "Guardrail"
) -> RepelloAIGuardrail:
    import litellm

    _repelloai_callback = RepelloAIGuardrail(
        guardrail_name=guardrail["guardrail_name"],
        api_key=litellm_params.api_key,
        api_base=litellm_params.api_base,
        asset_id=litellm_params.asset_id,
        unreachable_fallback=litellm_params.unreachable_fallback,
        event_hook=_event_hook_from_mode(litellm_params.mode),
        default_on=litellm_params.default_on or False,
    )
    litellm.logging_callback_manager.add_litellm_callback(_repelloai_callback)

    return _repelloai_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.REPELLOAI.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.REPELLOAI.value: RepelloAIGuardrail,
}
