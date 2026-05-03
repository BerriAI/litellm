"""Natasha-based Russian person-name (PER) masking for the LiteLLM proxy."""

from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .natasha_ru_person import NatashaRussianPersonGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(
    litellm_params: "LitellmParams", guardrail: "Guardrail"
) -> NatashaRussianPersonGuardrail:
    import litellm
    from litellm.types.guardrails import GuardrailEventHooks

    guardrail_name = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError("natasha_ru_person guardrail requires guardrail_name")

    mode = litellm_params.mode
    if isinstance(mode, list):
        mode = mode[0] if mode else GuardrailEventHooks.pre_call.value
    if isinstance(mode, str):
        event_hook: GuardrailEventHooks | str = GuardrailEventHooks(mode)
    else:
        event_hook = GuardrailEventHooks.pre_call

    optional = getattr(litellm_params, "optional_params", None)
    redaction_placeholder = None
    if optional is not None:
        redaction_placeholder = getattr(
            optional, "natasha_redaction_placeholder", None
        ) or getattr(optional, "natasha_ru_person_redaction_placeholder", None)

    cb = NatashaRussianPersonGuardrail(
        guardrail_name=guardrail_name,
        event_hook=event_hook,
        default_on=bool(litellm_params.default_on),
        redaction_placeholder=redaction_placeholder,
    )
    litellm.logging_callback_manager.add_litellm_callback(cb)
    return cb


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.NATASHA_RU_PERSON.value: initialize_guardrail,
}

guardrail_class_registry = {
    SupportedGuardrailIntegrations.NATASHA_RU_PERSON.value: NatashaRussianPersonGuardrail,
}
