from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .privaite import PrivaiteGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    _privaite_callback = PrivaiteGuardrail(
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
        preset=getattr(litellm_params, "preset", None),
        languages=getattr(litellm_params, "languages", None),
        deanonymize=getattr(litellm_params, "deanonymize", True),
        block_entities=getattr(litellm_params, "block_entities", None),
    )
    litellm.logging_callback_manager.add_litellm_callback(_privaite_callback)
    return _privaite_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.PRIVAITE.value: initialize_guardrail,
}

guardrail_class_registry = {
    SupportedGuardrailIntegrations.PRIVAITE.value: PrivaiteGuardrail,
}
