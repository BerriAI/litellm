from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .fiddler_guardrail import FiddlerGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    extra = getattr(litellm_params, "additional_provider_specific_params", None) or {}

    _fiddler_callback = FiddlerGuardrail(
        api_key=litellm_params.api_key,
        api_base=litellm_params.api_base,
        safety_threshold=extra.get("safety_threshold", 0.5),
        pii_threshold=extra.get("pii_threshold", 0.5),
        faithfulness_threshold=extra.get("faithfulness_threshold", 0.5),
        enable_safety=extra.get("enable_safety", True),
        enable_pii=extra.get("enable_pii", True),
        enable_faithfulness=extra.get("enable_faithfulness", True),
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(_fiddler_callback)
    return _fiddler_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.FIDDLER.value: initialize_guardrail,
}

guardrail_class_registry = {
    SupportedGuardrailIntegrations.FIDDLER.value: FiddlerGuardrail,
}
