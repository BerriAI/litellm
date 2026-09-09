from typing import TYPE_CHECKING, Final

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .spanda import SpandaGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    _spanda_callback: Final = SpandaGuardrail(
        api_base=getattr(litellm_params, "api_base", None),
        uncertainty_threshold=getattr(litellm_params, "uncertainty_threshold", 0.35),
        grounding_threshold=getattr(litellm_params, "grounding_threshold", 0.15),
        block_mode=getattr(litellm_params, "block_mode", False),
        guardrail_name=guardrail.get("guardrail_name", "spanda"),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(_spanda_callback)
    return _spanda_callback


guardrail_initializer_registry: Final = {
    SupportedGuardrailIntegrations.SPANDA.value: initialize_guardrail,
}

guardrail_class_registry: Final = {
    SupportedGuardrailIntegrations.SPANDA.value: SpandaGuardrail,
}
