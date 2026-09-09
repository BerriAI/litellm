from typing import TYPE_CHECKING, Final

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .spanda import SpandaGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail") -> SpandaGuardrail:
    import litellm

    raw_uncertainty: Final = getattr(litellm_params, "uncertainty_threshold", None)
    uncertainty_threshold: Final = 0.35 if raw_uncertainty is None else raw_uncertainty

    raw_grounding: Final = getattr(litellm_params, "grounding_threshold", None)
    grounding_threshold: Final = 0.15 if raw_grounding is None else raw_grounding

    _spanda_callback: Final = SpandaGuardrail(
        api_base=getattr(litellm_params, "api_base", None),
        uncertainty_threshold=uncertainty_threshold,
        grounding_threshold=grounding_threshold,
        block_mode=bool(getattr(litellm_params, "block_mode", False)),
        guardrail_name=guardrail.get("guardrail_name", "spanda"),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(_spanda_callback)
    return _spanda_callback


guardrail_initializer_registry: Final = {  # mutable-ok: guardrail initializer mapping
    SupportedGuardrailIntegrations.SPANDA.value: initialize_guardrail,
}

guardrail_class_registry: Final = {  # mutable-ok: guardrail class registry mapping
    SupportedGuardrailIntegrations.SPANDA.value: SpandaGuardrail,
}
