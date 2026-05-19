from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .peyeeye import PEyeEyeGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    _peyeeye_callback = PEyeEyeGuardrail(
        guardrail_name=guardrail.get("guardrail_name", ""),
        api_key=litellm_params.api_key,
        api_base=litellm_params.api_base,
        peyeeye_locale=litellm_params.peyeeye_locale,
        peyeeye_entities=litellm_params.peyeeye_entities,
        peyeeye_session_mode=litellm_params.peyeeye_session_mode,
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(_peyeeye_callback)

    return _peyeeye_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.PEYEEYE.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.PEYEEYE.value: PEyeEyeGuardrail,
}
