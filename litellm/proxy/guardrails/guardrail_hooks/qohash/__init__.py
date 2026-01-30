from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .qohash import QohashGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    _qohash_callback = QohashGuardrail(
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
        failure_mode=getattr(litellm_params, "failure_mode", "fail_closed"),
        additional_provider_specific_params=litellm_params.additional_provider_specific_params,
    )

    litellm.logging_callback_manager.add_litellm_callback(_qohash_callback)

    return _qohash_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.QOHASH.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.QOHASH.value: QohashGuardrail,
}
