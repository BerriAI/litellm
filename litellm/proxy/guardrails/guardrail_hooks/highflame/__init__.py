from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .highflame import HighflameGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    if litellm_params.guard_name is None:
        raise Exception(
            "HighflameGuardrailException - Please pass the Highflame guard name via 'litellm_params::guard_name'"
        )

    _highflame_callback = HighflameGuardrail(
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
        guardrail_name=guardrail.get("guardrail_name", ""),
        highflame_guard_name=litellm_params.guard_name,
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on or False,
        api_version=litellm_params.api_version or "v1",
        config=litellm_params.config,
        metadata=litellm_params.metadata,
        application=litellm_params.application,
    )
    litellm.logging_callback_manager.add_litellm_callback(_highflame_callback)

    return _highflame_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.HIGHFLAME.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.HIGHFLAME.value: HighflameGuardrail,
}
