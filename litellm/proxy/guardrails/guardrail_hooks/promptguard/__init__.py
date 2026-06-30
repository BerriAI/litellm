from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .promptguard import PromptGuardGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(
    litellm_params: "LitellmParams",
    guardrail: "Guardrail",
):
    import litellm

    _cb = PromptGuardGuardrail(
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
        block_on_error=litellm_params.block_on_error,
        guardrail_name=guardrail.get(
            "guardrail_name",
            "",
        ),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(
        _cb,
    )

    return _cb


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.PROMPTGUARD.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.PROMPTGUARD.value: PromptGuardGuardrail,
}
