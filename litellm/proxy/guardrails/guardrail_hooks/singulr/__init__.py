from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .singulr import SingulrGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(
    litellm_params: "LitellmParams",
    guardrail: "Guardrail",
):
    import litellm

    _cb = SingulrGuardrail(
        singulr_api_base=getattr(litellm_params, "singulr_api_base", None) or litellm_params.api_base,
        singulr_api_key=getattr(litellm_params, "singulr_api_key", None) or litellm_params.api_key,
        singulr_application_id=getattr(litellm_params, "singulr_application_id", None),
        singulr_guardrail_id=getattr(litellm_params, "singulr_guardrail_id", None),
        block_on_error=getattr(litellm_params, "block_on_error", None),
        timeout=litellm_params.timeout,
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
    SupportedGuardrailIntegrations.SINGULR.value: initialize_guardrail,
}

guardrail_class_registry = {
    SupportedGuardrailIntegrations.SINGULR.value: SingulrGuardrail,
}
