from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .purview_dlp import MicrosoftPurviewDLPGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(
    litellm_params: "LitellmParams", guardrail: "Guardrail"
) -> MicrosoftPurviewDLPGuardrail:
    import litellm

    callback = MicrosoftPurviewDLPGuardrail(
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        tenant_id=litellm_params.tenant_id,
        client_id=litellm_params.client_id,
        api_key=litellm_params.api_key,
        api_base=litellm_params.api_base,
        block_on_violation=(
            litellm_params.block_on_violation
            if litellm_params.block_on_violation is not None
            else True
        ),
        sensitive_info_types=litellm_params.sensitive_info_types,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(callback)
    return callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.MICROSOFT_PURVIEW.value: initialize_guardrail,
}

guardrail_class_registry = {
    SupportedGuardrailIntegrations.MICROSOFT_PURVIEW.value: MicrosoftPurviewDLPGuardrail,
}
