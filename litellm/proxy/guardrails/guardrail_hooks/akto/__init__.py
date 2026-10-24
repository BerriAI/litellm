from typing import TYPE_CHECKING, Final

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .akto import AktoGuardrail, streaming_sampling_rate_from

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    _akto_callback: Final = AktoGuardrail(
        akto_base_url=litellm_params.akto_base_url,
        akto_api_key=litellm_params.akto_api_key,
        akto_account_id=litellm_params.akto_account_id,
        akto_vxlan_id=litellm_params.akto_vxlan_id,
        context_source=litellm_params.context_source,
        akto_metadata=litellm_params.akto_metadata,
        streaming_sampling_rate=streaming_sampling_rate_from(litellm_params),
        guardrail_timeout=litellm_params.guardrail_timeout,
        file_guardrail_timeout=litellm_params.file_guardrail_timeout,
        unreachable_fallback=litellm_params.unreachable_fallback,
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )

    litellm.logging_callback_manager.add_litellm_callback(_akto_callback)
    return _akto_callback


guardrail_initializer_registry: Final = {
    SupportedGuardrailIntegrations.AKTO.value: initialize_guardrail,
}

guardrail_class_registry: Final = {
    SupportedGuardrailIntegrations.AKTO.value: AktoGuardrail,
}
