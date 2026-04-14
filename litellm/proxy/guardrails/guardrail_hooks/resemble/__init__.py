from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .resemble import ResembleGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    _resemble_callback = ResembleGuardrail(
        guardrail_name=guardrail.get("guardrail_name", ""),
        api_key=litellm_params.api_key,
        api_base=litellm_params.api_base,
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
        threshold=getattr(litellm_params, "resemble_threshold", None),
        media_type=getattr(litellm_params, "resemble_media_type", None),
        audio_source_tracing=getattr(
            litellm_params, "resemble_audio_source_tracing", None
        ),
        use_reverse_search=getattr(litellm_params, "resemble_use_reverse_search", None),
        zero_retention_mode=getattr(
            litellm_params, "resemble_zero_retention_mode", None
        ),
        metadata_key=getattr(litellm_params, "resemble_metadata_key", None),
        poll_interval_seconds=getattr(
            litellm_params, "resemble_poll_interval_seconds", None
        ),
        poll_timeout_seconds=getattr(
            litellm_params, "resemble_poll_timeout_seconds", None
        ),
        fail_closed=getattr(litellm_params, "resemble_fail_closed", None),
    )
    litellm.logging_callback_manager.add_litellm_callback(_resemble_callback)

    return _resemble_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.RESEMBLE.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.RESEMBLE.value: ResembleGuardrail,
}
