from typing import TYPE_CHECKING, Final

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .thirdlaw import ThirdlawGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail") -> ThirdlawGuardrail:
    import litellm

    # LitellmParams' multiple-inheritance MRO resolves guardrail_timeout to a
    # None default, so the 60s fallback must be applied here.
    _thirdlaw_callback: Final = ThirdlawGuardrail(
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
        additional_headers=getattr(litellm_params, "additional_headers", None),
        guardrail_timeout=getattr(litellm_params, "guardrail_timeout", None) or 60,
        streaming_buffer_until_moderated=getattr(litellm_params, "streaming_buffer_until_moderated", True),
        streaming_end_of_stream_only=getattr(litellm_params, "streaming_end_of_stream_only", True),
        streaming_sampling_rate=getattr(litellm_params, "streaming_sampling_rate", 5),
        unreachable_fallback=getattr(litellm_params, "unreachable_fallback", None) or "fail_closed",
        unscannable_stream_fallback=getattr(litellm_params, "unscannable_stream_fallback", None) or "fail_closed",
        additional_provider_specific_params=getattr(litellm_params, "additional_provider_specific_params", None),
        headers=getattr(litellm_params, "headers", None),
        extra_headers=getattr(litellm_params, "extra_headers", None),
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(_thirdlaw_callback)
    return _thirdlaw_callback


guardrail_initializer_registry: Final = {  # mutable-ok: the registry loader requires a plain dict
    SupportedGuardrailIntegrations.THIRDLAW.value: initialize_guardrail,
}

guardrail_class_registry: Final = {  # mutable-ok: the registry loader requires a plain dict
    SupportedGuardrailIntegrations.THIRDLAW.value: ThirdlawGuardrail,
}
