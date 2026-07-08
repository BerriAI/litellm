from typing import TYPE_CHECKING, Any, Optional

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .generic_guardrail_api import GenericGuardrailAPI

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def _get_config_value(litellm_params: Any, optional_params: Any, attribute_name: str) -> Optional[Any]:
    if optional_params is not None:
        value = (
            optional_params.get(attribute_name)
            if isinstance(optional_params, dict)
            else getattr(optional_params, attribute_name, None)
        )
        if value is not None:
            return value
    return getattr(litellm_params, attribute_name, None)


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    optional_params = getattr(litellm_params, "optional_params", None)

    _generic_guardrail_api_callback = GenericGuardrailAPI(
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
        headers=getattr(litellm_params, "headers", None),
        additional_provider_specific_params=getattr(litellm_params, "additional_provider_specific_params", {}),
        unreachable_fallback=getattr(litellm_params, "unreachable_fallback", "fail_closed"),
        fail_on_error=getattr(litellm_params, "fail_on_error", True),
        extra_headers=getattr(litellm_params, "extra_headers", None),
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
        streaming_end_of_stream_only=_get_config_value(litellm_params, optional_params, "streaming_end_of_stream_only"),
        streaming_sampling_rate=_get_config_value(litellm_params, optional_params, "streaming_sampling_rate"),
    )

    litellm.logging_callback_manager.add_litellm_callback(_generic_guardrail_api_callback)
    return _generic_guardrail_api_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.GENERIC_GUARDRAIL_API.value: initialize_guardrail,
}

guardrail_class_registry = {
    SupportedGuardrailIntegrations.GENERIC_GUARDRAIL_API.value: GenericGuardrailAPI,
}
