from typing import TYPE_CHECKING, Any, Final

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .generic_guardrail_api import GenericGuardrailAPI

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def _get_config_value(litellm_params: Any, optional_params: Any, attribute_name: str) -> Any | None:
    if optional_params is not None:
        value: Final = (
            optional_params.get(attribute_name)
            if isinstance(optional_params, dict)
            else getattr(optional_params, attribute_name, None)
        )
        if value is not None:
            return value
    return getattr(litellm_params, attribute_name, None)


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    optional_params: Final = getattr(litellm_params, "optional_params", None)

    _generic_guardrail_api_callback: Final = GenericGuardrailAPI(
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
        streaming_transform_mode=_get_config_value(litellm_params, optional_params, "streaming_transform_mode"),
        fire_and_forget=_get_config_value(litellm_params, optional_params, "fire_and_forget"),
        fire_and_forget_max_inflight=_get_config_value(litellm_params, optional_params, "fire_and_forget_max_inflight"),
        send_images=_get_config_value(litellm_params, optional_params, "send_images"),
        exclude_payload_fields=_get_config_value(litellm_params, optional_params, "exclude_payload_fields"),
        max_messages=_get_config_value(litellm_params, optional_params, "max_messages"),
        max_text_chars=_get_config_value(litellm_params, optional_params, "max_text_chars"),
        strip_patterns=_get_config_value(litellm_params, optional_params, "strip_patterns"),
        skip_if_system_prompt_matches=_get_config_value(
            litellm_params, optional_params, "skip_if_system_prompt_matches"
        ),
        skip_if_first_role_in=_get_config_value(litellm_params, optional_params, "skip_if_first_role_in"),
        skip_if_key_alias_in=_get_config_value(litellm_params, optional_params, "skip_if_key_alias_in"),
        skip_if_team_id_in=_get_config_value(litellm_params, optional_params, "skip_if_team_id_in"),
        run_only_on_call_types=_get_config_value(litellm_params, optional_params, "run_only_on_call_types"),
        skip_call_types=_get_config_value(litellm_params, optional_params, "skip_call_types"),
        guardrail_information_scope=_get_config_value(litellm_params, optional_params, "guardrail_information_scope"),
    )

    litellm.logging_callback_manager.add_litellm_callback(_generic_guardrail_api_callback)
    return _generic_guardrail_api_callback


guardrail_initializer_registry: Final = {
    SupportedGuardrailIntegrations.GENERIC_GUARDRAIL_API.value: initialize_guardrail,
}

guardrail_class_registry: Final = {
    SupportedGuardrailIntegrations.GENERIC_GUARDRAIL_API.value: GenericGuardrailAPI,
}
