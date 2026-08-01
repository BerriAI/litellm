from typing import TYPE_CHECKING

from pydantic import TypeAdapter

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .noma import NomaGuardrail
from .noma_v2 import NomaV2Guardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    use_v2 = getattr(litellm_params, "use_v2", False)
    if isinstance(use_v2, str):
        use_v2 = use_v2.lower() == "true"
    if use_v2:
        return initialize_guardrail_v2(litellm_params=litellm_params, guardrail=guardrail)

    import litellm

    _noma_callback = NomaGuardrail(
        guardrail_name=guardrail.get("guardrail_name", ""),
        api_key=litellm_params.api_key,
        api_base=litellm_params.api_base,
        application_id=litellm_params.application_id,
        monitor_mode=litellm_params.monitor_mode,
        block_failures=litellm_params.block_failures,
        anonymize_input=litellm_params.anonymize_input,
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(_noma_callback)

    return _noma_callback


_END_OF_STREAM_ONLY_ADAPTER: TypeAdapter[bool | None] = TypeAdapter(bool | None)
_SAMPLING_RATE_ADAPTER: TypeAdapter[int | None] = TypeAdapter(int | None)


def _get_config_value(litellm_params: "LitellmParams", optional_params: object, attribute_name: str) -> object:
    if optional_params is not None:
        value = getattr(optional_params, attribute_name, None)
        if value is not None:
            return value
    return getattr(litellm_params, attribute_name, None)


def initialize_guardrail_v2(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    optional_params = getattr(litellm_params, "optional_params", None)
    end_of_stream_only = _END_OF_STREAM_ONLY_ADAPTER.validate_python(
        _get_config_value(litellm_params, optional_params, "streaming_end_of_stream_only")
    )
    sampling_rate = _SAMPLING_RATE_ADAPTER.validate_python(
        _get_config_value(litellm_params, optional_params, "streaming_sampling_rate")
    )

    _noma_v2_callback = NomaV2Guardrail(
        guardrail_name=guardrail.get("guardrail_name", ""),
        api_key=litellm_params.api_key,
        api_base=litellm_params.api_base,
        application_id=litellm_params.application_id,
        monitor_mode=litellm_params.monitor_mode,
        block_failures=litellm_params.block_failures,
        streaming_end_of_stream_only=end_of_stream_only,
        streaming_sampling_rate=sampling_rate,
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(_noma_v2_callback)

    return _noma_v2_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.NOMA.value: initialize_guardrail,
    SupportedGuardrailIntegrations.NOMA_V2.value: initialize_guardrail_v2,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.NOMA.value: NomaGuardrail,
    SupportedGuardrailIntegrations.NOMA_V2.value: NomaV2Guardrail,
}
