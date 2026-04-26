from typing import TYPE_CHECKING

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
        return initialize_guardrail_v2(
            litellm_params=litellm_params, guardrail=guardrail
        )

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


def initialize_guardrail_v2(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    # Forward streaming knobs from YAML litellm_params only when set, so that
    # NomaV2Guardrail.__init__ leaves them unset on self and the default values
    # in UnifiedLLMGuardrails.async_post_call_streaming_iterator_hook remain the
    # source of truth for unconfigured deployments.
    extra_kwargs: dict = {}
    streaming_end_of_stream_only = getattr(
        litellm_params, "streaming_end_of_stream_only", None
    )
    if streaming_end_of_stream_only is not None:
        extra_kwargs["streaming_end_of_stream_only"] = streaming_end_of_stream_only
    streaming_sampling_rate = getattr(litellm_params, "streaming_sampling_rate", None)
    if streaming_sampling_rate is not None:
        extra_kwargs["streaming_sampling_rate"] = streaming_sampling_rate

    _noma_v2_callback = NomaV2Guardrail(
        guardrail_name=guardrail.get("guardrail_name", ""),
        api_key=litellm_params.api_key,
        api_base=litellm_params.api_base,
        application_id=litellm_params.application_id,
        monitor_mode=litellm_params.monitor_mode,
        block_failures=litellm_params.block_failures,
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
        **extra_kwargs,
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
