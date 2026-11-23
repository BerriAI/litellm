from __future__ import annotations

from typing import TYPE_CHECKING, Final

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .dataandmore import DataAndMoreGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(
    litellm_params: LitellmParams,
    guardrail: Guardrail,
) -> DataAndMoreGuardrail:
    import litellm

    fallback = litellm_params.unreachable_fallback
    _dataandmore_callback: Final = DataAndMoreGuardrail(
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
        additional_provider_specific_params=litellm_params.additional_provider_specific_params,
        unreachable_fallback=fallback if fallback is not None else "fail_open",
        fail_on_error=litellm_params.fail_on_error,
        extra_headers=litellm_params.extra_headers,
        guardrail_name=guardrail["guardrail_name"],
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
        timeout=litellm_params.timeout,
    )

    litellm.logging_callback_manager.add_litellm_callback(_dataandmore_callback)
    return _dataandmore_callback


guardrail_initializer_registry: Final = {
    SupportedGuardrailIntegrations.DATAANDMORE.value: initialize_guardrail,
}

guardrail_class_registry: Final = {
    SupportedGuardrailIntegrations.DATAANDMORE.value: DataAndMoreGuardrail,
}
