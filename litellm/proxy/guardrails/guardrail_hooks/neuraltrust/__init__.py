from __future__ import annotations

from typing import TYPE_CHECKING, Final

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .neuraltrust import NeuralTrustGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: LitellmParams, guardrail: Guardrail) -> NeuralTrustGuardrail:
    import litellm

    _callback: Final = NeuralTrustGuardrail(
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
        collector_key=litellm_params.collector_key,
        unreachable_fallback=litellm_params.unreachable_fallback,
        timeout=litellm_params.timeout,
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(_callback)
    return _callback


guardrail_initializer_registry: Final = {  # mutable-ok: guardrail_registry discovers dict registries
    SupportedGuardrailIntegrations.NEURALTRUST.value: initialize_guardrail,
}

guardrail_class_registry: Final = {  # mutable-ok: guardrail_registry discovers dict registries
    SupportedGuardrailIntegrations.NEURALTRUST.value: NeuralTrustGuardrail,
}
