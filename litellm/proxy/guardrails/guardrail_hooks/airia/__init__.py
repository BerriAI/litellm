from typing import TYPE_CHECKING, Final

from litellm.proxy.guardrails.guardrail_hooks.airia.airia import AiriaGuardrail
from litellm.types.guardrails import SupportedGuardrailIntegrations

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail") -> AiriaGuardrail:
    import litellm

    _airia_callback: Final = AiriaGuardrail(
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
        timeout=litellm_params.timeout,
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(_airia_callback)

    return _airia_callback


guardrail_initializer_registry: Final = {  # mutable-ok: module-level registry, built once and never mutated
    SupportedGuardrailIntegrations.AIRIA.value: initialize_guardrail,
}

guardrail_class_registry: Final = {  # mutable-ok: module-level registry, built once and never mutated
    SupportedGuardrailIntegrations.AIRIA.value: AiriaGuardrail,
}
