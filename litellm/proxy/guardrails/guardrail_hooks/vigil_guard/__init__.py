from typing import TYPE_CHECKING, Final

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .vigil_guard import VigilGuardGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    _vigil_guard_callback: Final = VigilGuardGuardrail(
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
        unreachable_fallback=litellm_params.unreachable_fallback,
        timeout=litellm_params.timeout,
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(_vigil_guard_callback)
    return _vigil_guard_callback


guardrail_initializer_registry: Final = {
    SupportedGuardrailIntegrations.VIGIL_GUARD.value: initialize_guardrail,
}


guardrail_class_registry: Final = {
    SupportedGuardrailIntegrations.VIGIL_GUARD.value: VigilGuardGuardrail,
}
