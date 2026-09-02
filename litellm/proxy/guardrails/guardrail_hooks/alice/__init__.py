from typing import TYPE_CHECKING, Final

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .alice import AliceGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    _alice_guardrail_callback: Final = AliceGuardrail(
        api_key=litellm_params.api_key,
        api_base=litellm_params.api_base,
        unreachable_fallback=getattr(litellm_params, "unreachable_fallback", "fail_closed"),
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )

    litellm.logging_callback_manager.add_litellm_callback(_alice_guardrail_callback)
    return _alice_guardrail_callback


guardrail_initializer_registry: Final = {  # mutable-ok: module-level registry, built once and never mutated
    SupportedGuardrailIntegrations.ALICE.value: initialize_guardrail,
}


guardrail_class_registry: Final = {  # mutable-ok: module-level registry, built once and never mutated
    SupportedGuardrailIntegrations.ALICE.value: AliceGuardrail,
}
