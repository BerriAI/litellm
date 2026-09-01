from typing import TYPE_CHECKING, Final

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .wingback import WingbackGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail") -> WingbackGuardrail:
    import litellm

    _wingback_callback: Final = WingbackGuardrail(
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
        wingback_app_id=litellm_params.wingback_app_id,
        additional_provider_specific_params=litellm_params.additional_provider_specific_params,
        unreachable_fallback=litellm_params.unreachable_fallback or "fail_closed",
        fail_on_error=litellm_params.fail_on_error,
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )

    litellm.logging_callback_manager.add_litellm_callback(_wingback_callback)
    return _wingback_callback


guardrail_initializer_registry: Final = {  # mutable-ok: module registry merged at import by guardrail_registry
    SupportedGuardrailIntegrations.WINGBACK.value: initialize_guardrail,
}

guardrail_class_registry: Final = {  # mutable-ok: module registry merged at import by guardrail_registry
    SupportedGuardrailIntegrations.WINGBACK.value: WingbackGuardrail,
}
