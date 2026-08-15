from typing import TYPE_CHECKING, Any, Final

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .wingback import WingbackGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def _get_config_value(litellm_params: Any, attribute_name: str) -> Any | None:
    return getattr(litellm_params, attribute_name, None)


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    additional_params: Final = dict(
        getattr(litellm_params, "additional_provider_specific_params", None) or {}
    )
    wingback_app_id = _get_config_value(litellm_params, "wingback_app_id")
    if wingback_app_id:
        additional_params.setdefault("wingback_app_id", wingback_app_id)

    _wingback_callback: Final = WingbackGuardrail(
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
        wingback_app_id=wingback_app_id,
        additional_provider_specific_params=additional_params,
        unreachable_fallback=_get_config_value(litellm_params, "unreachable_fallback") or "fail_open",
        fail_on_error=_get_config_value(litellm_params, "fail_on_error"),
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )

    litellm.logging_callback_manager.add_litellm_callback(_wingback_callback)
    return _wingback_callback


guardrail_initializer_registry: Final = {
    SupportedGuardrailIntegrations.WINGBACK.value: initialize_guardrail,
}

guardrail_class_registry: Final = {
    SupportedGuardrailIntegrations.WINGBACK.value: WingbackGuardrail,
}
