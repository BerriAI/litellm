from typing import TYPE_CHECKING, Literal, Set

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .silmaril import SilmarilGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def _get_silmaril_unreachable_fallback(
    litellm_params: "LitellmParams", guardrail: "Guardrail"
) -> Literal["fail_closed", "fail_open"]:
    raw_litellm_params = guardrail.get("litellm_params")
    if (
        isinstance(raw_litellm_params, dict)
        and "unreachable_fallback" in raw_litellm_params
    ):
        return litellm_params.unreachable_fallback

    fields_set: Set[str] = getattr(litellm_params, "model_fields_set", set())
    if "unreachable_fallback" in fields_set:
        return litellm_params.unreachable_fallback

    return "fail_open"


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    _instance = SilmarilGuardrail(
        api_base=getattr(litellm_params, "api_base", None),
        api_key=getattr(litellm_params, "api_key", None),
        headers=getattr(litellm_params, "headers", None),
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
        additional_provider_specific_params=getattr(
            litellm_params, "additional_provider_specific_params", None
        ),
        unreachable_fallback=_get_silmaril_unreachable_fallback(
            litellm_params, guardrail
        ),
        extra_headers=getattr(litellm_params, "extra_headers", None),
    )

    litellm.logging_callback_manager.add_litellm_callback(_instance)

    return _instance


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.SILMARIL.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.SILMARIL.value: SilmarilGuardrail,
}
