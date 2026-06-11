from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .hlido import HlidoGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    optional_params = getattr(litellm_params, "optional_params", None)

    def _config_value(attribute_name: str):
        if optional_params is not None:
            value = getattr(optional_params, attribute_name, None)
            if value is not None:
                return value
        return getattr(litellm_params, attribute_name, None)

    allowed_tiers = _config_value("allowed_tiers")
    slugs = _config_value("slugs")

    _hlido_callback = HlidoGuardrail(
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
        min_score=_config_value("min_score"),
        allowed_tiers=tuple(allowed_tiers) if allowed_tiers else None,
        slugs=tuple(slugs) if slugs else None,
        on_unverified=_config_value("on_unverified"),
        on_error=_config_value("on_error"),
        cache_ttl=_config_value("cache_ttl"),
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(_hlido_callback)
    return _hlido_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.HLIDO.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.HLIDO.value: HlidoGuardrail,
}

__all__ = ["HlidoGuardrail"]
