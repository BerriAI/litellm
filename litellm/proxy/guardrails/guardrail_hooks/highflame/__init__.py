from typing import TYPE_CHECKING

from litellm._logging import verbose_proxy_logger
from litellm.types.guardrails import SupportedGuardrailIntegrations

from .highflame import HighflameGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    # `javelin` is a deprecated alias of `highflame` — Javelin was renamed to
    # Highflame. Existing `guardrail: javelin` configs keep working (routed to the
    # Highflame guardrail) but should migrate.
    if str(getattr(litellm_params, "guardrail", "") or "").lower() == "javelin":
        verbose_proxy_logger.warning(
            "The 'javelin' guardrail is deprecated and now routes to 'highflame'. "
            "Update your config to `guardrail: highflame` and set `api_base` to "
            "https://api.highflame.ai. See https://docs.highflame.ai"
        )

    _highflame_callback = HighflameGuardrail(
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on or False,
        capabilities=getattr(litellm_params, "capabilities", None),
        application=litellm_params.application,
        shield_mode=getattr(litellm_params, "shield_mode", "enforce") or "enforce",
        token_url=getattr(litellm_params, "token_url", None),
        metadata=litellm_params.metadata,
    )
    litellm.logging_callback_manager.add_litellm_callback(_highflame_callback)

    return _highflame_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.HIGHFLAME.value: initialize_guardrail,
    # Deprecated alias — keeps existing `guardrail: javelin` deployments working.
    SupportedGuardrailIntegrations.JAVELIN.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.HIGHFLAME.value: HighflameGuardrail,
    # Deprecated alias — keeps existing `guardrail: javelin` deployments working.
    SupportedGuardrailIntegrations.JAVELIN.value: HighflameGuardrail,
}
