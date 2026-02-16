"""Alice WonderFence guardrail integration for LiteLLM."""

from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .alice_wonderfence import (
    WonderFenceGuardrail,
    WonderFenceMissingSecrets,
)

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(
    litellm_params: "LitellmParams", guardrail: "Guardrail"
) -> WonderFenceGuardrail:
    import litellm

    guardrail_name = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError("Alice WonderFence guardrail requires a guardrail_name")

    wonderfence_guardrail = WonderFenceGuardrail(
        guardrail_name=guardrail_name,
        api_key=litellm_params.api_key or "",
        api_base=getattr(litellm_params, "api_base", None),
        app_name=getattr(litellm_params, "app_name", None),
        api_timeout=getattr(litellm_params, "api_timeout", None) or 20.0,
        platform=getattr(litellm_params, "platform", None),
        event_hook=litellm_params.mode,  # type: ignore[arg-type]
        default_on=litellm_params.default_on if litellm_params.default_on is not None else True,
    )

    litellm.logging_callback_manager.add_litellm_callback(wonderfence_guardrail)
    return wonderfence_guardrail


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.ALICE_WONDERFENCE.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.ALICE_WONDERFENCE.value: WonderFenceGuardrail,
}


__all__ = [
    "WonderFenceGuardrail",
    "WonderFenceMissingSecrets",
    "initialize_guardrail",
]
