"""Alice WonderFence guardrail integration for LiteLLM."""

from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .alice_wonderfence import (
    WonderFenceBlockedError,
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

    # Pass only fields the user (or pydantic default) actually populated. The
    # constructor owns the defaults, so `or X` chains here would silently
    # override explicit falsy values like `api_timeout=0` or `fail_open=False`.
    init_kwargs: dict = {
        "guardrail_name": guardrail_name,
        "api_key": litellm_params.api_key,
        "api_base": litellm_params.api_base,
        "platform": litellm_params.platform,
        "max_cached_clients": litellm_params.max_cached_clients,
        "connection_pool_limit": litellm_params.connection_pool_limit,
        "event_hook": litellm_params.mode,
        "default_on": (
            litellm_params.default_on if litellm_params.default_on is not None else True
        ),
    }
    if litellm_params.api_timeout is not None:
        init_kwargs["api_timeout"] = litellm_params.api_timeout
    if litellm_params.fail_open is not None:
        init_kwargs["fail_open"] = litellm_params.fail_open
    if litellm_params.block_message is not None:
        init_kwargs["block_message"] = litellm_params.block_message
    if litellm_params.debug is not None:
        init_kwargs["debug"] = litellm_params.debug

    wonderfence_guardrail = WonderFenceGuardrail(**init_kwargs)

    litellm.logging_callback_manager.add_litellm_callback(wonderfence_guardrail)
    return wonderfence_guardrail


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.ALICE_WONDERFENCE.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.ALICE_WONDERFENCE.value: WonderFenceGuardrail,
}


__all__ = [
    "WonderFenceBlockedError",
    "WonderFenceGuardrail",
    "WonderFenceMissingSecrets",
    "initialize_guardrail",
]
