"""Built-in Sensitive Data Routing guardrail: reroutes sensitive prompts to an on-premise model."""

from typing import TYPE_CHECKING, Final

from litellm.types.guardrails import GuardrailEventHooks, Mode, SupportedGuardrailIntegrations
from litellm.types.proxy.guardrails.guardrail_hooks.sensitive_data_routing import (
    SensitiveDataRoutingGuardrailConfigModel,
)

from .sensitive_data_routing import SensitiveDataRoutingGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def _to_event_hook(
    mode: str | list[str] | Mode | None,
) -> GuardrailEventHooks | list[GuardrailEventHooks] | Mode | None:
    if mode is None or isinstance(mode, Mode):
        return mode
    if isinstance(mode, str):
        return GuardrailEventHooks(mode)
    return [GuardrailEventHooks(single_mode) for single_mode in mode]


def initialize_guardrail(
    litellm_params: "LitellmParams",
    guardrail: "Guardrail",
) -> SensitiveDataRoutingGuardrail:
    """Initialize the Sensitive Data Routing guardrail from config."""
    import litellm

    guardrail_name: Final = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError("sensitive_data_routing guardrail requires a guardrail_name")

    config: Final = SensitiveDataRoutingGuardrailConfigModel.model_validate(litellm_params.model_dump())
    if not config.on_premise_model:
        raise ValueError("sensitive_data_routing guardrail requires 'on_premise_model'")

    instance: Final = SensitiveDataRoutingGuardrail(
        on_premise_model=config.on_premise_model,
        guardrail_name=guardrail_name,
        prebuilt_patterns=config.prebuilt_patterns,
        regex_patterns=config.regex_patterns,
        keywords=config.keywords,
        sticky_session=config.sticky_session,
        session_ttl_seconds=config.session_ttl_seconds,
        event_hook=_to_event_hook(litellm_params.mode),
        default_on=bool(litellm_params.default_on),
    )
    litellm.logging_callback_manager.add_litellm_callback(instance)
    return instance


guardrail_initializer_registry: Final = {
    SupportedGuardrailIntegrations.SENSITIVE_DATA_ROUTING.value: initialize_guardrail,
}

guardrail_class_registry: Final = {
    SupportedGuardrailIntegrations.SENSITIVE_DATA_ROUTING.value: SensitiveDataRoutingGuardrail,
}

__all__ = [
    "SensitiveDataRoutingGuardrail",
    "initialize_guardrail",
]
