"""Sensitive Data Routing guardrail: reroutes requests with sensitive data to an on-premise model."""

from typing import TYPE_CHECKING, List

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .sensitive_data_routing import SensitiveDataRoutingGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(
    litellm_params: "LitellmParams",
    guardrail: "Guardrail",
) -> SensitiveDataRoutingGuardrail:
    import litellm

    guardrail_name = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError("sensitive_data_routing guardrail requires a guardrail_name")

    on_premise_model = getattr(litellm_params, "on_premise_model", None)
    if not on_premise_model:
        raise ValueError(
            "sensitive_data_routing guardrail requires 'on_premise_model' (the model_list "
            "name to route sensitive requests to)"
        )

    instance = SensitiveDataRoutingGuardrail(
        guardrail_name=guardrail_name,
        on_premise_model=on_premise_model,
        prebuilt_patterns=getattr(litellm_params, "prebuilt_patterns", None),
        regex_patterns=getattr(litellm_params, "regex_patterns", None),
        keywords=getattr(litellm_params, "keywords", None),
        sticky_session=bool(getattr(litellm_params, "sticky_session", True)),
        session_ttl_seconds=int(getattr(litellm_params, "session_ttl_seconds", 14400)),
        event_hook=getattr(litellm_params, "mode", None),
        default_on=bool(getattr(litellm_params, "default_on", False)),
    )
    litellm.logging_callback_manager.add_litellm_callback(instance)
    return instance


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.SENSITIVE_DATA_ROUTING.value: initialize_guardrail,
}

guardrail_class_registry = {
    SupportedGuardrailIntegrations.SENSITIVE_DATA_ROUTING.value: SensitiveDataRoutingGuardrail,
}

__all__: List[str] = [
    "SensitiveDataRoutingGuardrail",
    "initialize_guardrail",
]
