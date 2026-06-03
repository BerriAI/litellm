"""Sensitive Data Routing guardrail: reroutes requests with sensitive data to an on-premise model."""

from typing import TYPE_CHECKING, Any, List

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .sensitive_data_routing import SensitiveDataRoutingGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def _get_param(
    litellm_params: "LitellmParams",
    guardrail: "Guardrail",
    key: str,
    default: Any = None,
) -> Any:
    value = getattr(litellm_params, key, None)
    if value is not None:
        return value
    raw = guardrail.get("litellm_params")
    if isinstance(raw, dict) and key in raw:
        return raw[key]
    return default


def initialize_guardrail(
    litellm_params: "LitellmParams",
    guardrail: "Guardrail",
) -> SensitiveDataRoutingGuardrail:
    import litellm

    guardrail_name = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError("sensitive_data_routing guardrail requires a guardrail_name")

    on_premise_model = _get_param(litellm_params, guardrail, "on_premise_model")
    if not on_premise_model:
        raise ValueError(
            "sensitive_data_routing guardrail requires 'on_premise_model' (the model_list "
            "name to route sensitive requests to)"
        )

    instance = SensitiveDataRoutingGuardrail(
        guardrail_name=guardrail_name,
        on_premise_model=on_premise_model,
        prebuilt_patterns=_get_param(litellm_params, guardrail, "prebuilt_patterns"),
        regex_patterns=_get_param(litellm_params, guardrail, "regex_patterns"),
        keywords=_get_param(litellm_params, guardrail, "keywords"),
        sticky_session=bool(
            _get_param(litellm_params, guardrail, "sticky_session", True)
        ),
        session_ttl_seconds=int(
            _get_param(litellm_params, guardrail, "session_ttl_seconds", 14400)
        ),
        event_hook=_get_param(litellm_params, guardrail, "mode"),
        default_on=bool(_get_param(litellm_params, guardrail, "default_on", False)),
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
