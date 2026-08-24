from typing import TYPE_CHECKING, Final

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .conduct import ConductGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    _conduct_callback: Final = ConductGuardrail(
        api_base=getattr(litellm_params, "api_base", None),
        api_key=getattr(litellm_params, "api_key", None),
        workspace_id=getattr(litellm_params, "workspace_id", None),
        fail_mode=getattr(litellm_params, "fail_mode", "fail_closed"),
        tool_name=getattr(litellm_params, "tool_name", "llm_call"),
        timeout=getattr(litellm_params, "timeout", 8.0),
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(_conduct_callback)

    return _conduct_callback


guardrail_initializer_registry: Final = {
    SupportedGuardrailIntegrations.CONDUCT.value: initialize_guardrail,
}


guardrail_class_registry: Final = {
    SupportedGuardrailIntegrations.CONDUCT.value: ConductGuardrail,
}
