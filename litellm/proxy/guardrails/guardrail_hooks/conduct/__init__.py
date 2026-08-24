from typing import TYPE_CHECKING, Final

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .conduct import ConductGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    """Initialize the Conduct guardrail from LiteLLM's config block.

    Maps LiteLLM's idiomatic ``api_base`` / ``api_key`` to Conduct's
    ``api_url`` / ``agent_token`` constructor arguments. All other
    settings pass through unchanged.
    """
    import litellm

    _conduct_callback: Final = ConductGuardrail(
        api_url=getattr(litellm_params, "api_base", None),
        agent_token=getattr(litellm_params, "api_key", None),
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
