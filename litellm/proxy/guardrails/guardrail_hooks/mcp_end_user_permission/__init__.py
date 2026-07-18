from typing import TYPE_CHECKING, Any, Dict, cast

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .mcp_end_user_permission import MCPEndUserPermissionGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    # Default to always-on. Only disable if the user explicitly sets default_on: false.
    # We check the raw guardrail dict because LitellmParams normalizes None → False,
    # making it impossible to distinguish "not set" from "explicitly false" via litellm_params.
    _raw_default_on = cast(Dict[str, Any], guardrail).get("litellm_params", {}).get("default_on")
    _default_on = False if _raw_default_on is False else True

    _callback = MCPEndUserPermissionGuardrail(
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=_default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(_callback)
    return _callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.MCP_END_USER_PERMISSION.value: initialize_guardrail,
}

guardrail_class_registry = {
    SupportedGuardrailIntegrations.MCP_END_USER_PERMISSION.value: MCPEndUserPermissionGuardrail,
}
