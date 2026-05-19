from typing import TYPE_CHECKING, Literal, Optional, cast

import litellm
from litellm.proxy.guardrails.guardrail_hooks.mcp_security.mcp_security_guardrail import (
    MCPSecurityGuardrail,
)
from litellm.types.guardrails import SupportedGuardrailIntegrations

if TYPE_CHECKING:
    from litellm import Router
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(
    litellm_params: "LitellmParams",
    guardrail: "Guardrail",
    llm_router: Optional["Router"] = None,
):
    guardrail_name = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError("MCP Security: guardrail_name is required")

    on_violation: Literal["block", "alert"] = cast(
        Literal["block", "alert"],
        getattr(litellm_params, "on_violation", "block"),
    )

    mcp_security_guardrail = MCPSecurityGuardrail(
        guardrail_name=guardrail_name,
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on or False,
        on_violation=on_violation,
    )

    litellm.logging_callback_manager.add_litellm_callback(mcp_security_guardrail)
    return mcp_security_guardrail


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.MCP_SECURITY.value: initialize_guardrail,
}

guardrail_class_registry = {
    SupportedGuardrailIntegrations.MCP_SECURITY.value: MCPSecurityGuardrail,
}
