"""MCP JWT Signer guardrail — built-in LiteLLM guardrail for zero trust MCP auth."""

from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .mcp_jwt_signer import MCPJWTSigner, get_mcp_jwt_signer

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(
    litellm_params: "LitellmParams", guardrail: "Guardrail"
) -> MCPJWTSigner:
    import litellm

    guardrail_name = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError("MCPJWTSigner guardrail requires a guardrail_name")

    mode = litellm_params.mode
    if mode != "pre_mcp_call":
        raise ValueError(
            f"MCPJWTSigner guardrail '{guardrail_name}' has mode='{mode}' but must use "
            "mode='pre_mcp_call'. JWT injection only fires for MCP tool calls."
        )

    optional_params = getattr(litellm_params, "optional_params", None)

    def _get(key):  # type: ignore[no-untyped-def]
        if optional_params is not None:
            v = getattr(optional_params, key, None)
            if v is not None:
                return v
        return getattr(litellm_params, key, None)

    signer = MCPJWTSigner(
        guardrail_name=guardrail_name,
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
        issuer=_get("issuer"),
        audience=_get("audience"),
        ttl_seconds=_get("ttl_seconds"),
    )
    litellm.logging_callback_manager.add_litellm_callback(signer)
    return signer


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.MCP_JWT_SIGNER.value: initialize_guardrail,
}

guardrail_class_registry = {
    SupportedGuardrailIntegrations.MCP_JWT_SIGNER.value: MCPJWTSigner,
}

__all__ = [
    "MCPJWTSigner",
    "initialize_guardrail",
    "get_mcp_jwt_signer",
]
