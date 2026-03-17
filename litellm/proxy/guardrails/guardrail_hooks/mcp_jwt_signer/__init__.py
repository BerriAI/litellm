"""MCP JWT Signer guardrail — built-in LiteLLM guardrail for zero trust MCP auth."""

from typing import TYPE_CHECKING, Any

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .mcp_jwt_signer import MCPJWTSigner, get_mcp_jwt_signer

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def _get_param(litellm_params: "LitellmParams", key: str, default: Any = None) -> Any:
    """
    Extract a config param from litellm_params, checking optional_params first
    (where YAML extras land) then the top-level object.
    """
    optional_params = getattr(litellm_params, "optional_params", None)
    if optional_params is not None:
        v = getattr(optional_params, key, None)
        if v is not None:
            return v
    v = getattr(litellm_params, key, None)
    return v if v is not None else default


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

    signer = MCPJWTSigner(
        guardrail_name=guardrail_name,
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
        # Core claims
        issuer=_get_param(litellm_params, "issuer"),
        audience=_get_param(litellm_params, "audience"),
        ttl_seconds=_get_param(litellm_params, "ttl_seconds"),
        # FR-5: inbound token verification
        access_token_discovery_uri=_get_param(
            litellm_params, "access_token_discovery_uri"
        ),
        access_token_introspection_endpoint=_get_param(
            litellm_params, "access_token_introspection_endpoint"
        ),
        # FR-12: end-user identity mapping
        end_user_claim_sources=_get_param(litellm_params, "end_user_claim_sources"),
        # FR-13: claim operations
        add_claims=_get_param(litellm_params, "add_claims"),
        set_claims=_get_param(litellm_params, "set_claims"),
        remove_claims=_get_param(litellm_params, "remove_claims"),
        # FR-14: two-token model
        channel_token_header=_get_param(litellm_params, "channel_token_header"),
        channel_token_discovery_uri=_get_param(
            litellm_params, "channel_token_discovery_uri"
        ),
        channel_token_jwks_uri=_get_param(litellm_params, "channel_token_jwks_uri"),
        # FR-15: claim validation
        required_claims=_get_param(litellm_params, "required_claims"),
        optional_claims=_get_param(litellm_params, "optional_claims"),
        # FR-9: debug headers
        debug_header=_get_param(litellm_params, "debug_header", default=True),
        # FR-10: configurable scope
        allowed_tools=_get_param(litellm_params, "allowed_tools"),
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
