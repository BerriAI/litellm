from typing import TYPE_CHECKING, Final

from litellm.types.guardrails import SupportedGuardrailIntegrations
from litellm.types.proxy.guardrails.guardrail_hooks.agent_365 import (
    AGENT_365_PROD_API_BASE,
    AGENT_365_PROD_RESOURCE_APP_ID,
)

from .agent_365 import Agent365Guardrail, AgentIdentityCredentials

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams
    from litellm.types.proxy.guardrails.guardrail_hooks.agent_365 import Agent365AuthMode


def _resolve_agent_identity(
    auth_mode: "Agent365AuthMode",
    agent_identity_client_id: str | None,
    agent_user_upn: str | None,
) -> AgentIdentityCredentials | None:
    if auth_mode != "agent_identity":
        return None
    if not (agent_identity_client_id and agent_user_upn):
        raise ValueError(
            "Microsoft Agent 365: auth_mode='agent_identity' requires agent_identity_client_id and agent_user_upn"
        )
    return AgentIdentityCredentials(client_id=agent_identity_client_id, agent_user_upn=agent_user_upn)


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail") -> Agent365Guardrail:
    import litellm
    from litellm.secret_managers.main import get_secret_str

    tenant_id: Final = litellm_params.tenant_id or get_secret_str("AGENT365_TENANT_ID")
    client_id: Final = litellm_params.client_id or get_secret_str("AGENT365_CLIENT_ID")
    client_secret: Final = (
        litellm_params.client_secret or litellm_params.api_key or get_secret_str("AGENT365_CLIENT_SECRET")
    )
    api_base: Final = litellm_params.api_base or get_secret_str("AGENT365_API_BASE")
    resource_app_id: Final = litellm_params.resource_app_id or get_secret_str("AGENT365_RESOURCE_APP_ID")

    if not tenant_id:
        raise ValueError("Microsoft Agent 365: tenant_id is required")
    if not client_id:
        raise ValueError("Microsoft Agent 365: client_id is required")
    if not client_secret:
        raise ValueError(
            "Microsoft Agent 365: client secret is required. Set client_secret, api_key, or AGENT365_CLIENT_SECRET"
        )

    guardrail_name: Final = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError("Microsoft Agent 365: guardrail_name is required")
    agent_identity: Final = _resolve_agent_identity(
        auth_mode=litellm_params.auth_mode or "on_behalf_of",
        agent_identity_client_id=litellm_params.agent_identity_client_id
        or get_secret_str("AGENT365_AGENT_IDENTITY_CLIENT_ID"),
        agent_user_upn=litellm_params.agent_user_upn or get_secret_str("AGENT365_AGENT_USER_UPN"),
    )

    agent_365_guardrail: Final = Agent365Guardrail(
        guardrail_name=guardrail_name,
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret,
        api_base=api_base or AGENT_365_PROD_API_BASE,
        resource_app_id=resource_app_id or AGENT_365_PROD_RESOURCE_APP_ID,
        agent_id=litellm_params.agent_id,
        agent_identity=agent_identity,
        request_timeout=litellm_params.timeout if litellm_params.timeout is not None else 10.0,
        unreachable_fallback=litellm_params.unreachable_fallback,
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(agent_365_guardrail)
    return agent_365_guardrail


guardrail_initializer_registry: Final = {  # mutable-ok: registry auto-discovery requires a dict instance
    SupportedGuardrailIntegrations.AGENT_365.value: initialize_guardrail,
}

guardrail_class_registry: Final = {  # mutable-ok: registry auto-discovery requires a dict instance
    SupportedGuardrailIntegrations.AGENT_365.value: Agent365Guardrail,
}
