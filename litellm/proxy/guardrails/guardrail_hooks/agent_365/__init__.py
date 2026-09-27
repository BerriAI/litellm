from typing import TYPE_CHECKING, Final

from litellm._logging import verbose_proxy_logger
from litellm.types.guardrails import SupportedGuardrailIntegrations

from .agent_365 import Agent365Guardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail") -> Agent365Guardrail:
    import litellm
    from litellm.secret_managers.main import get_secret_str

    tenant_id: Final = litellm_params.tenant_id or get_secret_str("AGENT365_TENANT_ID")
    client_id: Final = litellm_params.client_id or get_secret_str("AGENT365_CLIENT_ID")
    client_secret: Final = (
        litellm_params.client_secret or litellm_params.api_key or get_secret_str("AGENT365_CLIENT_SECRET")
    )

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

    extras: Final = litellm_params.model_extra or {}
    ignored_overrides: Final = tuple(
        key
        for key, value in (
            ("api_base", litellm_params.api_base),
            ("resource_app_id", extras.get("resource_app_id")),
            ("agent_id", extras.get("agent_id")),
        )
        if value is not None
    )
    if ignored_overrides:
        verbose_proxy_logger.warning(
            "Microsoft Agent 365 (%s): ignoring %s; evaluations always go to the production Agent 365 endpoint "
            "and the agent identity is the caller's key alias",
            guardrail_name,
            ", ".join(ignored_overrides),
        )

    agent_365_guardrail: Final = Agent365Guardrail(
        guardrail_name=guardrail_name,
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret,
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
