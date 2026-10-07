"""
A2A Agent Routing

Handles routing for A2A agents (models with "a2a/<agent-name>" prefix).
Looks up agents in the registry and injects their API base URL.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

from fastapi import HTTPException

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.agent_endpoints.auth.managed_authorization import agent_invocation_policy
from litellm.types.agents import AgentResponse

if TYPE_CHECKING:
    from litellm.router import Router


async def route_a2a_agent_request(
    data: Mapping[str, object],
    route_type: str,
    user_api_key_dict: UserAPIKeyAuth | None = None,
    *,
    registered_agent: AgentResponse | None = None,
    llm_router: "Router | None" = None,
) -> Any | None:
    """
    Route A2A agent requests directly to litellm with injected API base.

    Returns None if not an A2A request (allows normal routing to continue).
    """
    # Import here to avoid circular imports
    from litellm.proxy.agent_endpoints.auth.agent_permission_handler import (
        AgentRequestHandler,
    )
    from litellm.proxy.common_utils.registry_read_through import (
        get_agent_with_read_through,
    )
    from litellm.proxy.route_llm_request import (
        ROUTE_ENDPOINT_MAPPING,
        ProxyModelNotFoundError,
    )

    model_name: Final = data.get("model", "")

    # Check if this is an A2A agent request
    if not isinstance(model_name, str) or not model_name.startswith("a2a/"):
        return None

    # Extract agent name (e.g., "a2a/my-agent" -> "my-agent")
    agent_name: Final = model_name[4:]

    # Look up agent in registry
    registered: Final = registered_agent or await get_agent_with_read_through(agent_name)
    if registered is None:
        verbose_proxy_logger.error("[A2A] Agent '%s' not found in registry", agent_name)
        route_name = ROUTE_ENDPOINT_MAPPING.get(route_type, route_type)
        raise ProxyModelNotFoundError(route=route_name, model_name=model_name, retryable_with_model_read_through=False)

    agent: Final = agent_invocation_policy(user_api_key_dict, registered)

    # Verify the caller is permitted to use this agent (admins bypass the check)
    is_admin: Final = user_api_key_dict is not None and (
        user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN
        or user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value
    )
    if not is_admin or agent.identity_managed:
        is_allowed: Final = await AgentRequestHandler.is_agent_allowed(
            agent_id=agent.agent_id,
            user_api_key_auth=user_api_key_dict,
        )
        if not is_allowed:
            raise HTTPException(
                status_code=403,
                detail=f"Agent '{agent_name}' is not allowed for your key/team. Contact proxy admin for access.",
            )

    # Get API base URL from agent config
    if not agent.agent_card_params or "url" not in agent.agent_card_params:
        verbose_proxy_logger.error("[A2A] Agent '%s' has no URL configured", agent_name)
        route_name = ROUTE_ENDPOINT_MAPPING.get(route_type, route_type)
        raise ProxyModelNotFoundError(route=route_name, model_name=model_name, retryable_with_model_read_through=False)

    # Inject API base and route to litellm
    api_base: Final = agent.agent_card_params["url"]
    verbose_proxy_logger.debug("[A2A] Routing %s to %s", model_name, api_base)

    invocation_fee: Final = user_api_key_dict.agent_invocation_cost if user_api_key_dict is not None else None
    provider_data: Final = MappingProxyType({key: value for key, value in data.items() if key != "litellm_params"})
    deployments: Final = (
        llm_router.get_model_list(model_name=model_name)
        if llm_router is not None and model_name in llm_router.model_names
        else None
    )
    if deployments and all(
        deployment["litellm_params"].get("model") == model_name
        and deployment["litellm_params"].get("custom_llm_provider") in (None, "a2a")
        for deployment in deployments
    ):
        return getattr(llm_router, route_type)(
            **MappingProxyType(
                {**provider_data, "api_base": api_base, "cost_per_query": invocation_fee, "disable_fallbacks": True}
            )
        )
    return getattr(litellm, f"{route_type}")(
        **MappingProxyType({**provider_data, "api_base": api_base, "cost_per_query": invocation_fee})
    )
