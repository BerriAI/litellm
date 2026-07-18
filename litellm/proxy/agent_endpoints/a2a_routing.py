"""
A2A Agent Routing

Handles routing for A2A agents (models with "a2a/<agent-name>" prefix).
Looks up agents in the registry and injects their API base URL.
"""

from typing import Any, Optional

import litellm
from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth


async def route_a2a_agent_request(
    data: dict,
    route_type: str,
    user_api_key_dict: Optional[UserAPIKeyAuth] = None,
) -> Optional[Any]:
    """
    Route A2A agent requests directly to litellm with injected API base.

    Returns None if not an A2A request (allows normal routing to continue).
    """
    # Import here to avoid circular imports
    from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry
    from litellm.proxy.agent_endpoints.auth.agent_permission_handler import (
        AgentRequestHandler,
    )
    from litellm.proxy.route_llm_request import (
        ROUTE_ENDPOINT_MAPPING,
        ProxyModelNotFoundError,
    )

    model_name = data.get("model", "")

    # Check if this is an A2A agent request
    if not isinstance(model_name, str) or not model_name.startswith("a2a/"):
        return None

    # Extract agent name (e.g., "a2a/my-agent" -> "my-agent")
    agent_name = model_name[4:]

    # Look up agent in registry
    agent = global_agent_registry.get_agent_by_name(agent_name)
    if agent is None:
        verbose_proxy_logger.error(f"[A2A] Agent '{agent_name}' not found in registry")
        route_name = ROUTE_ENDPOINT_MAPPING.get(route_type, route_type)
        raise ProxyModelNotFoundError(route=route_name, model_name=model_name)

    # Verify the caller is permitted to use this agent (admins bypass the check)
    is_admin = user_api_key_dict is not None and (
        user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN
        or user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value
    )
    if not is_admin:
        is_allowed = await AgentRequestHandler.is_agent_allowed(
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
        verbose_proxy_logger.error(f"[A2A] Agent '{agent_name}' has no URL configured")
        route_name = ROUTE_ENDPOINT_MAPPING.get(route_type, route_type)
        raise ProxyModelNotFoundError(route=route_name, model_name=model_name)

    # Inject API base and route to litellm
    data["api_base"] = agent.agent_card_params["url"]
    verbose_proxy_logger.debug(f"[A2A] Routing {model_name} to {data['api_base']}")

    return getattr(litellm, f"{route_type}")(**data)
