from typing import Final

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.agent_endpoints.auth.agent_access_groups import resolve_agent_access_group_ceiling
from litellm.proxy.agent_endpoints.managed_identity import raise_identity_failure
from litellm.types.proxy.agent_identity import AgentIdentityFailure


async def managed_agent_servers(auth: UserAPIKeyAuth) -> tuple[str, ...]:
    from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import MCPRequestHandler
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import global_mcp_server_manager

    agent: Final = auth.managed_agent_policy
    if agent is None:
        return ()

    async def group_ids(_agent_id: str) -> tuple[str, ...]:
        return tuple(agent.access_group_ids or ())

    try:
        base: Final = frozenset(await MCPRequestHandler._get_allowed_mcp_servers_for_agent(auth))
        ceiling: Final = await resolve_agent_access_group_ceiling(agent.agent_id, load_access_group_ids=group_ids)
        own: Final = (
            base
            if ceiling is None
            else base.intersection(global_mcp_server_manager.expand_permission_list(sorted(ceiling.mcp_server_ids)))
        )
        context: Final = auth.managed_agent_context
        if context is None or context.mode == "autonomous":
            return tuple(sorted(own))
        if context.user_id is None:
            return ()
        human: Final = await MCPRequestHandler.reload_admitted_user(context.user_id)
        allowed: Final = await MCPRequestHandler._resolve_admitted_subject_servers(human)
        return tuple(sorted(own.intersection(allowed)))
    except Exception:  # noqa: BLE001  # Authorization boundary: every unresolved policy must deny access
        raise_identity_failure(
            AgentIdentityFailure(code="policy_unavailable", message="Agent MCP policy is unavailable")
        )


async def managed_agent_tools(server_id: str, auth: UserAPIKeyAuth) -> list[str] | None:
    from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import MCPRequestHandler

    if server_id not in await managed_agent_servers(auth):
        return []
    try:
        own: Final = await MCPRequestHandler._get_agent_tool_permissions_for_server(server_id, auth)
        context: Final = auth.managed_agent_context
        if context is None or context.mode == "autonomous":
            return own
        if context.user_id is None:
            return []
        human: Final = await MCPRequestHandler.reload_admitted_user(context.user_id)
        human_tools: Final = await MCPRequestHandler._resolve_admitted_subject_tools(server_id, human)
        if own is None:
            return human_tools
        return own if human_tools is None else sorted(frozenset(own).intersection(human_tools))
    except Exception:  # noqa: BLE001  # Authorization boundary: every unresolved policy must deny access
        raise_identity_failure(
            AgentIdentityFailure(code="policy_unavailable", message="Agent tool policy is unavailable")
        )
