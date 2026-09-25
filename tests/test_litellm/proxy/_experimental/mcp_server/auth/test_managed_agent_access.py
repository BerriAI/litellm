from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from litellm.proxy import proxy_server
from litellm.proxy._experimental.mcp_server import mcp_server_manager
from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import MCPRequestHandler
from litellm.proxy._types import LiteLLM_ObjectPermissionTable, LiteLLM_UserTable, UserAPIKeyAuth
from litellm.proxy.auth import auth_checks
from litellm.types.agents import AgentResponse
from litellm.types.proxy.agent_identity import ManagedAgentContext


def actor(tools: tuple[str, ...] | None, *, delegated: bool = False) -> UserAPIKeyAuth:
    permission: Final = LiteLLM_ObjectPermissionTable(
        object_permission_id="agent-permissions",
        mcp_servers=["slack", "linear"],
        mcp_tool_permissions={"slack": list(tools)} if tools is not None else None,
    )
    agent: Final = AgentResponse(
        agent_id="publisher",
        agent_name="Publisher",
        agent_card_params={},
        object_permission=permission.model_dump(),
        identity_managed=True,
    )
    auth: Final = UserAPIKeyAuth(agent_id=agent.agent_id)
    auth.managed_agent_policy = agent
    auth.managed_agent_context = ManagedAgentContext(
        agent_id=agent.agent_id,
        mode="delegated" if delegated else "autonomous",
        user_id="human" if delegated else None,
    )
    return auth


@pytest.fixture(autouse=True)
def isolated_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_server_manager, "global_mcp_server_manager", mcp_server_manager.MCPServerManager())
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())


@pytest.mark.asyncio
@pytest.mark.parametrize("tools", (None, (), ("read",), ("read", "write")))
async def test_autonomous_agent_uses_only_its_own_tool_grants(tools: tuple[str, ...] | None) -> None:
    auth: Final = actor(tools)
    assert set(await MCPRequestHandler.get_allowed_mcp_servers(auth)) == {"slack", "linear"}
    actual: Final = await MCPRequestHandler.get_allowed_tools_for_server("slack", auth)
    assert (frozenset(actual) if actual is not None else None) == (frozenset(tools) if tools is not None else None)
    assert await MCPRequestHandler.get_allowed_tools_for_server("ungranted-server", auth) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "agent_tools,user_tools,expected",
    (
        (None, ("read",), ("read",)),
        (("read",), None, ("read",)),
        (("read", "write"), ("read",), ("read",)),
        (("read",), ("write",), ()),
        ((), None, ()),
    ),
)
async def test_delegated_server_and_tool_intersections(
    monkeypatch: pytest.MonkeyPatch,
    agent_tools: tuple[str, ...] | None,
    user_tools: tuple[str, ...] | None,
    expected: tuple[str, ...],
) -> None:
    permission: Final = LiteLLM_ObjectPermissionTable(
        object_permission_id="user-permissions",
        mcp_servers=["slack", "user-only"],
        mcp_tool_permissions={"slack": list(user_tools)} if user_tools is not None else None,
    )
    user: Final = LiteLLM_UserTable(user_id="human", teams=[], object_permission=permission)
    monkeypatch.setattr(auth_checks, "get_user_object", AsyncMock(return_value=user))
    auth: Final = actor(agent_tools, delegated=True)
    assert await MCPRequestHandler.get_allowed_mcp_servers(auth) == ["slack"]
    assert await MCPRequestHandler.get_allowed_tools_for_server("slack", auth) == list(expected)
    assert await MCPRequestHandler.get_allowed_tools_for_server("linear", auth) == []
    assert await MCPRequestHandler.get_allowed_tools_for_server("user-only", auth) == []


@pytest.mark.asyncio
async def test_unavailable_delegated_user_never_leaves_agent_permissions_unrestricted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(auth_checks, "get_user_object", AsyncMock(side_effect=RuntimeError("DB unavailable")))
    with pytest.raises(HTTPException) as failure:
        await MCPRequestHandler.get_allowed_tools_for_server("slack", actor(None, delegated=True))
    assert failure.value.status_code == 503


@pytest.mark.asyncio
@pytest.mark.parametrize("servers,expected", (((), ()), (("slack",), ("slack",)), (("user-only",), ())))
async def test_access_groups_cap_agent_servers_without_granting_new_ones(
    monkeypatch: pytest.MonkeyPatch,
    servers: tuple[str, ...],
    expected: tuple[str, ...],
) -> None:
    from litellm.proxy._types import LiteLLM_AccessGroupTable

    group: Final = LiteLLM_AccessGroupTable(
        access_group_id="group", access_group_name="Restricted", access_mcp_server_ids=list(servers)
    )
    monkeypatch.setattr(auth_checks, "get_access_object", AsyncMock(return_value=group))
    auth: Final = actor(None)
    assert auth.managed_agent_policy is not None
    auth.managed_agent_policy = auth.managed_agent_policy.model_copy(update={"access_group_ids": ["group"]})
    assert tuple(await MCPRequestHandler.get_allowed_mcp_servers(auth)) == expected
    if "slack" not in expected:
        assert await MCPRequestHandler.get_allowed_tools_for_server("slack", auth) == []
