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


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["tools", "servers", "disabled", "outage"])
async def test_delegated_mcp_revokes_warm_human_policy_before_tool_execution(
    monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache, object_permission_cache_key

    permission: Final = LiteLLM_ObjectPermissionTable(
        object_permission_id="user-grant", mcp_servers=["slack"], mcp_tool_permissions={"slack": ["read", "write"]}
    )
    user: Final = LiteLLM_UserTable(
        user_id="human", teams=[], organization_memberships=[], object_permission_id="user-grant"
    )
    cache: Final = UserApiKeyCache()
    cache.set_cache("human", user)
    cache.set_cache(object_permission_cache_key("user-grant"), permission)
    client: Final = MagicMock()
    client.writer_db.litellm_usertable.find_unique = AsyncMock(return_value=user)
    client.writer_db.litellm_objectpermissiontable.find_unique = AsyncMock(return_value=permission)
    monkeypatch.setattr(proxy_server, "prisma_client", client)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    auth: Final = actor(("read", "write"), delegated=True)
    assert set(await MCPRequestHandler.get_allowed_tools_for_server("slack", auth)) == {"read", "write"}
    if change == "disabled":
        client.writer_db.litellm_usertable.find_unique.return_value = user.model_copy(
            update={"metadata": {"scim_active": False}}
        )
    elif change == "outage":
        client.writer_db.litellm_usertable.find_unique.side_effect = RuntimeError("writer unavailable")
    elif change == "servers":
        client.writer_db.litellm_objectpermissiontable.find_unique.return_value = permission.model_copy(
            update={"mcp_servers": [], "mcp_tool_permissions": {}}
        )
    else:
        client.writer_db.litellm_objectpermissiontable.find_unique.return_value = permission.model_copy(
            update={"mcp_tool_permissions": {"slack": ["read"]}}
        )
    if change in ("disabled", "outage"):
        with pytest.raises(HTTPException):
            await MCPRequestHandler.get_allowed_tools_for_server("slack", auth)
    else:
        assert await MCPRequestHandler.get_allowed_tools_for_server("slack", auth) == (
            ["read"] if change == "tools" else []
        )
    client.db.litellm_usertable.find_unique.assert_not_called()
    client.db.litellm_objectpermissiontable.find_unique.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["proxy_admin", "proxy_admin_viewer", "internal_user"])
@pytest.mark.parametrize("open_channel", ["none", "operator", "submitted"])
@pytest.mark.parametrize("has_grant", [True, False])
@pytest.mark.parametrize("agent_tools", [("read", "write"), None])
async def test_delegated_mcp_uses_explicit_team_grants_even_for_dashboard_admins(
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    open_channel: str,
    has_grant: bool,
    agent_tools: tuple[str, ...] | None,
) -> None:
    from litellm.proxy._types import LiteLLM_TeamTable
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    manager: Final = mcp_server_manager.global_mcp_server_manager
    manager.registry = {
        name: MCPServer(
            server_id=name,
            name=name,
            transport="http",
            url="https://example.com/mcp",
            allow_all_keys=open_channel == "operator",
        )
        for name in ("slack", "linear")
    }
    from litellm.proxy._experimental.mcp_server import db

    monkeypatch.setattr(
        db,
        "get_active_submitted_mcp_server_ids_for_user",
        AsyncMock(return_value=["slack", "linear"] if open_channel == "submitted" else []),
    )
    permission: Final = LiteLLM_ObjectPermissionTable(
        object_permission_id="team-grant", mcp_servers=["slack"], mcp_tool_permissions={"slack": ["read"]}
    )
    user: Final = LiteLLM_UserTable(
        user_id="human", user_role=role, teams=["team"] if has_grant else [], organization_memberships=[]
    )
    team: Final = LiteLLM_TeamTable(
        team_id="team",
        models=[],
        members_with_roles=[{"user_id": "human", "role": "user"}],
        object_permission_id="team-grant",
    )
    client: Final = MagicMock()
    client.writer_db.litellm_usertable.find_unique = AsyncMock(return_value=user)
    client.writer_db.litellm_teamtable.find_unique = AsyncMock(return_value=team)
    client.writer_db.litellm_objectpermissiontable.find_unique = AsyncMock(return_value=permission)
    monkeypatch.setattr(proxy_server, "prisma_client", client)
    auth: Final = actor(agent_tools, delegated=True)
    assert await MCPRequestHandler.get_allowed_mcp_servers(auth) == (["slack"] if has_grant else [])
    assert await MCPRequestHandler.get_allowed_tools_for_server("slack", auth) == (["read"] if has_grant else [])
    assert await MCPRequestHandler.get_allowed_tools_for_server("linear", auth) == []
    admitted: Final = await MCPRequestHandler.reload_admitted_user("human", requires_fresh_policy=True)
    assert admitted.user_role == role


@pytest.mark.asyncio
async def test_explicit_grants_never_fall_back_to_open_servers_on_resolution_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy._experimental.mcp_server import db
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    manager: Final = mcp_server_manager.global_mcp_server_manager
    manager.registry = {"slack": MCPServer(server_id="slack", name="slack", transport="http", allow_all_keys=True)}
    monkeypatch.setattr(db, "get_active_submitted_mcp_server_ids_for_user", AsyncMock(return_value=["slack"]))
    auth: Final = UserAPIKeyAuth(user_id="human")
    auth.mcp_explicit_grants_only = True
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(MCPRequestHandler, "get_mcp_server_access", AsyncMock(side_effect=RuntimeError("unavailable")))
        assert await manager.get_allowed_mcp_servers(auth) == []
        auth.mcp_explicit_grants_only = False
        assert await manager.get_allowed_mcp_servers(auth) == ["slack"]
