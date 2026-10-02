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


def _server_row(server_id: str, access_groups: tuple[str, ...]) -> MagicMock:
    row: Final = MagicMock()
    row.server_id = server_id
    row.mcp_access_groups = list(access_groups)
    return row


def _toolset_row(server_id: str, tool_name: str) -> MagicMock:
    row: Final = MagicMock()
    row.tools = [{"server_id": server_id, "tool_name": tool_name}]
    return row


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["tool", "server", "outage"])
async def test_autonomous_agent_toolset_and_access_group_revocations_bind_on_the_next_request(
    monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """The agent's entitlements are read through the shared toolset and access-group resolvers. Once the
    writer revokes a tool or drops the server from the group, the next managed request must be denied
    even though the legacy cache still holds the warm grant and the replica still shows the old rows"""
    from litellm.caching.caching import DualCache
    from litellm.proxy._experimental.mcp_server import toolset_db

    warm_toolset: Final = _toolset_row("slack", "read")
    list_toolsets: Final = AsyncMock(return_value=[warm_toolset])
    monkeypatch.setattr(toolset_db, "list_mcp_toolsets", list_toolsets)
    client: Final = MagicMock()
    client.db.litellm_mcpservertable.find_many = AsyncMock(return_value=[_server_row("linear", ("grp",))])
    client.writer_db.litellm_mcpservertable.find_many = AsyncMock(return_value=[_server_row("linear", ("grp",))])
    monkeypatch.setattr(proxy_server, "prisma_client", client)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", DualCache())
    permission: Final = LiteLLM_ObjectPermissionTable(
        object_permission_id="agent-permissions", mcp_toolsets=["ts"], mcp_access_groups=["grp"]
    )
    auth: Final = actor(None)
    assert auth.managed_agent_policy is not None
    auth.managed_agent_policy = auth.managed_agent_policy.model_copy(
        update={"object_permission": permission.model_dump()}
    )
    auth.requires_fresh_policy = True

    assert set(await MCPRequestHandler.get_allowed_mcp_servers(auth)) == {"slack", "linear"}
    assert await MCPRequestHandler.get_allowed_tools_for_server("slack", auth) == ["read"]

    if change == "tool":
        list_toolsets.return_value = [_toolset_row("slack", "other")]
        assert await MCPRequestHandler.get_allowed_tools_for_server("slack", auth) == ["other"]
    elif change == "server":
        client.writer_db.litellm_mcpservertable.find_many.return_value = []
        assert set(await MCPRequestHandler.get_allowed_mcp_servers(auth)) == {"slack"}
        assert await MCPRequestHandler.get_allowed_tools_for_server("linear", auth) == []
    else:
        list_toolsets.side_effect = RuntimeError("writer unavailable")
        with pytest.raises(HTTPException) as failure:
            await MCPRequestHandler.get_allowed_tools_for_server("slack", auth)
        assert failure.value.status_code == 503
    for call in list_toolsets.await_args_list:
        assert call.kwargs["use_writer"] is True, "managed agent toolsets must be read from the writer"
    client.db.litellm_mcpservertable.find_many.assert_not_called()


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
    auth.team_id = "team"
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


@pytest.mark.asyncio
async def test_absent_agent_policy_and_missing_delegated_subject_grant_no_servers() -> None:
    from litellm.proxy._experimental.mcp_server.auth.managed_agent_access import managed_agent_servers

    assert await managed_agent_servers(UserAPIKeyAuth()) == ()
    auth: Final = actor(None, delegated=True)
    assert auth.managed_agent_context is not None
    auth.managed_agent_context = auth.managed_agent_context.model_copy(update={"user_id": None})
    assert await MCPRequestHandler.get_allowed_mcp_servers(auth) == []
    assert await MCPRequestHandler.get_allowed_tools_for_server("slack", auth) == []


@pytest.mark.asyncio
async def test_tool_policy_outage_after_server_admission_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    permission: Final = LiteLLM_ObjectPermissionTable(object_permission_id="human-grant", mcp_servers=["slack"])
    user: Final = LiteLLM_UserTable(user_id="human", teams=[], object_permission=permission)
    monkeypatch.setattr(
        auth_checks, "get_user_object", AsyncMock(side_effect=[user, RuntimeError("tool lookup unavailable")])
    )
    with pytest.raises(HTTPException) as failure:
        await MCPRequestHandler.get_allowed_tools_for_server("slack", actor(None, delegated=True))
    assert failure.value.status_code == 503


@pytest.mark.asyncio
@pytest.mark.parametrize("role", (None, "proxy_admin", "internal_user"))
@pytest.mark.parametrize("scoped", (False, True))
async def test_manager_preserves_managed_server_grants_across_open_channels(
    monkeypatch: pytest.MonkeyPatch, role: str | None, scoped: bool
) -> None:
    from litellm.proxy._experimental.mcp_server import db
    from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import MCPServerAccess
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    manager: Final = mcp_server_manager.global_mcp_server_manager
    manager.registry = {
        "open": MCPServer(server_id="open", name="open", transport="http", allow_all_keys=True),
        "submitted": MCPServer(server_id="submitted", name="submitted", transport="http"),
        "passthrough": MCPServer(
            server_id="passthrough", name="passthrough", transport="http", auth_type="true_passthrough"
        ),
    }
    monkeypatch.setattr(db, "get_active_submitted_mcp_server_ids_for_user", AsyncMock(return_value=["submitted"]))
    auth: Final = actor(None)
    auth.user_role = role
    assert not auth.mcp_explicit_grants_only
    access: Final = MCPServerAccess(server_ids=("slack", "open")) if scoped else None
    assert set(await manager.get_allowed_mcp_servers(auth, access=access)) == (
        {"slack"} if scoped else {"slack", "linear"}
    )


@pytest.mark.asyncio
async def test_manager_does_not_replace_managed_policy_failure_with_open_servers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    manager: Final = mcp_server_manager.global_mcp_server_manager
    manager.registry = {"open": MCPServer(server_id="open", name="open", transport="http", allow_all_keys=True)}
    monkeypatch.setattr(auth_checks, "get_user_object", AsyncMock(side_effect=RuntimeError("writer unavailable")))
    with pytest.raises(HTTPException) as failure:
        await manager.get_allowed_mcp_servers(actor(None, delegated=True))
    assert failure.value.status_code == 503


@pytest.mark.asyncio
async def test_inline_tool_grant_admits_its_server_without_widening_tools() -> None:
    auth: Final = actor(("read",))
    assert auth.managed_agent_policy is not None
    auth.managed_agent_policy = auth.managed_agent_policy.model_copy(
        update={"object_permission": {"object_permission_id": "tools", "mcp_tool_permissions": {"slack": ["read"]}}}
    )
    assert await MCPRequestHandler.get_allowed_mcp_servers(auth) == ["slack"]
    assert await MCPRequestHandler.get_allowed_tools_for_server("slack", auth) == ["read"]
    assert await MCPRequestHandler.get_allowed_tools_for_server("linear", auth) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("selected_team", (None, "selected"))
@pytest.mark.parametrize("selected_grant", (False, True))
async def test_delegation_never_borrows_another_teams_server_or_tools(
    monkeypatch: pytest.MonkeyPatch, selected_team: str | None, selected_grant: bool
) -> None:
    from litellm.proxy._types import LiteLLM_TeamTable
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

    user: Final = LiteLLM_UserTable(user_id="human", teams=["selected", "other"], organization_memberships=[])
    permission: Final = LiteLLM_ObjectPermissionTable(
        object_permission_id="selected-grant",
        mcp_servers=["slack"] if selected_grant else [],
        mcp_tool_permissions={"slack": ["read"]} if selected_grant else {},
    )
    teams: Final = {
        name: LiteLLM_TeamTable(
            team_id=name,
            models=[],
            members_with_roles=[{"user_id": "human", "role": "user"}],
            object_permission=permission if name == "selected" else LiteLLM_ObjectPermissionTable(
                object_permission_id="other-grant", mcp_servers=["slack", "linear"]
            ),
        )
        for name in ("selected", "other")
    }

    async def get_team(team_id: str, **kwargs: object) -> LiteLLM_TeamTable:
        return teams[team_id]

    monkeypatch.setattr(auth_checks, "get_user_object", AsyncMock(return_value=user))
    monkeypatch.setattr(auth_checks, "get_team_object", get_team)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", UserApiKeyCache())
    auth: Final = actor(None, delegated=True)
    auth.team_id = selected_team
    expected: Final = ["slack"] if selected_team and selected_grant else []
    assert await MCPRequestHandler.get_allowed_mcp_servers(auth) == expected
    assert await MCPRequestHandler.get_allowed_tools_for_server("slack", auth) == (["read"] if expected else [])
    assert await MCPRequestHandler.get_allowed_tools_for_server("linear", auth) == []
    ordinary: Final = await MCPRequestHandler.reload_admitted_user("human", requires_fresh_policy=True)
    assert set(await MCPRequestHandler.resolve_admitted_subject_servers(ordinary)) == {"slack", "linear"}


@pytest.mark.asyncio
@pytest.mark.parametrize("entitlement", ("group", "toolset"))
async def test_managed_mcp_rejects_unavailable_authoritative_entitlements(
    monkeypatch: pytest.MonkeyPatch, entitlement: str
) -> None:
    client: Final = MagicMock()
    client.writer_db.litellm_mcpservertable.find_many = AsyncMock(side_effect=RuntimeError("writer unavailable"))
    client.writer_db.litellm_mcptoolsettable.find_many = AsyncMock(side_effect=RuntimeError("writer unavailable"))
    monkeypatch.setattr(proxy_server, "prisma_client", client)
    permission: Final = LiteLLM_ObjectPermissionTable(
        object_permission_id="entitlements",
        mcp_access_groups=["group"] if entitlement == "group" else [],
        mcp_toolsets=["toolset"] if entitlement == "toolset" else [],
    )
    auth: Final = actor(None)
    assert auth.managed_agent_policy is not None
    auth.managed_agent_policy = auth.managed_agent_policy.model_copy(update={"object_permission": permission.model_dump()})
    auth.requires_fresh_policy = True
    with pytest.raises(HTTPException) as failure:
        await MCPRequestHandler.get_allowed_tools_for_server("slack", auth)
    assert failure.value.status_code == 503
    client.db.litellm_mcpservertable.find_many.assert_not_called()
    client.db.litellm_mcptoolsettable.find_many.assert_not_called()


@pytest.mark.asyncio
async def test_managed_agent_mcp_access_is_capped_at_the_invoking_callers_grants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The managed MCP path must honour the agent_caller ceiling the same way the unmanaged path does:
    the agent's own policy grants slack and linear, but the team echoed back on the request reaches
    only slack, so the agent may use slack alone."""
    from litellm.proxy._types import AgentCaller

    monkeypatch.setattr(
        MCPRequestHandler,
        "_get_allowed_mcp_servers_for_team",
        AsyncMock(return_value=["slack"]),
    )
    monkeypatch.setattr(
        MCPRequestHandler,
        "_apply_user_server_ceiling",
        AsyncMock(side_effect=lambda servers, _auth: (tuple(servers), False)),
    )

    monkeypatch.setattr(
        MCPRequestHandler,
        "_get_team_object_permission",
        AsyncMock(
            return_value=LiteLLM_ObjectPermissionTable(
                object_permission_id="caller-team-permissions",
                mcp_servers=["slack"],
                mcp_tool_permissions={"slack": ["read"]},
            )
        ),
    )
    monkeypatch.setattr(
        MCPRequestHandler,
        "_apply_user_tool_ceiling",
        AsyncMock(side_effect=lambda tools, _server_id, _auth: tools),
    )

    auth: Final = actor(("read", "write"))
    auth.agent_caller = AgentCaller(user_id="alice", team_id="callers")

    assert set(await MCPRequestHandler.get_allowed_mcp_servers(auth)) == {"slack"}
    assert await MCPRequestHandler.get_allowed_tools_for_server("slack", auth) == ["read"]


@pytest.mark.asyncio
@pytest.mark.parametrize("fresh", [False, True])
@pytest.mark.parametrize("caller_kind", ["team", "user"])
async def test_caller_mcp_revocation_uses_fresh_policy(
    monkeypatch: pytest.MonkeyPatch, fresh: bool, caller_kind: str,
) -> None:
    from litellm.proxy._types import LiteLLM_TeamTable
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache, object_permission_cache_key
    from litellm.types.agents import AgentCaller

    cached_permission: Final = LiteLLM_ObjectPermissionTable(
        object_permission_id="caller-permission", mcp_servers=["slack", "linear"],
        mcp_tool_permissions={"slack": ["read", "write"]},
    )
    current_permission: Final = LiteLLM_ObjectPermissionTable(
        object_permission_id="caller-permission", mcp_servers=["slack"],
        mcp_tool_permissions={"slack": ["read"]},
    )
    team: Final = LiteLLM_TeamTable(
        team_id="caller", object_permission_id="caller-permission", object_permission=current_permission,
    )
    user: Final = LiteLLM_UserTable(
        user_id="caller", teams=[], object_permission_id="caller-permission", object_permission=current_permission,
    )
    database: Final = MagicMock()
    database.writer_db.litellm_teamtable.find_unique = AsyncMock(return_value=team)
    database.writer_db.litellm_usertable.find_unique = AsyncMock(return_value=user)
    database.writer_db.litellm_objectpermissiontable.find_unique = AsyncMock(return_value=current_permission)
    cache: Final = UserApiKeyCache()
    cache.set_cache("team_id:caller", team.model_copy(update={"object_permission": cached_permission}))
    cache.set_cache("caller", user.model_copy(update={"object_permission": cached_permission}))
    cache.set_cache(object_permission_cache_key("caller-permission"), cached_permission)
    monkeypatch.setattr(proxy_server, "prisma_client", database)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    auth: Final = actor(("read", "write"))
    auth.requires_fresh_policy = fresh
    auth.agent_caller = AgentCaller(team_id="caller") if caller_kind == "team" else AgentCaller(user_id="caller")

    assert set(await MCPRequestHandler.get_allowed_mcp_servers(auth)) == ({"slack"} if fresh else {"slack", "linear"})
    assert await MCPRequestHandler.get_allowed_tools_for_server("slack", auth) == (["read"] if fresh else ["read", "write"])


@pytest.mark.asyncio
@pytest.mark.parametrize("fresh", [False, True])
async def test_caller_team_outage_cannot_remove_authoritative_server_ceiling(
    monkeypatch: pytest.MonkeyPatch, fresh: bool,
) -> None:
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from litellm.types.agents import AgentCaller

    database: Final = MagicMock()
    database.writer_db.litellm_teamtable.find_unique = AsyncMock(side_effect=RuntimeError("writer unavailable"))
    database.db.litellm_teamtable.find_unique = AsyncMock(side_effect=RuntimeError("reader unavailable"))
    monkeypatch.setattr(proxy_server, "prisma_client", database)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", UserApiKeyCache())
    auth: Final = actor(("read",))
    auth.agent_caller = AgentCaller(team_id="caller")
    auth.requires_fresh_policy = fresh

    if fresh:
        with pytest.raises(HTTPException) as failure:
            await MCPRequestHandler.get_allowed_mcp_servers(auth)
        assert failure.value.status_code == 503
    else:
        assert set(await MCPRequestHandler.get_allowed_mcp_servers(auth)) == {"slack", "linear"}
