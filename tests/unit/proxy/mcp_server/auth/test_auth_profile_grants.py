import asyncio
from collections.abc import Awaitable, Callable
from typing import Final

import pytest
from fastapi import HTTPException

from litellm.proxy import proxy_server
from litellm.proxy._types import UI_TEAM_ID, LiteLLM_ObjectPermissionTable, UserAPIKeyAuth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.mcp_server.auth.auth_profile_grants import caller_auth_profile_grants
from litellm.types.agents import AgentResponse
from litellm.types.proxy.agent_identity import ManagedAgentContext
from tests.unit.proxy.mcp_prisma_fake import FakeGrantPrisma
from tests.unit.proxy.rust_auth_stores import RustAuthStores


def _permission(permission_id: str, *grants: str) -> LiteLLM_ObjectPermissionTable:
    return LiteLLM_ObjectPermissionTable(object_permission_id=permission_id, mcp_auth_profiles=list(grants))


def _caller(
    *,
    key: LiteLLM_ObjectPermissionTable | None = None,
    agent: tuple[str, ...] | None = None,
    mode: str | None = None,
    team_id: str | None = "payments",
    admitted_subject: bool = False,
) -> UserAPIKeyAuth:
    managed: Final = (
        {}
        if agent is None
        else {
            "managed_agent_policy": AgentResponse(
                agent_id="nightly-agent",
                agent_name="nightly-agent",
                agent_card_params={},
                object_permission={"object_permission_id": "perm-agent", "mcp_auth_profiles": list(agent)},
            ),
        }
    )
    context: Final = (
        {}
        if mode is None
        else {
            "managed_agent_context": ManagedAgentContext(
                agent_id="nightly-agent", mode=mode, user_id="maria" if mode == "delegated" else None
            )
        }
    )
    auth: Final = UserAPIKeyAuth(
        api_key="hashed-key",
        team_id=team_id,
        agent_id=None if agent is None else "nightly-agent",
        object_permission=key,
        object_permission_id=None if key is None else key.object_permission_id,
    )
    return auth.model_copy(update={"mcp_admitted_user_subject": admitted_subject, **managed, **context})


@pytest.fixture
def database(monkeypatch: pytest.MonkeyPatch, rust_auth_stores: RustAuthStores) -> FakeGrantPrisma:
    fake: Final = FakeGrantPrisma(
        rust_auth_stores,
        teams={"payments": "perm-team", UI_TEAM_ID: "perm-ui"},
        permissions={"perm-team": ("ana-team",), "perm-ui": ("ana-ui",), "perm-bot": ("ana-bots",)},
    )
    monkeypatch.setattr(proxy_server, "prisma_client", fake)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", UserApiKeyCache())
    return fake


KEY: Final = _permission("perm-key", "ana-key")


@pytest.mark.asyncio
@pytest.mark.usefixtures("database")
@pytest.mark.parametrize(
    ("caller", "expected"),
    [
        (_caller(), (("ana-team",),)),
        (_caller(key=KEY), (("ana-key",), ("ana-team",))),
        (_caller(key=_permission("perm-empty")), ((), ("ana-team",))),
        (_caller(key=KEY, agent=("ana-agent",), mode="autonomous"), (("ana-agent",), ("ana-key",), ("ana-team",))),
        (_caller(key=KEY, agent=("ana-agent",), mode="delegated"), ()),
        (_caller(key=KEY, agent=("ana-agent",)), (("ana-key",), ("ana-team",))),
        (_caller(key=KEY, team_id=None), (("ana-key",),)),
        (_caller(key=KEY, team_id=UI_TEAM_ID), (("ana-key",),)),
        (_caller(key=KEY, admitted_subject=True), ()),
    ],
    ids=[
        "single-team-only",
        "key-before-team",
        "empty-key-list-falls-to-team",
        "autonomous-agent-before-key",
        "delegated-agent-holds-none",
        "no-agent-context-is-not-an-agent",
        "teamless-key",
        "dashboard-team-holds-none",
        "gateway-signed-in-person",
    ],
)
async def test_grants_follow_the_most_specific_carrier_first(
    caller: UserAPIKeyAuth, expected: tuple[tuple[str, ...], ...]
) -> None:
    assert await caller_auth_profile_grants(caller) == expected


@pytest.mark.asyncio
async def test_a_key_cached_without_its_permission_row_reads_the_row_and_fails_when_the_read_fails(
    database: FakeGrantPrisma,
) -> None:
    unhydrated: Final = UserAPIKeyAuth(api_key="hashed-key", object_permission_id="perm-bot")
    assert await caller_auth_profile_grants(unhydrated) == (("ana-bots",),)

    database.fail_permission_reads()
    with pytest.raises(ConnectionError):
        await caller_auth_profile_grants(unhydrated)


@pytest.mark.asyncio
async def test_a_team_cached_without_its_permission_row_reads_the_row_and_fails_when_the_read_fails(
    database: FakeGrantPrisma,
) -> None:
    from litellm.proxy._types import LiteLLM_TeamTableCachedObj

    await proxy_server.user_api_key_cache.async_set_cache(
        key="team_id:payments",
        value=LiteLLM_TeamTableCachedObj(team_id="payments", object_permission_id="perm-team"),
    )
    caller: Final = _caller()
    assert await caller_auth_profile_grants(caller) == (("ana-team",),)

    database.fail_permission_reads()
    with pytest.raises(ConnectionError):
        await caller_auth_profile_grants(caller)


async def _toolset_scoped(caller: UserAPIKeyAuth, scope: str) -> UserAPIKeyAuth:
    from litellm.proxy.mcp_server.server import apply_toolset_scope
    from litellm.responses.mcp.litellm_proxy_mcp_handler import LiteLLM_Proxy_MCP_Handler

    async def acting_as_itself(user_api_key_auth: UserAPIKeyAuth) -> UserAPIKeyAuth:
        return user_api_key_auth

    async def holds_the_toolset(user_api_key_auth: UserAPIKeyAuth) -> frozenset[str]:
        return frozenset({"nightly-tools"})

    if scope == "gateway":
        return await apply_toolset_scope(caller, "nightly-tools", acting_as_itself, holds_the_toolset)
    return await LiteLLM_Proxy_MCP_Handler._apply_toolset_permissions(["nightly-tools"], [], caller)


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["gateway", "responses"])
async def test_a_toolset_scoped_key_cached_without_its_row_keeps_its_own_grants(
    database: FakeGrantPrisma, scope: str
) -> None:
    scoped: Final = await _toolset_scoped(UserAPIKeyAuth(api_key="hashed-key", object_permission_id="perm-bot"), scope)
    assert scoped.object_permission is not None, "the toolset scope stands in a synthetic permission row"

    assert await caller_auth_profile_grants(scoped) == (("ana-bots",),)

    database.fail_permission_reads()
    with pytest.raises(ConnectionError):
        await caller_auth_profile_grants(scoped)


@pytest.mark.asyncio
async def test_a_key_naming_a_permission_row_on_a_proxy_without_a_database_grants_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(proxy_server, "prisma_client", None)

    assert await caller_auth_profile_grants(UserAPIKeyAuth(api_key="hashed-key", object_permission_id="perm-key")) == ()


@pytest.mark.asyncio
async def test_a_toolset_scoped_key_cached_without_its_row_never_sends_the_default_credential(
    database: FakeGrantPrisma,
) -> None:
    import httpx2

    from litellm.proxy.mcp_server.mcp_server_manager import MCPServerManager
    from litellm.proxy.mcp_server.outbound_credentials.client_credentials import ClientCredentialsTokenSource
    from litellm.proxy.mcp_server.outbound_credentials.resolver import UpstreamCredentialProvider
    from litellm.types.mcp import MCPAuth
    from tests.unit.proxy.mcp_server.auth_profile_fakes import (
        AnalyticsUpstream,
        BotTokenEndpoint,
        profiled_analytics_server,
    )

    upstream: Final = AnalyticsUpstream()
    manager: Final = MCPServerManager(
        cred_provider=UpstreamCredentialProvider(
            client_credentials_source=ClientCredentialsTokenSource(BotTokenEndpoint())
        ),
        upstream_transport=httpx2.MockTransport(upstream.respond),
    )
    manager.registry["analytics"] = profiled_analytics_server(
        auth_type=MCPAuth.bearer_token, oauth2_flow=None, authentication_token="shared-default-token"
    )
    manager.tool_name_to_mcp_server_name_mapping["analytics-get_issue"] = "analytics"
    scoped: Final = await _toolset_scoped(
        UserAPIKeyAuth(api_key="hashed-key", object_permission_id="perm-bot"), "gateway"
    )

    await asyncio.create_task(
        manager.call_tool(server_name="analytics", name="get_issue", arguments={}, user_api_key_auth=scoped)
    )
    assert upstream.calls("tools/call") == ("Bearer bot-app-token-1",), upstream.requests

    database.fail_permission_reads()
    with pytest.raises(HTTPException) as failure:
        await asyncio.create_task(
            manager.call_tool(server_name="analytics", name="get_issue", arguments={}, user_api_key_auth=scoped)
        )
    assert failure.value.status_code == 503
    assert upstream.calls("tools/call") == ("Bearer bot-app-token-1",), "a failed grant read sent a request"


@pytest.mark.asyncio
async def test_an_autonomous_agent_request_reads_its_team_no_more_often_than_main(
    monkeypatch: pytest.MonkeyPatch, rust_auth_stores: RustAuthStores
) -> None:
    from litellm.proxy.auth.auth_checks import get_team_object
    from litellm.proxy.mcp_server import operations
    from litellm.proxy.mcp_server.mcp_server_manager import global_mcp_server_manager
    from litellm.types.mcp import MCPTransport
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    database: Final = FakeGrantPrisma(
        rust_auth_stores,
        teams={"payments": "perm-team"},
        permissions={"perm-team": ()},
        servers={"perm-team": ("plain",)},
    )
    monkeypatch.setattr(proxy_server, "prisma_client", database)
    monkeypatch.setitem(proxy_server.general_settings, "supported_db_objects", [])
    monkeypatch.setitem(
        global_mcp_server_manager.registry,
        "plain",
        MCPServer(server_id="plain", name="plain", server_name="plain", transport=MCPTransport.http),
    )
    caller: Final = _caller(agent=(), mode="autonomous").model_copy(update={"requires_fresh_policy": True})
    caller.managed_agent_policy.object_permission["mcp_servers"] = ["plain"]

    async def reads_of(mcp_request: Callable[[], Awaitable[object]]) -> tuple[int, int]:
        cache: Final = UserApiKeyCache()
        monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
        database.forget_reads()
        await get_team_object(team_id="payments", prisma_client=database, user_api_key_cache=cache)
        await asyncio.create_task(mcp_request())
        return database.reads()

    on_main: Final = await reads_of(lambda: global_mcp_server_manager.get_allowed_mcp_servers(caller))
    with_grants: Final = await reads_of(lambda: operations._get_allowed_mcp_servers(caller, None))

    assert on_main == (1, 1), "one team read and one permission read"
    assert with_grants == on_main


@pytest.mark.asyncio
async def test_an_autonomous_agent_whose_team_is_cached_without_its_row_reads_the_row(
    database: FakeGrantPrisma,
) -> None:
    from litellm.proxy._types import LiteLLM_TeamTableCachedObj

    await proxy_server.user_api_key_cache.async_set_cache(
        key="team_id:payments",
        value=LiteLLM_TeamTableCachedObj(team_id="payments", object_permission_id="perm-team"),
    )
    caller: Final = _caller(agent=(), mode="autonomous").model_copy(update={"requires_fresh_policy": True})

    assert await caller_auth_profile_grants(caller) == ((), ("ana-team",))
