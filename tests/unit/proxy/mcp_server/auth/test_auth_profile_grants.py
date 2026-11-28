import math
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Final, Literal
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.models.object_permission import StoredProfileGrantError
from litellm.proxy import proxy_server
from litellm.proxy._types import (
    UI_TEAM_ID,
    LiteLLM_JWTAuth,
    LiteLLM_ObjectPermissionTable,
    LiteLLM_TeamTableCachedObj,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import get_key_object
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.db.db_lookup_gate import db_lookup_stall_tracker
from litellm.proxy.mcp_server.auth.auth_profile_grants import caller_auth_profile_grants
from litellm.proxy.mcp_server.auth.user_api_key_auth_mcp import MCPRequestHandler
from litellm.redis.stores.auth_cache import AuthObject
from litellm.types.agents import AgentResponse
from litellm.types.mcp_server.auth_profile_grants import UnreadableGrants
from litellm.types.proxy.agent_identity import ManagedAgentContext
from tests.unit.proxy.rust_auth_stores import (
    FakeKeyStore,
    FakeTeamStore,
    RustAuthStores,
    RustStoreRows,
    install_rust_auth_stores,
)


def _permission(permission_id: str, grants: Mapping[str, str] | None = None) -> LiteLLM_ObjectPermissionTable:
    return LiteLLM_ObjectPermissionTable(object_permission_id=permission_id, mcp_auth_profiles=grants)


def _caller(
    *,
    key: LiteLLM_ObjectPermissionTable | None = None,
    mode: Literal["autonomous", "delegated"] | None = None,
    team_id: str | None = "payments",
    admitted_subject: bool = False,
) -> UserAPIKeyAuth:
    managed: Final = (
        {}
        if mode is None
        else {
            "managed_agent_policy": AgentResponse(
                agent_id="agents-runner",
                agent_name="agents-runner",
                agent_card_params={},
                object_permission={"object_permission_id": "perm-agent"},
            ),
            "requires_fresh_policy": True,
        }
    )
    context: Final = (
        {}
        if mode is None
        else {
            "managed_agent_context": ManagedAgentContext(
                agent_id="agents-runner", mode=mode, user_id="reader" if mode == "delegated" else None
            )
        }
    )
    auth: Final = UserAPIKeyAuth(
        api_key="hashed-key",
        team_id=team_id,
        agent_id=None if mode is None else "agents-runner",
        object_permission=key,
        object_permission_id=None if key is None else key.object_permission_id,
    )
    return auth.model_copy(update={"mcp_admitted_user_subject": admitted_subject, **managed, **context})


@pytest.fixture
def rust_permission_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    permissions: Final = {
        "perm-team": _permission("perm-team", TEAM),
        "perm-bots": _permission("perm-bots", {"tickets": "bots"}),
        "perm-key": KEY,
        "perm-agent": _permission("perm-agent", {"tickets": "agents"}),
    }
    install_rust_auth_stores(
        monkeypatch,
        RustAuthStores(
            team=FakeTeamStore(rows={"payments": {"team_id": "payments", "object_permission_id": "perm-team"}}),
            object_permission=RustStoreRows(
                reader={
                    **permissions,
                    "agent-current": _permission("agent-current", {"tickets": "retired"}),
                    "agent-deleted": _permission("agent-deleted", {"tickets": "retired"}),
                    "timeout": _permission("timeout", {"tickets": "retired"}),
                    "disconnected": _permission("disconnected", {"tickets": "retired"}),
                },
                writer={
                    **permissions,
                    "agent-current": _permission("agent-current", {"tickets": "bots"}),
                    "timeout": TimeoutError(),
                    "disconnected": RuntimeError("database disconnected"),
                },
            ),
        ),
    )
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    monkeypatch.setattr(proxy_server, "user_api_key_cache", UserApiKeyCache())


KEY: Final = _permission("perm-key", {"tickets": "bots"})
EMPTY: Final[Mapping[str, str]] = {}
TEAM: Final = {"tickets": "ops"}


@pytest.mark.asyncio
@pytest.mark.usefixtures("rust_permission_rows")
@pytest.mark.parametrize(
    ("caller", "expected"),
    [
        pytest.param(_caller(), (TEAM,), id="a-single-team-grants-by-itself"),
        pytest.param(_caller(key=KEY), ({"tickets": "bots"}, TEAM), id="the-key-comes-before-the-team"),
        pytest.param(
            _caller(key=_permission("perm-empty")), (EMPTY, TEAM), id="a-key-without-grants-still-comes-before-the-team"
        ),
        pytest.param(
            _caller(key=KEY, mode="autonomous"),
            ({"tickets": "agents"}, {"tickets": "bots"}, TEAM),
            id="an-autonomous-agent-comes-before-the-key",
        ),
        pytest.param(
            _caller(key=KEY, mode="delegated"),
            (),
            id="a-delegated-agent-holds-no-carrier",
        ),
        pytest.param(_caller(key=KEY).model_copy(update={"is_session_token": True}), (TEAM,), id="session-team-only"),
        pytest.param(
            _caller(key=KEY, team_id=None), ({"tickets": "bots"},), id="a-key-without-a-team-has-no-team-step"
        ),
        pytest.param(
            _caller(key=KEY, team_id=UI_TEAM_ID), ({"tickets": "bots"},), id="the-dashboard-team-holds-no-carrier"
        ),
        pytest.param(_caller(key=KEY, team_id=""), ({"tickets": "bots"},), id="an-empty-team-id-holds-no-carrier"),
        pytest.param(
            _caller(key=_permission(""), team_id=None).model_copy(update={"object_permission_id": ""}),
            (),
            id="an-empty-key-permission-id-holds-no-carrier",
        ),
        pytest.param(
            _caller(key=KEY, admitted_subject=True),
            (),
            id="an-admitted-user-subject-skips-the-key-and-team-steps",
        ),
        pytest.param(None, (), id="no-caller-holds-no-carrier"),
    ],
)
async def test_grants_follow_the_most_specific_carrier_first(
    caller: UserAPIKeyAuth | None, expected: tuple[Mapping[str, str], ...]
) -> None:
    assert await caller_auth_profile_grants(caller) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("caller", "expected"),
    [
        pytest.param(
            UserAPIKeyAuth(api_key="hashed-key", object_permission_id="perm-bots"),
            ({"tickets": "bots"},),
            id="a-key-cached-without-its-row-is-hydrated-by-id",
        ),
        pytest.param(UserAPIKeyAuth(api_key="hashed-key"), (), id="a-key-without-a-row-grants-nothing"),
    ],
)
@pytest.mark.usefixtures("rust_permission_rows")
async def test_a_key_cached_without_its_permission_row_is_hydrated_by_id(
    caller: UserAPIKeyAuth, expected: tuple[Mapping[str, str], ...]
) -> None:
    assert await caller_auth_profile_grants(caller) == expected


@pytest.mark.asyncio
@pytest.mark.usefixtures("rust_permission_rows")
async def test_a_key_naming_a_permission_row_that_cannot_be_found_has_unreadable_grants() -> None:
    caller: Final = UserAPIKeyAuth(api_key="hashed-key", object_permission_id="perm-missing")
    assert await caller_auth_profile_grants(caller) == UnreadableGrants()


@pytest.mark.asyncio
async def test_a_key_naming_a_permission_row_on_a_proxy_without_a_database_has_unreadable_grants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    caller: Final = UserAPIKeyAuth(api_key="hashed-key", object_permission_id="perm-missing")
    assert await caller_auth_profile_grants(caller) == UnreadableGrants()


@pytest.mark.asyncio
@pytest.mark.usefixtures("rust_permission_rows")
@pytest.mark.parametrize(("link", "expected"), [("perm-team", (TEAM,)), ("perm-missing", UnreadableGrants())])
async def test_a_team_cached_without_its_permission_reads_it_by_id(
    link: str, expected: tuple[Mapping[str, str], ...] | UnreadableGrants
) -> None:
    await proxy_server.user_api_key_cache.async_set_cache(
        key=AuthObject.team("payments"), value=LiteLLM_TeamTableCachedObj(team_id="payments", object_permission_id=link)
    )
    assert await caller_auth_profile_grants(_caller()) == expected


@pytest.mark.asyncio
@pytest.mark.usefixtures("rust_permission_rows")
async def test_a_team_with_an_empty_permission_id_holds_no_carrier() -> None:
    await proxy_server.user_api_key_cache.async_set_cache(
        key=AuthObject.team("payments"), value=LiteLLM_TeamTableCachedObj(team_id="payments", object_permission_id="")
    )
    assert await caller_auth_profile_grants(_caller(key=KEY)) == ({"tickets": "bots"},)


@pytest.mark.asyncio
@pytest.mark.usefixtures("rust_permission_rows")
@pytest.mark.parametrize(
    ("teams", "expected"), [(("payments",), (TEAM,)), (("payments", "payments"), (TEAM,)), (("payments", "design"), ())]
)
async def test_jwt_only_inherits_one_unambiguous_team(
    monkeypatch: pytest.MonkeyPatch, teams: tuple[str, ...], expected: tuple[Mapping[str, str], ...]
) -> None:
    monkeypatch.setattr(proxy_server, "litellm_jwtauth", LiteLLM_JWTAuth(team_ids_jwt_field="groups"))
    caller: Final = _caller().model_copy(update={"jwt_claims": {"groups": list(teams)}})
    assert await caller_auth_profile_grants(caller) == expected


@pytest.mark.asyncio
@pytest.mark.usefixtures("rust_permission_rows")
async def test_a_jwt_without_team_claims_keeps_the_callers_team(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proxy_server, "litellm_jwtauth", LiteLLM_JWTAuth(team_ids_jwt_field="groups"))
    caller: Final = _caller().model_copy(update={"jwt_claims": {"sub": "reader"}})
    assert await caller_auth_profile_grants(caller) == (TEAM,)


@pytest.mark.asyncio
async def test_unreadable_agent_link_cannot_degrade_to_the_keys_grant(monkeypatch: pytest.MonkeyPatch) -> None:
    prisma: Final = MagicMock(**{"writer_db.litellm_agentstable.find_unique": AsyncMock(side_effect=ConnectionError)})
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", UserApiKeyCache())
    caller: Final = _caller(key=KEY, team_id=None).model_copy(update={"agent_id": "agents-runner"})
    assert await caller_auth_profile_grants(caller) == UnreadableGrants()


@pytest.mark.asyncio
@pytest.mark.usefixtures("rust_permission_rows")
async def test_a_managed_agent_without_a_permission_holds_no_agent_grant() -> None:
    policy: Final = AgentResponse(agent_id="agents-runner", agent_name="agents-runner", agent_card_params={})
    caller: Final = _caller(key=KEY, mode="autonomous").model_copy(update={"managed_agent_policy": policy})
    assert await caller_auth_profile_grants(caller) == ({"tickets": "bots"}, TEAM)


@pytest.mark.asyncio
@pytest.mark.usefixtures("rust_permission_rows")
async def test_a_scoped_permission_cannot_substitute_for_a_missing_key_permission() -> None:
    caller: Final = _caller(key=KEY).model_copy(update={"object_permission_id": "original"})
    assert await caller_auth_profile_grants(caller) == UnreadableGrants()


@pytest.mark.asyncio
async def test_a_validated_malformed_key_row_returns_unreadable_grants() -> None:
    row: Final = LiteLLM_ObjectPermissionTable.model_validate(
        {"object_permission_id": "perm-key", "mcp_auth_profiles": {"tickets": ["Build"]}}
    )
    assert row.mcp_auth_profiles is StoredProfileGrantError.UNREADABLE
    assert await caller_auth_profile_grants(_caller(key=row, team_id=None)) == UnreadableGrants()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("current_agent", "expected"),
    [("original", ()), (None, UnreadableGrants()), ("new-agent", UnreadableGrants())],
)
async def test_a_key_rebound_to_another_agent_since_admission_has_unreadable_grants(
    monkeypatch: pytest.MonkeyPatch,
    current_agent: str | None,
    expected: tuple[Mapping[str, str], ...] | UnreadableGrants,
) -> None:
    caller: Final = UserAPIKeyAuth(api_key="fixture", agent_id="original")
    caller.via_virtual_key = True
    rows: Final = FakeKeyStore(
        rows={"fixture": {"token": "fixture", "agent_id": "original"}},
        writer={"fixture": {"token": "fixture", "agent_id": current_agent}},
    )
    install_rust_auth_stores(monkeypatch, RustAuthStores(key=rows))
    agents: Final = SimpleNamespace(find_unique=AsyncMock(return_value=None))
    monkeypatch.setattr(
        proxy_server, "prisma_client", SimpleNamespace(writer_db=SimpleNamespace(litellm_agentstable=agents))
    )
    refreshed: Final = await MCPRequestHandler.refresh_catalog_authority(caller)
    assert refreshed is not None and refreshed.agent_id == "original"
    assert await caller_auth_profile_grants(refreshed) == expected


@pytest.mark.asyncio
@pytest.mark.usefixtures("rust_permission_rows")
async def test_team_cache_entry_without_the_grant_column_uses_the_saved_grant() -> None:
    await proxy_server.user_api_key_cache.async_set_cache(
        key=AuthObject.team("payments"),
        value=LiteLLM_TeamTableCachedObj(
            team_id="payments",
            object_permission_id="perm-team",
            object_permission=LiteLLM_ObjectPermissionTable(object_permission_id="perm-team"),
        ),
    )
    assert await caller_auth_profile_grants(_caller()) == (TEAM,)


@pytest.mark.asyncio
@pytest.mark.parametrize("fresh", [False, True])
@pytest.mark.parametrize("grants", [None, {}, {"tickets": "ops"}])
async def test_a_team_loaded_from_the_database_needs_one_permission_query(
    monkeypatch: pytest.MonkeyPatch, grants: Mapping[str, str] | None, fresh: bool
) -> None:
    rows: Final = RustStoreRows()
    rows.seed("perm-team", _permission("perm-team", grants))
    install_rust_auth_stores(
        monkeypatch,
        RustAuthStores(
            team=FakeTeamStore(rows={"payments": {"team_id": "payments", "object_permission_id": "perm-team"}}),
            object_permission=rows,
        ),
    )
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    monkeypatch.setattr(proxy_server, "user_api_key_cache", UserApiKeyCache())
    caller: Final = _caller().model_copy(update={"requires_fresh_policy": fresh})
    assert await caller_auth_profile_grants(caller) == (grants or {},)
    assert rows.lookups == [("perm-team", fresh)]


@pytest.mark.asyncio
@pytest.mark.parametrize("grants", [None, {}, {"tickets": "bots"}])
async def test_a_key_loaded_from_the_database_needs_one_permission_query(
    monkeypatch: pytest.MonkeyPatch, grants: Mapping[str, str] | None
) -> None:
    rows: Final = RustStoreRows()
    rows.seed("perm-key", _permission("perm-key", grants))
    install_rust_auth_stores(
        monkeypatch,
        RustAuthStores(
            key=FakeKeyStore(rows={"hashed-key": {"token": "hashed-key", "object_permission_id": "perm-key"}}),
            object_permission=rows,
        ),
    )
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    monkeypatch.setattr(proxy_server, "user_api_key_cache", UserApiKeyCache())
    caller: Final = await get_key_object("hashed-key", proxy_server.prisma_client, proxy_server.user_api_key_cache)
    assert await caller_auth_profile_grants(caller.model_copy(update={"team_id": None})) == (grants or {},)
    assert rows.lookups == [("perm-key", False)]


@pytest.mark.asyncio
@pytest.mark.parametrize("grants", [None, {}, {"tickets": "bots"}])
async def test_a_grant_column_read_needs_only_one_writer_read(
    monkeypatch: pytest.MonkeyPatch, grants: Mapping[str, str] | None
) -> None:
    rows: Final = RustStoreRows(writer={"linked": _permission("linked", grants)})
    install_rust_auth_stores(monkeypatch, RustAuthStores(object_permission=rows))
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    monkeypatch.setattr(proxy_server, "user_api_key_cache", UserApiKeyCache())
    old: Final = LiteLLM_ObjectPermissionTable(object_permission_id="linked")
    await proxy_server.user_api_key_cache.async_set_cache(key=AuthObject.object_permission("linked"), value=old)
    caller: Final = _caller(key=old, team_id=None)
    assert await caller_auth_profile_grants(caller) == (grants or {},)
    assert await caller_auth_profile_grants(caller) == (grants or {},)
    assert rows.lookups == [("linked", True)]
    fresh: Final = caller.model_copy(update={"requires_fresh_policy": True})
    assert await caller_auth_profile_grants(fresh) == (grants or {},)
    assert rows.lookups == [("linked", True), ("linked", True)]


@pytest.mark.asyncio
@pytest.mark.usefixtures("rust_permission_rows")
@pytest.mark.parametrize(
    ("row", "expected"),
    [
        pytest.param(None, (), id="a-config-agent-without-a-row-holds-no-carrier"),
        pytest.param(SimpleNamespace(object_permission_id=None), (), id="an-agent-linking-no-permission"),
        pytest.param(SimpleNamespace(object_permission_id="perm-team"), (TEAM,), id="an-agent-linking-a-permission"),
        pytest.param(SimpleNamespace(object_permission_id="perm-missing"), UnreadableGrants(), id="a-dangling-link"),
    ],
)
async def test_agent_grants_follow_the_current_link(
    monkeypatch: pytest.MonkeyPatch,
    row: SimpleNamespace | None,
    expected: tuple[Mapping[str, str], ...] | UnreadableGrants,
) -> None:
    table: Final = SimpleNamespace(find_unique=AsyncMock(return_value=row))
    monkeypatch.setattr(
        proxy_server, "prisma_client", SimpleNamespace(writer_db=SimpleNamespace(litellm_agentstable=table))
    )
    assert await caller_auth_profile_grants(UserAPIKeyAuth(agent_id="runner")) == expected


@pytest.mark.asyncio
@pytest.mark.usefixtures("rust_permission_rows")
async def test_a_config_agent_without_a_row_keeps_the_keys_grant(monkeypatch: pytest.MonkeyPatch) -> None:
    table: Final = SimpleNamespace(find_unique=AsyncMock(return_value=None))
    monkeypatch.setattr(
        proxy_server, "prisma_client", SimpleNamespace(writer_db=SimpleNamespace(litellm_agentstable=table))
    )
    caller: Final = _caller(key=KEY, team_id=None).model_copy(update={"agent_id": "config-agent"})
    assert await caller_auth_profile_grants(caller) == ({"tickets": "bots"},)


@pytest.mark.asyncio
@pytest.mark.parametrize("grants", [None, {}, {"tickets": "Build"}])
async def test_catalog_refresh_and_grant_read_share_one_writer_permission(
    monkeypatch: pytest.MonkeyPatch, grants: Mapping[str, str] | None
) -> None:
    key: Final = FakeKeyStore(writer={"fixture": {"token": "fixture", "object_permission_id": "linked"}})
    rows: Final = RustStoreRows(writer={"linked": _permission("linked", grants)})
    install_rust_auth_stores(monkeypatch, RustAuthStores(key=key, object_permission=rows))
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    monkeypatch.setattr(proxy_server, "user_api_key_cache", UserApiKeyCache())
    caller: Final = UserAPIKeyAuth(api_key="fixture", object_permission_id="linked")
    caller.via_virtual_key = True

    refreshed: Final = await MCPRequestHandler.refresh_catalog_authority(caller)
    assert refreshed is not None
    assert await caller_auth_profile_grants(refreshed) == (grants or {},)
    assert key.lookups == [("fixture", True)]
    assert rows.lookups == [("linked", True)]

    rows.writer["linked"] = _permission("linked", TEAM)
    assert await caller_auth_profile_grants(refreshed) == (TEAM,)
    assert rows.lookups == [("linked", True), ("linked", True)]


@pytest.mark.asyncio
@pytest.mark.usefixtures("rust_permission_rows")
async def test_legacy_agent_cache_absence_checks_the_current_link(monkeypatch: pytest.MonkeyPatch) -> None:
    await proxy_server.user_api_key_cache.async_set_cache(
        key=AuthObject.agent_object_permission("runner"), value="__agent_no_mcp_permission__"
    )
    prisma: Final = MagicMock(
        **{
            "db.litellm_agentstable.find_unique": AsyncMock(return_value=None),
            "writer_db.litellm_agentstable.find_unique": AsyncMock(
                return_value=SimpleNamespace(object_permission_id="perm-team")
            ),
        }
    )
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)
    assert await caller_auth_profile_grants(UserAPIKeyAuth(agent_id="runner")) == (TEAM,)


@pytest.mark.asyncio
@pytest.mark.usefixtures("rust_permission_rows")
async def test_a_writer_timeout_reading_a_missing_grant_column_records_the_stall() -> None:
    caller: Final = _caller(key=LiteLLM_ObjectPermissionTable(object_permission_id="timeout"), team_id=None)
    db_lookup_stall_tracker.clear()
    try:
        assert await caller_auth_profile_grants(caller) == UnreadableGrants()
        assert db_lookup_stall_tracker.stalled_within(math.inf)
    finally:
        db_lookup_stall_tracker.clear()


@pytest.mark.asyncio
@pytest.mark.usefixtures("rust_permission_rows")
@pytest.mark.parametrize("link", ["timeout", "disconnected"])
async def test_fresh_key_permission_store_failure_returns_unreadable_grants(link: str) -> None:
    caller: Final = UserAPIKeyAuth(
        object_permission_id=link, object_permission=_permission(link, {"tickets": "retired"})
    )
    assert await caller_auth_profile_grants(caller) == ({"tickets": "retired"},)
    fresh: Final = caller.model_copy(update={"requires_fresh_policy": True})
    assert await caller_auth_profile_grants(fresh) == UnreadableGrants()


@pytest.mark.asyncio
@pytest.mark.usefixtures("rust_permission_rows")
@pytest.mark.parametrize("fresh", [False, True])
async def test_an_agent_reads_its_link_and_grant_from_the_writer(monkeypatch: pytest.MonkeyPatch, fresh: bool) -> None:
    stale: Final = AsyncMock(return_value=SimpleNamespace(object_permission_id="perm-team"))
    current: Final = AsyncMock(return_value=SimpleNamespace(object_permission_id="agent-current"))
    prisma: Final = MagicMock(
        **{"db.litellm_agentstable.find_unique": stale, "writer_db.litellm_agentstable.find_unique": current}
    )
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)
    caller: Final = UserAPIKeyAuth(agent_id="runner", team_id="payments", requires_fresh_policy=fresh)
    assert await caller_auth_profile_grants(caller) == ({"tickets": "bots"}, TEAM)
    stale.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("caller", [UserAPIKeyAuth(agent_id="runner"), UserAPIKeyAuth(team_id="payments")])
async def test_agents_and_teams_hold_no_grants_without_a_database(
    monkeypatch: pytest.MonkeyPatch, caller: UserAPIKeyAuth
) -> None:
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    assert await caller_auth_profile_grants(caller) == ()


@pytest.mark.asyncio
@pytest.mark.usefixtures("rust_permission_rows")
@pytest.mark.parametrize("admitted_grants", [None, {"tickets": "retired"}])
@pytest.mark.parametrize(
    ("link", "expected"), [("agent-current", ({"tickets": "bots"},)), ("agent-deleted", UnreadableGrants())]
)
async def test_a_managed_agent_reads_its_permissions_current_grants(
    link: str, admitted_grants: Mapping[str, str] | None, expected: tuple[Mapping[str, str], ...] | UnreadableGrants
) -> None:
    stale: Final = _permission(link, {"tickets": "retired"})
    await proxy_server.user_api_key_cache.async_set_cache(key=AuthObject.object_permission(link), value=stale)
    admitted: Final = {
        "object_permission_id": link,
        **({} if admitted_grants is None else {"mcp_auth_profiles": admitted_grants}),
    }
    policy: Final = AgentResponse(
        agent_id="agent", agent_name="agent", agent_card_params={}, object_permission=admitted
    )
    caller: Final = UserAPIKeyAuth(agent_id=policy.agent_id).model_copy(
        update={"managed_agent_policy": policy, "requires_fresh_policy": True}
    )
    assert await caller_auth_profile_grants(caller) == expected


@pytest.mark.asyncio
@pytest.mark.usefixtures("rust_permission_rows")
async def test_returned_grants_are_read_only_copies() -> None:
    key: Final = _permission("perm-key", {"tickets": "bots"})
    grants: Final = await caller_auth_profile_grants(_caller(key=key, team_id=None))
    assert not isinstance(grants, UnreadableGrants)
    with pytest.raises(TypeError):
        grants[0]["tickets"] = "ops"  # pyright: ignore[reportIndexIssue]  # the mutation under test must fail
    assert key.mcp_auth_profiles == {"tickets": "bots"}


@pytest.mark.asyncio
async def test_a_malformed_stored_grant_stays_unreadable_through_the_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    malformed: Final = LiteLLM_ObjectPermissionTable.model_validate(
        {"object_permission_id": "linked", "mcp_auth_profiles": ["Build"]}
    )
    rows: Final = RustStoreRows(writer={"linked": malformed})
    install_rust_auth_stores(monkeypatch, RustAuthStores(object_permission=rows))
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    monkeypatch.setattr(proxy_server, "user_api_key_cache", UserApiKeyCache())
    caller: Final = _caller(key=LiteLLM_ObjectPermissionTable(object_permission_id="linked"), team_id=None)
    assert await caller_auth_profile_grants(caller) == UnreadableGrants()
    assert await caller_auth_profile_grants(caller) == UnreadableGrants()
    assert rows.lookups == [("linked", True)]


@pytest.mark.asyncio
@pytest.mark.parametrize("grants", [None, {}, {"tickets": "bots"}])
async def test_a_key_served_from_the_cache_needs_no_grant_query(
    monkeypatch: pytest.MonkeyPatch, grants: Mapping[str, str] | None
) -> None:
    rows: Final = RustStoreRows()
    rows.seed("perm-key", _permission("perm-key", grants))
    install_rust_auth_stores(
        monkeypatch,
        RustAuthStores(
            key=FakeKeyStore(rows={"hashed-key": {"token": "hashed-key", "object_permission_id": "perm-key"}}),
            object_permission=rows,
        ),
    )
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    monkeypatch.setattr(proxy_server, "user_api_key_cache", UserApiKeyCache())
    _ = await get_key_object("hashed-key", proxy_server.prisma_client, proxy_server.user_api_key_cache)
    cached: Final = await get_key_object("hashed-key", proxy_server.prisma_client, proxy_server.user_api_key_cache)
    assert await caller_auth_profile_grants(cached.model_copy(update={"team_id": None})) == (grants or {},)
    assert rows.lookups == [("perm-key", False)]
