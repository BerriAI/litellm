import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from litellm.constants import UI_SESSION_TOKEN_TEAM_ID
from litellm.proxy._experimental.mcp_server.ui_session_utils import (
    build_effective_auth_contexts,
    clone_user_api_key_auth_with_team,
    granted_toolset_ids,
    toolset_grant_contexts,
    resolve_ui_session_team_ids,
)
from litellm.proxy._types import LiteLLM_ObjectPermissionTable, UserAPIKeyAuth


def test_clone_user_api_key_auth_with_team_creates_independent_copy():
    original = UserAPIKeyAuth(team_id="team-original", user_id="user-123")

    cloned = clone_user_api_key_auth_with_team(original, "team-override")

    assert cloned is not original
    assert cloned.team_id == "team-override"
    assert original.team_id == "team-original"


@pytest.mark.asyncio
async def test_resolve_ui_session_team_ids_returns_unique_ids(monkeypatch):
    user_auth = UserAPIKeyAuth(
        team_id=UI_SESSION_TOKEN_TEAM_ID,
        user_id="user-1",
    )

    fake_user = SimpleNamespace(
        teams=["team-a", "team-b", "team-a", "", None, "team-c"]
    )

    monkeypatch.setattr(
        "litellm.proxy.auth.auth_checks.get_user_object",
        AsyncMock(return_value=fake_user),
    )

    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(proxy_server, "prisma_client", object())
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", None)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", None)

    team_ids = await resolve_ui_session_team_ids(user_auth)

    assert team_ids == ["team-a", "team-b", "team-c"]


@pytest.mark.asyncio
async def test_resolve_ui_session_team_ids_short_circuits_when_not_ui_session():
    normal_user = UserAPIKeyAuth(team_id="regular-team", user_id="user-1")

    result = await resolve_ui_session_team_ids(normal_user)

    assert result == []


@pytest.mark.asyncio
async def test_build_effective_auth_contexts_returns_cloned_contexts(monkeypatch):
    user_auth = UserAPIKeyAuth(team_id=UI_SESSION_TOKEN_TEAM_ID, user_id="user-42")

    mock_resolve = AsyncMock(return_value=["team-one", "team-two"])
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.ui_session_utils.resolve_ui_session_team_ids",
        mock_resolve,
    )

    contexts = await build_effective_auth_contexts(user_auth)

    assert [ctx.team_id for ctx in contexts] == ["team-one", "team-two"]
    assert all(ctx is not user_auth for ctx in contexts)
    mock_resolve.assert_awaited_once_with(user_auth)


@pytest.mark.asyncio
async def test_build_effective_auth_contexts_returns_original_when_no_resolution(
    monkeypatch,
):
    user_auth = UserAPIKeyAuth(team_id="existing-team", user_id="user-7")

    mock_resolve = AsyncMock(return_value=[])
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.ui_session_utils.resolve_ui_session_team_ids",
        mock_resolve,
    )

    contexts = await build_effective_auth_contexts(user_auth)

    assert contexts == [user_auth]
    mock_resolve.assert_awaited_once_with(user_auth)


@pytest.mark.asyncio
async def test_build_effective_auth_contexts_handles_unpicklable_parent_span(
    monkeypatch,
):
    class DummySpan:
        def __init__(self) -> None:
            self._lock = threading.RLock()

    parent_span = DummySpan()
    user_auth = UserAPIKeyAuth(
        team_id=UI_SESSION_TOKEN_TEAM_ID,
        user_id="user-span",
        parent_otel_span=parent_span,
    )

    mock_resolve = AsyncMock(return_value=["team-span"])
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.ui_session_utils.resolve_ui_session_team_ids",
        mock_resolve,
    )

    contexts = await build_effective_auth_contexts(user_auth)

    assert contexts[0].team_id == "team-span"
    assert contexts[0].parent_otel_span is parent_span


@pytest.mark.asyncio
async def test_build_effective_auth_contexts_appends_admitted_user_context(monkeypatch):
    """LIT-4861: the dashboard session must resolve with the user's admitted identity so the
    page list and every per-server action endpoint see user-level grants the same way the
    gateway session does."""
    user_auth = UserAPIKeyAuth(team_id=UI_SESSION_TOKEN_TEAM_ID, user_id="user-42")
    admitted_auth = UserAPIKeyAuth(user_id="user-42")

    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.ui_session_utils.resolve_ui_session_team_ids",
        AsyncMock(return_value=["team-one"]),
    )
    reload_mock = AsyncMock(return_value=admitted_auth)
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler.reload_admitted_user",
        reload_mock,
    )

    contexts = await build_effective_auth_contexts(user_auth)

    assert contexts[-1].user_id == "user-42" and contexts[-1].team_id is None
    assert [ctx.team_id for ctx in contexts[:-1]] == ["team-one"]
    reload_mock.assert_awaited_once_with("user-42", requires_fresh_policy=False)


@pytest.mark.asyncio
async def test_build_effective_auth_contexts_never_widens_caller_passed_keys(monkeypatch):
    normal_user = UserAPIKeyAuth(team_id="regular-team", user_id="user-1")
    reload_mock = AsyncMock()
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler.reload_admitted_user",
        reload_mock,
    )

    contexts = await build_effective_auth_contexts(normal_user)

    assert contexts == [normal_user]
    reload_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_build_effective_auth_contexts_survives_admitted_reload_failure(monkeypatch):
    user_auth = UserAPIKeyAuth(team_id=UI_SESSION_TOKEN_TEAM_ID, user_id="user-9")

    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.ui_session_utils.resolve_ui_session_team_ids",
        AsyncMock(return_value=["team-a"]),
    )
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler.reload_admitted_user",
        AsyncMock(side_effect=HTTPException(status_code=503, detail="db down")),
    )

    contexts = await build_effective_auth_contexts(user_auth)

    assert [ctx.team_id for ctx in contexts] == ["team-a"]


@pytest.mark.asyncio
async def test_acting_user_auth_returns_admitted_subject_for_non_admin_sessions(monkeypatch):
    """LIT-4861: acting-as-user MCP routes must resolve a non-admin dashboard session as the
    admitted subject so tool ceilings, reachability, and limits bind exactly as on /mcp."""
    from litellm.proxy._experimental.mcp_server.ui_session_utils import acting_user_auth

    user_auth = UserAPIKeyAuth(team_id=UI_SESSION_TOKEN_TEAM_ID, user_id="user-42", user_role="internal_user")
    admitted_auth = UserAPIKeyAuth(user_id="user-42")
    reload_mock = AsyncMock(return_value=admitted_auth)
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler.reload_admitted_user",
        reload_mock,
    )

    result = await acting_user_auth(user_auth)

    assert result.user_id == "user-42" and result.team_id is None
    reload_mock.assert_awaited_once_with("user-42", requires_fresh_policy=False)


@pytest.mark.asyncio
async def test_acting_user_auth_keeps_admin_sessions_and_passed_keys_unchanged(monkeypatch):
    from litellm.proxy._experimental.mcp_server.ui_session_utils import acting_user_auth

    reload_mock = AsyncMock()
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler.reload_admitted_user",
        reload_mock,
    )

    admin_session = UserAPIKeyAuth(team_id=UI_SESSION_TOKEN_TEAM_ID, user_id="admin-1", user_role="proxy_admin")
    assert await acting_user_auth(admin_session) is admin_session

    passed_key = UserAPIKeyAuth(team_id="regular-team", user_id="user-1", user_role="internal_user")
    assert await acting_user_auth(passed_key) is passed_key

    reload_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_acting_user_auth_falls_back_to_session_auth_on_reload_failure(monkeypatch):
    from litellm.proxy._experimental.mcp_server.ui_session_utils import acting_user_auth

    user_auth = UserAPIKeyAuth(team_id=UI_SESSION_TOKEN_TEAM_ID, user_id="user-9", user_role="internal_user")
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler.reload_admitted_user",
        AsyncMock(side_effect=HTTPException(status_code=503, detail="db down")),
    )

    assert await acting_user_auth(user_auth) is user_auth


@pytest.mark.asyncio
async def test_admitted_user_context_carries_the_request_span(monkeypatch):
    """Swapping the principal must not drop the request: the admitted subject is rebuilt from the
    user row and carries no span of its own, so every consumer would otherwise lose trace linkage
    for the resolution and logging it drives."""
    from litellm.proxy._experimental.mcp_server.ui_session_utils import acting_user_auth

    class DummySpan:
        def __init__(self) -> None:
            self._lock = threading.RLock()

    parent_span = DummySpan()
    user_auth = UserAPIKeyAuth(
        team_id=UI_SESSION_TOKEN_TEAM_ID,
        user_id="user-42",
        user_role="internal_user",
        parent_otel_span=parent_span,
    )
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler.reload_admitted_user",
        AsyncMock(return_value=UserAPIKeyAuth(user_id="user-42")),
    )

    assert (await acting_user_auth(user_auth)).parent_otel_span is parent_span
    assert (await build_effective_auth_contexts(user_auth))[-1].parent_otel_span is parent_span


def _toolset_permission(*toolset_ids: str) -> LiteLLM_ObjectPermissionTable:
    return LiteLLM_ObjectPermissionTable(
        object_permission_id=f"op-{'-'.join(toolset_ids)}", mcp_toolsets=list(toolset_ids)
    )


@pytest.mark.asyncio
async def test_granted_toolset_ids_unions_own_and_team_grants_over_every_effective_context():
    """A dashboard session of a user in two teams holds the toolsets of both teams plus the ones on
    the user row itself, exactly the grant sources the aggregate /mcp listing expands."""
    session = UserAPIKeyAuth(team_id=UI_SESSION_TOKEN_TEAM_ID, user_id="user-1")
    team_a = UserAPIKeyAuth(team_id="team-a", user_id="user-1")
    team_b = UserAPIKeyAuth(team_id="team-b", user_id="user-1", object_permission=_toolset_permission())
    admitted = UserAPIKeyAuth(user_id="user-1", object_permission=_toolset_permission("ts-user"))
    team_grants = {"team-a": _toolset_permission("ts-a", "ts-shared"), "team-b": _toolset_permission("ts-b")}

    async def effective_contexts(auth: UserAPIKeyAuth) -> list[UserAPIKeyAuth]:
        assert auth is session
        return [team_a, team_b, admitted]

    async def team_permission(auth: UserAPIKeyAuth) -> LiteLLM_ObjectPermissionTable | None:
        return team_grants.get(auth.team_id or "")

    granted = await granted_toolset_ids(session, effective_contexts, team_permission)

    assert granted == frozenset({"ts-a", "ts-shared", "ts-b", "ts-user"})


@pytest.mark.asyncio
async def test_granted_toolset_ids_is_empty_when_neither_key_nor_team_grants_a_toolset():
    key = UserAPIKeyAuth(api_key="sk-test", team_id="team-a", object_permission=_toolset_permission())

    async def effective_contexts(auth: UserAPIKeyAuth) -> list[UserAPIKeyAuth]:
        return [auth]

    async def no_team_permission(auth: UserAPIKeyAuth) -> LiteLLM_ObjectPermissionTable | None:
        return None

    assert await granted_toolset_ids(key, effective_contexts, no_team_permission) == frozenset()


async def _same_context(auth: UserAPIKeyAuth) -> list[UserAPIKeyAuth]:
    return [auth]


async def _team_grants_ts_team(auth: UserAPIKeyAuth) -> LiteLLM_ObjectPermissionTable | None:
    return _toolset_permission("ts-team")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "own",
    [
        LiteLLM_ObjectPermissionTable(object_permission_id="op", mcp_toolsets=["ts-own"]),
        LiteLLM_ObjectPermissionTable(object_permission_id="op", mcp_servers=["srv-own"]),
        LiteLLM_ObjectPermissionTable(object_permission_id="op", mcp_tool_permissions={"srv-own": ["add"]}),
        LiteLLM_ObjectPermissionTable(object_permission_id="op", mcp_access_groups=["group-own"]),
    ],
)
async def test_a_key_declaring_its_own_mcp_grant_does_not_inherit_the_team_toolsets(own):
    """The key/team rule of the aggregate listing: a key's own MCP grant is a ceiling the team cannot widen."""
    key = UserAPIKeyAuth(api_key="sk-test", team_id="team-a", object_permission=own)

    granted = await granted_toolset_ids(key, _same_context, _team_grants_ts_team, require_key_access=False)

    assert granted == frozenset(own.mcp_toolsets or ())


@pytest.mark.asyncio
async def test_require_key_mcp_access_defined_stops_a_key_inheriting_team_toolsets_but_not_a_session():
    key = UserAPIKeyAuth(api_key="sk-test", team_id="team-a")
    session = UserAPIKeyAuth(team_id=UI_SESSION_TOKEN_TEAM_ID, user_id="user-1")

    assert await granted_toolset_ids(key, _same_context, _team_grants_ts_team, require_key_access=False) == {"ts-team"}
    assert await granted_toolset_ids(key, _same_context, _team_grants_ts_team, require_key_access=True) == frozenset()
    assert await granted_toolset_ids(session, _same_context, _team_grants_ts_team, require_key_access=True) == {
        "ts-team"
    }


@pytest.mark.asyncio
async def test_toolset_grant_contexts_of_a_virtual_key_is_the_key_alone():
    key = UserAPIKeyAuth(api_key="sk-test", team_id="team-a")

    async def never(auth: UserAPIKeyAuth) -> None:
        raise AssertionError("a virtual key has no admitted sources")

    assert await toolset_grant_contexts(key, admitted_context=never, admitted_sources=never) == (key,)


def _admitted(user_id: str, own: LiteLLM_ObjectPermissionTable | None = None) -> UserAPIKeyAuth:
    subject = UserAPIKeyAuth(user_id=user_id, object_permission=own)
    subject.mcp_admitted_user_subject = True
    return subject


@pytest.mark.asyncio
async def test_toolset_grant_contexts_of_a_dashboard_session_are_its_admitted_users_grant_sources():
    """The dashboard session fans out through the same roster-checked source builder as the aggregate
    /mcp resolution, applied to the admitted user it acts as, so a cached membership a team has since
    revoked never reaches the toolset check."""
    session = UserAPIKeyAuth(team_id=UI_SESSION_TOKEN_TEAM_ID, user_id="user-1")
    admitted = _admitted("user-1")
    own_source = UserAPIKeyAuth(user_id="user-1")
    team_source = UserAPIKeyAuth(user_id="user-1", team_id="team-a")

    async def admitted_context(auth: UserAPIKeyAuth) -> UserAPIKeyAuth:
        assert auth is session
        return admitted

    async def admitted_sources(auth: UserAPIKeyAuth) -> list[UserAPIKeyAuth]:
        assert auth is admitted
        return [own_source, team_source]

    assert await toolset_grant_contexts(session, admitted_context, admitted_sources) == (own_source, team_source)


@pytest.mark.asyncio
async def test_toolset_grant_contexts_of_a_gateway_admitted_user_are_its_own_grant_sources():
    admitted = _admitted("user-1")
    team_source = UserAPIKeyAuth(user_id="user-1", team_id="team-a")

    async def no_dashboard_context(auth: UserAPIKeyAuth) -> None:
        return None

    async def admitted_sources(auth: UserAPIKeyAuth) -> list[UserAPIKeyAuth]:
        assert auth is admitted
        return [team_source]

    assert await toolset_grant_contexts(admitted, no_dashboard_context, admitted_sources) == (team_source,)


@pytest.mark.asyncio
async def test_a_source_declaring_its_own_mcp_grant_never_reads_its_team():
    key = UserAPIKeyAuth(api_key="sk-test", team_id="team-a", object_permission=_toolset_permission("ts-own"))
    team_reads: list[str | None] = []  # mutable-ok: records the lookups the code under test performs

    async def team_permission(auth: UserAPIKeyAuth) -> LiteLLM_ObjectPermissionTable | None:
        team_reads.append(auth.team_id)
        return _toolset_permission("ts-team")

    granted = await granted_toolset_ids(key, _same_context, team_permission, require_key_access=False)

    assert granted == {"ts-own"}
    assert team_reads == []


async def _team_a_unreadable(auth: UserAPIKeyAuth) -> LiteLLM_ObjectPermissionTable | None:
    if auth.team_id == "team-a":
        raise RuntimeError("team row unreadable")
    return _toolset_permission("ts-b")


@pytest.mark.asyncio
async def test_an_unreadable_team_grants_nothing_while_the_direct_and_other_team_grants_still_count():
    """A dashboard user whose own row grants ts-user and who sits on team-a and team-b keeps ts-user and
    ts-b when team-a cannot be read; team-a itself contributes nothing rather than failing the lookup."""
    admitted = _admitted("user-1", _toolset_permission("ts-user"))
    team_a = UserAPIKeyAuth(user_id="user-1", team_id="team-a")
    team_b = UserAPIKeyAuth(user_id="user-1", team_id="team-b")

    async def sources(auth: UserAPIKeyAuth) -> list[UserAPIKeyAuth]:
        return [admitted, team_a, team_b]

    assert await granted_toolset_ids(admitted, sources, _team_a_unreadable) == {"ts-user", "ts-b"}


@pytest.mark.asyncio
async def test_a_key_whose_only_grant_source_is_an_unreadable_team_is_granted_nothing():
    key = UserAPIKeyAuth(api_key="sk-test", team_id="team-a")

    assert await granted_toolset_ids(key, _same_context, _team_a_unreadable, require_key_access=False) == frozenset()


async def _hydrates_op_key_to_srv_own(auth: UserAPIKeyAuth) -> LiteLLM_ObjectPermissionTable | None:
    if auth.object_permission is not None:
        return auth.object_permission
    if auth.object_permission_id == "op-key":
        return LiteLLM_ObjectPermissionTable(object_permission_id="op-key", mcp_servers=["srv-own"])
    return None


@pytest.mark.asyncio
async def test_a_key_cached_with_its_own_grant_unhydrated_is_scoped_to_that_grant_not_its_team():
    """The main auth flow can cache a key with object_permission_id set and object_permission None. The
    row it names is the key's ceiling, so it is loaded and read as the key's own grant instead of letting the
    key inherit its team's toolsets."""
    key = UserAPIKeyAuth(api_key="sk-test", team_id="team-a", object_permission_id="op-key")

    granted = await granted_toolset_ids(
        key,
        _same_context,
        _team_grants_ts_team,
        require_key_access=False,
        own_object_permission=_hydrates_op_key_to_srv_own,
    )

    assert granted == frozenset()


async def _own_row_unreadable(auth: UserAPIKeyAuth) -> LiteLLM_ObjectPermissionTable | None:
    raise RuntimeError("object permission row unreadable")


async def _own_row_gone(auth: UserAPIKeyAuth) -> LiteLLM_ObjectPermissionTable | None:
    return None


@pytest.mark.asyncio
@pytest.mark.parametrize("load_own", [_own_row_unreadable, _own_row_gone])
async def test_a_key_naming_an_own_grant_that_cannot_be_read_is_granted_nothing_rather_than_its_team(load_own):
    key = UserAPIKeyAuth(api_key="sk-test", team_id="team-a", object_permission_id="op-key")

    granted = await granted_toolset_ids(
        key, _same_context, _team_grants_ts_team, require_key_access=False, own_object_permission=load_own
    )

    assert granted == frozenset()


@pytest.mark.asyncio
async def test_a_key_naming_no_own_grant_is_not_hydrated_before_inheriting_its_team():
    key = UserAPIKeyAuth(api_key="sk-test", team_id="team-a")

    granted = await granted_toolset_ids(
        key, _same_context, _team_grants_ts_team, require_key_access=False, own_object_permission=_own_row_unreadable
    )

    assert granted == {"ts-team"}
