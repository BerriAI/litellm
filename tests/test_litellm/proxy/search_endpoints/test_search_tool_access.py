import os
import sys
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.constants import UI_SESSION_TOKEN_TEAM_ID
from litellm.proxy._types import (
    LiteLLM_ObjectPermissionTable,
    LiteLLM_TeamTable,
    LitellmUserRoles,
    ProxyException,
    UserAPIKeyAuth,
)
from litellm.proxy.search_endpoints.search_tool_access import (
    authorize_search_tool,
    resolve_allowlist_teams,
    session_team_ids_from_db,
)


def _team(team_id: str, search_tools: list[str] | None = None) -> LiteLLM_TeamTable:
    return LiteLLM_TeamTable(
        team_id=team_id,
        object_permission=(
            LiteLLM_ObjectPermissionTable(object_permission_id=f"op-{team_id}", search_tools=search_tools)
            if search_tools is not None
            else None
        ),
    )


def _dashboard_session(search_tools: list[str] | None = None) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="internal_user",
        team_id=UI_SESSION_TOKEN_TEAM_ID,
        object_permission=(
            LiteLLM_ObjectPermissionTable(object_permission_id="op-key", search_tools=search_tools)
            if search_tools is not None
            else None
        ),
    )


def _key_on_team(team_id: str) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="internal_user",
        team_id=team_id,
    )


def _lookup_failing_with_404(team_id: str) -> AsyncMock:
    return AsyncMock(
        side_effect=HTTPException(
            status_code=404,
            detail={"error": f"Team doesn't exist in db. Team={team_id}."},
        )
    )


@pytest.mark.asyncio
async def test_resolve_returns_no_teams_when_the_key_has_none():
    lookup = AsyncMock()
    session_team_ids = AsyncMock()

    assert await resolve_allowlist_teams(_key_on_team(""), lookup, session_team_ids) == ()
    lookup.assert_not_awaited()
    session_team_ids.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_looks_up_a_real_team_and_never_widens_to_the_session_path():
    team = _team("team-1")
    lookup = AsyncMock(return_value=team)
    session_team_ids = AsyncMock()

    assert await resolve_allowlist_teams(_key_on_team("team-1"), lookup, session_team_ids) == (team,)
    assert [awaited.args[0] for awaited in lookup.await_args_list] == ["team-1"]
    session_team_ids.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_propagates_a_real_team_lookup_failure():
    """A key naming a team that does not exist must be rejected, not treated as unscoped."""
    lookup = _lookup_failing_with_404("deleted-team")

    with pytest.raises(HTTPException) as exc_info:
        await resolve_allowlist_teams(_key_on_team("deleted-team"), lookup, AsyncMock())

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_resolve_maps_the_session_sentinel_onto_the_users_real_teams():
    """The sentinel itself is never looked up; the session user's own teams are."""
    teams = (_team("team-1"), _team("team-2"))
    lookup = AsyncMock(side_effect=teams)
    session_team_ids = AsyncMock(return_value=["team-1", "team-2"])

    assert await resolve_allowlist_teams(_dashboard_session(), lookup, session_team_ids) == teams
    assert [awaited.args[0] for awaited in lookup.await_args_list] == ["team-1", "team-2"]


@pytest.mark.asyncio
async def test_resolve_drops_only_the_session_team_that_cannot_be_loaded():
    """Dropping an unloadable team can only narrow the union, so the rest still bind."""
    loadable = _team("team-2")

    async def lookup_impl(team_id, _user_api_key_dict):
        if team_id == "team-1":
            raise HTTPException(status_code=404, detail={"error": "Team doesn't exist in db. Team=team-1."})
        return loadable

    session_team_ids = AsyncMock(return_value=["team-1", "team-2"])

    resolved = await resolve_allowlist_teams(_dashboard_session(), AsyncMock(side_effect=lookup_impl), session_team_ids)

    assert resolved == (loadable,)


@pytest.mark.asyncio
async def test_resolve_denies_a_session_when_no_team_can_be_loaded():
    """
    The collapse case. Dropping teams narrows a union only while one survives; once none do there
    is nothing left to narrow, and returning () would read as 'belongs to no team' and go
    unrestricted. The failure has to surface instead.
    """
    lookup = _lookup_failing_with_404("team-1")
    session_team_ids = AsyncMock(return_value=["team-1", "team-2"])

    with pytest.raises(HTTPException) as exc_info:
        await resolve_allowlist_teams(_dashboard_session(), lookup, session_team_ids)

    assert exc_info.value.status_code == 404
    assert lookup.await_count == 2


@pytest.mark.asyncio
async def test_resolve_surfaces_a_failure_to_resolve_the_session_users_teams():
    """A session user whose own record cannot be read is not a session user with no teams."""
    session_team_ids = AsyncMock(side_effect=ValueError("User doesn't exist in db. 'user_id'=internal_user."))

    with pytest.raises(ValueError, match="User doesn't exist in db"):
        await resolve_allowlist_teams(_dashboard_session(), AsyncMock(), session_team_ids)


@pytest.mark.asyncio
async def test_resolve_returns_no_teams_for_a_session_user_on_no_teams():
    """The one empty result that is genuinely unscoped, and the reason the collapse needs its own case."""
    lookup = AsyncMock()

    assert await resolve_allowlist_teams(_dashboard_session(), lookup, AsyncMock(return_value=[])) == ()
    lookup.assert_not_awaited()


@pytest.mark.asyncio
async def test_session_team_ids_do_not_swallow_a_failure_to_load_the_session_user(monkeypatch):
    """Resolution reads the session user directly so a DB failure cannot arrive as an empty team list."""
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)

    with pytest.raises(Exception, match="No db connected"):
        await session_team_ids_from_db(_dashboard_session())


@pytest.mark.asyncio
async def test_authorize_permits_a_tool_any_one_team_allowlists():
    assert (
        await authorize_search_tool(
            search_tool_name="db-tool-2",
            user_api_key_dict=_dashboard_session(),
            teams=(_team("team-1", ["db-tool-1"]), _team("team-2", ["db-tool-2"])),
        )
        is True
    )


@pytest.mark.asyncio
async def test_authorize_denies_a_tool_no_team_allowlists():
    with pytest.raises(ProxyException) as exc_info:
        await authorize_search_tool(
            search_tool_name="db-tool-3",
            user_api_key_dict=_dashboard_session(),
            teams=(_team("team-1", ["db-tool-1"]), _team("team-2", ["db-tool-2"])),
        )

    assert exc_info.value.code == "403"
    assert "db-tool-3" in exc_info.value.message


@pytest.mark.asyncio
async def test_authorize_is_unrestricted_at_team_level_with_no_teams():
    assert (
        await authorize_search_tool(
            search_tool_name="db-tool-3",
            user_api_key_dict=_dashboard_session(),
            teams=(),
        )
        is True
    )


@pytest.mark.asyncio
async def test_authorize_enforces_the_key_allowlist_before_any_team_grant():
    """A team grant never widens a key past its own allowlist."""
    with pytest.raises(ProxyException) as exc_info:
        await authorize_search_tool(
            search_tool_name="db-tool-3",
            user_api_key_dict=_dashboard_session(["db-tool-1"]),
            teams=(_team("team-1", ["db-tool-3"]),),
        )

    assert exc_info.value.code == "403"
