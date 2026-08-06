import contextlib
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.constants import UI_SESSION_TOKEN_TEAM_ID
from litellm.proxy._types import (
    LiteLLM_ObjectPermissionTable,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from litellm.proxy.proxy_server import app

SEARCH_TOOLS = [
    {
        "search_tool_name": "db-tool-1",
        "litellm_params": {"search_provider": "perplexity", "api_key": "pplx-secret-1"},
    },
    {
        "search_tool_name": "db-tool-2",
        "litellm_params": {"search_provider": "tavily", "api_key": "tvly-secret-2"},
    },
    {
        "search_tool_name": "db-tool-3",
        "litellm_params": {"search_provider": "exa", "api_key": "exa-secret-3"},
    },
]


@contextlib.contextmanager
def _override_auth(user):
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    app.dependency_overrides[user_api_key_auth] = lambda: user
    try:
        yield
    finally:
        app.dependency_overrides.pop(user_api_key_auth, None)


@contextlib.contextmanager
def _mock_search_backend(lookup_team_object, session_team_ids=()):
    """Patch the router, prisma client, and downstream request processing so /search reaches
    the authorization checks and, if they pass, returns without calling a provider."""
    router = MagicMock()
    router.search_tools = SEARCH_TOOLS
    processor = MagicMock()
    processor.return_value.base_process_llm_request = AsyncMock(return_value={"object": "search", "results": []})
    session_user = LiteLLM_UserTable(user_id="internal_user", teams=list(session_team_ids))
    with (
        patch("litellm.proxy.proxy_server.llm_router", router),
        patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
        patch("litellm.proxy.auth.auth_checks.get_team_object", lookup_team_object),
        patch("litellm.proxy.auth.auth_checks.get_user_object", AsyncMock(return_value=session_user)),
        patch(
            "litellm.proxy.search_endpoints.endpoints.ProxyBaseLLMRequestProcessing",
            processor,
        ),
    ):
        yield


def _team_ids_looked_up(lookup: AsyncMock) -> list[str]:
    return [awaited.kwargs["team_id"] for awaited in lookup.await_args_list]


def _dashboard_session_key(search_tools: list[str] | None) -> UserAPIKeyAuth:
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


def _team_allowing(team_id: str, search_tools: list[str]) -> LiteLLM_TeamTable:
    return LiteLLM_TeamTable(
        team_id=team_id,
        object_permission=LiteLLM_ObjectPermissionTable(
            object_permission_id=f"op-{team_id}",
            search_tools=search_tools,
        ),
    )


def _lookup_returning(*teams: LiteLLM_TeamTable) -> AsyncMock:
    by_id = {team.team_id: team for team in teams}

    async def lookup(**kwargs):
        team_id = kwargs["team_id"]
        if team_id not in by_id:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Team doesn't exist in db. Team={team_id}."},
            )
        return by_id[team_id]

    return AsyncMock(side_effect=lookup)


def _team_lookup_fails_with_404(team_id: str) -> AsyncMock:
    return AsyncMock(
        side_effect=HTTPException(
            status_code=404,
            detail={"error": f"Team doesn't exist in db. Team={team_id}."},
        )
    )


@pytest.mark.asyncio
async def test_search_dashboard_session_is_scoped_by_the_users_real_team_allowlist():
    """
    Regression: the Admin UI session key carries the reserved team id ``litellm-dashboard``, which
    has no row in LiteLLM_TeamTable. Treating that as "no team" dropped team-level authorization
    entirely, so a dashboard session with no key allowlist could invoke every tool on the proxy.
    The session must be scoped by the allowlists of the teams its user actually belongs to.
    """
    lookup = _lookup_returning(_team_allowing("team-1", ["db-tool-2"]))

    with (
        _mock_search_backend(lookup, session_team_ids=["team-1"]),
        _override_auth(_dashboard_session_key(None)),
    ):
        blocked = TestClient(app).post("/search/db-tool-1", json={"query": "litellm"})
        allowed = TestClient(app).post("/search/db-tool-2", json={"query": "litellm"})

    assert blocked.status_code == 403
    assert "db-tool-1" in blocked.text
    assert allowed.status_code == 200
    assert _team_ids_looked_up(lookup) == ["team-1", "team-1"]


@pytest.mark.asyncio
async def test_search_dashboard_session_is_permitted_by_any_of_the_users_teams():
    """A user on several teams may invoke a tool any one of those teams allowlists."""
    lookup = _lookup_returning(
        _team_allowing("team-1", ["db-tool-2"]),
        _team_allowing("team-2", ["db-tool-3"]),
    )

    with (
        _mock_search_backend(lookup, session_team_ids=["team-1", "team-2"]),
        _override_auth(_dashboard_session_key(None)),
    ):
        allowed = TestClient(app).post("/search/db-tool-3", json={"query": "litellm"})
        blocked = TestClient(app).post("/search/db-tool-1", json={"query": "litellm"})

    assert allowed.status_code == 200
    assert blocked.status_code == 403


@pytest.mark.asyncio
async def test_search_dashboard_session_key_does_not_look_up_the_ui_team():
    """
    Regression: resolving ``litellm-dashboard`` as a real team raised 404, so no non-admin could
    invoke a search tool from the dashboard.
    """
    ui_team_is_not_a_real_team = _team_lookup_fails_with_404(UI_SESSION_TOKEN_TEAM_ID)

    with (
        _mock_search_backend(ui_team_is_not_a_real_team),
        _override_auth(_dashboard_session_key(["db-tool-3"])),
    ):
        response = TestClient(app).post("/search/db-tool-3", json={"query": "litellm"})

    assert response.status_code == 200
    ui_team_is_not_a_real_team.assert_not_awaited()


@pytest.mark.asyncio
async def test_search_dashboard_session_denied_when_none_of_the_users_teams_load():
    """
    Regression: a dashboard session whose every team failed to load resolved to an empty team set,
    which reads as "belongs to no team" and skips team authorization altogether. A user who could
    invoke nothing must not become a user who can invoke everything.
    """
    every_team_is_gone = _team_lookup_fails_with_404("team-1")

    with (
        _mock_search_backend(every_team_is_gone, session_team_ids=["team-1", "team-2"]),
        _override_auth(_dashboard_session_key(None)),
    ):
        response = TestClient(app).post("/search/db-tool-1", json={"query": "litellm"})

    assert response.status_code == 404
    assert every_team_is_gone.await_count == 2


@pytest.mark.asyncio
async def test_search_dashboard_session_key_still_bound_by_its_key_allowlist():
    """
    Resolving the session through its user's teams must not widen access: a dashboard session
    whose key allowlists only one tool is still refused every other tool.
    """
    lookup = _lookup_returning(_team_allowing("team-1", []))

    with (
        _mock_search_backend(lookup, session_team_ids=["team-1"]),
        _override_auth(_dashboard_session_key(["db-tool-3"])),
    ):
        response = TestClient(app).post("/search/db-tool-1", json={"query": "litellm"})

    assert response.status_code == 403
    assert "db-tool-1" in response.text


@pytest.mark.asyncio
async def test_search_real_team_allowlist_still_blocks_a_tool_it_does_not_permit():
    """A caller with a real team is still resolved and scoped by that team's allowlist."""
    team_member = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="internal_user",
        team_id="team-1",
    )
    lookup = _lookup_returning(_team_allowing("team-1", ["db-tool-2"]))

    with _mock_search_backend(lookup), _override_auth(team_member):
        blocked = TestClient(app).post("/search/db-tool-1", json={"query": "litellm"})
        allowed = TestClient(app).post("/search/db-tool-2", json={"query": "litellm"})

    assert blocked.status_code == 403
    assert allowed.status_code == 200
    assert _team_ids_looked_up(lookup) == ["team-1", "team-1"]


@pytest.mark.asyncio
async def test_search_missing_real_team_is_still_rejected():
    """
    A caller whose real team cannot be resolved must not fall through to "no team", which
    would drop that team's allowlist and let it call tools it may not.
    """
    team_member = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="internal_user",
        team_id="deleted-team",
    )
    lookup = _team_lookup_fails_with_404("deleted-team")

    with _mock_search_backend(lookup), _override_auth(team_member):
        response = TestClient(app).post("/search/db-tool-1", json={"query": "litellm"})

    assert response.status_code == 404
    assert _team_ids_looked_up(lookup) == ["deleted-team"]
