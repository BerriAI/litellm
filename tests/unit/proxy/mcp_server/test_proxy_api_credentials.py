"""Tests for minting the ``lite login`` credential from a consented native-client grant."""

from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.constants import CLI_JWT_EXPIRATION_HOURS
from litellm.models.user import LiteLLM_UserTable
from litellm.proxy._types import LitellmUserRoles
from litellm.proxy.auth.auth_checks import ExperimentalUIJWTToken
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.management_endpoints.ui_sso import CliSsoTeamDetail
from litellm.proxy.mcp_server.gateway_dcr_flow import ConsentTeam, MintedProxyCredential
from litellm.proxy.mcp_server.proxy_api_credentials import (
    lookup_consent_teams,
    lookup_consent_teams_from_database,
    mint_proxy_credential,
)

_LOAD_USER = "litellm.proxy.mcp_server.proxy_api_credentials.load_active_user_by_id"
_FETCH_TEAMS = "litellm.proxy.mcp_server.proxy_api_credentials.fetch_cli_sso_team_details"
_PRISMA = "litellm.proxy.proxy_server.prisma_client"
TEAM_DETAILS = (
    CliSsoTeamDetail(team_id="team-a", team_alias="Team A", team_models=("gpt-5.4-mini",)),
    CliSsoTeamDetail(team_id="team-b", team_models=(), team_model_aliases={"fast": "gpt-5.4-mini"}),
)


def _user(**overrides) -> LiteLLM_UserTable:
    return LiteLLM_UserTable(
        **{
            "user_id": "u1",
            "user_role": "internal_user",
            "teams": ["team-a", "team-b"],
            "models": ["gpt-5.4"],
            **overrides,
        }
    )


def _decoded(minted: MintedProxyCredential):
    key_object = ExperimentalUIJWTToken.get_key_object_from_ui_hash_key(minted.key)
    assert key_object is not None
    return key_object


@pytest.fixture(autouse=True)
def _salt_key(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-proxy-api-credentials-tests")


@pytest.fixture
def fetch_teams(monkeypatch):
    monkeypatch.setattr(_PRISMA, object(), raising=False)
    fetch = AsyncMock(return_value=TEAM_DETAILS)
    monkeypatch.setattr(_FETCH_TEAMS, fetch)
    return fetch


@pytest.fixture
def load_user(monkeypatch):
    load = AsyncMock(return_value=_user())
    monkeypatch.setattr(_LOAD_USER, load)
    return load


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["unavailable", "unresolvable", "no_active_key"])
async def test_mint_passes_user_lookup_failures_through(failure, load_user, fetch_teams):
    load_user.return_value = failure
    assert await mint_proxy_credential("u1", "team-a") == failure
    fetch_teams.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stored_role, minted_role",
    [
        (None, LitellmUserRoles.INTERNAL_USER),
        ("made_up_role", LitellmUserRoles.INTERNAL_USER),
        ("proxy_admin", LitellmUserRoles.PROXY_ADMIN),
    ],
)
async def test_mint_carries_the_role_the_proxy_enforces_for_the_user(load_user, fetch_teams, stored_role, minted_role):
    """A user JWT auth upserted has no role in the database, and the proxy already treats
    such a user as an internal user on every request, so the credential says the same."""
    load_user.return_value = _user(user_role=stored_role)
    minted = await mint_proxy_credential("u1", "team-a")
    assert isinstance(minted, MintedProxyCredential)
    assert _decoded(minted).user_role == minted_role


@pytest.mark.asyncio
async def test_mint_refuses_a_teamless_grant_for_a_team_member(load_user, fetch_teams):
    """The consent page is the only place a team gets chosen, so a grant sealed without one
    is refused for a user with teams instead of minting an unscoped credential or drifting
    onto the first team, on redemption and on every refresh alike."""
    assert await mint_proxy_credential("u1", None) == "team_required"
    load_user.assert_awaited_once_with("u1", source="database")
    fetch_teams.assert_awaited_once_with(ANY, ["team-a", "team-b"])


@pytest.mark.asyncio
async def test_mint_for_a_member_of_only_deleted_teams_is_unscoped(load_user, fetch_teams):
    """Memberships whose team rows are gone offer nothing to pick, so they never lock the
    user out of signing in."""
    load_user.return_value = _user(teams=["team-gone"])
    fetch_teams.return_value = ()
    minted = await mint_proxy_credential("u1", None)
    assert isinstance(minted, MintedProxyCredential)
    assert minted.user_id == "u1"
    assert minted.team_id is None
    assert minted.expires_in == CLI_JWT_EXPIRATION_HOURS * 3600
    decoded = _decoded(minted)
    assert decoded.user_id == "u1"
    assert decoded.team_id is None
    assert decoded.team_alias is None
    assert decoded.models == ["gpt-5.4"]
    assert decoded.is_session_token is True


@pytest.mark.asyncio
async def test_mint_honors_the_consented_team(load_user, fetch_teams):
    minted = await mint_proxy_credential("u1", "team-b")
    assert isinstance(minted, MintedProxyCredential)
    assert minted.team_id == "team-b"
    decoded = _decoded(minted)
    assert decoded.team_id == "team-b"
    assert decoded.team_alias is None
    assert decoded.team_models == []
    assert decoded.team_model_aliases == {"fast": "gpt-5.4-mini"}


@pytest.mark.asyncio
async def test_mint_reads_the_users_teams_from_the_database_not_a_stale_cached_row(fetch_teams, monkeypatch):
    """JWT auth caches the user it creates before it adds that user to the JWT's team, and adding a member
    never evicts the cached row, so a mint off the cached row refused the very first token exchange as not
    a member. The mint has to read the database row, whatever the cache holds."""
    from litellm.proxy import proxy_server

    cache = UserApiKeyCache()
    await cache.async_set_cache(
        key="stale-cache-user", value=_user(user_id="stale-cache-user", teams=[]), model_type=LiteLLM_UserTable
    )
    prisma = MagicMock()
    prisma.writer_db.litellm_usertable.find_unique = AsyncMock(
        return_value=_user(user_id="stale-cache-user", teams=["team-a"])
    )
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)

    minted = await mint_proxy_credential("stale-cache-user", "team-a")

    assert isinstance(minted, MintedProxyCredential)
    assert minted.team_id == "team-a"
    assert _decoded(minted).team_id == "team-a"


@pytest.mark.asyncio
async def test_mint_refuses_a_user_scim_deactivated_after_the_cache_last_saw_them_active(fetch_teams, monkeypatch):
    """SCIM deactivation writes the user row without evicting the cached copy, so a mint off the cache would
    keep issuing credentials for the management-object TTL. The mint reads the database row, so the
    deactivated user is refused on the first refresh after the deactivation."""
    from litellm.proxy import proxy_server

    cache = UserApiKeyCache()
    await cache.async_set_cache(
        key="deactivated-user", value=_user(user_id="deactivated-user", teams=["team-a"]), model_type=LiteLLM_UserTable
    )
    prisma = MagicMock()
    prisma.writer_db.litellm_usertable.find_unique = AsyncMock(
        return_value=_user(user_id="deactivated-user", teams=["team-a"], metadata={"scim_active": False})
    )
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)

    assert await mint_proxy_credential("deactivated-user", "team-a") == "no_active_key"
    fetch_teams.assert_not_awaited()


@pytest.mark.asyncio
async def test_mint_refuses_a_team_the_user_is_not_on(load_user, fetch_teams):
    assert await mint_proxy_credential("u1", "team-c") == "not_a_member"
    fetch_teams.assert_not_awaited()


@pytest.mark.asyncio
async def test_mint_refuses_when_the_teams_grants_are_unknown(load_user, fetch_teams):
    fetch_teams.return_value = TEAM_DETAILS[:1]
    assert await mint_proxy_credential("u1", "team-b") == "not_a_member"


@pytest.mark.asyncio
async def test_mint_reports_unavailable_when_the_team_lookup_fails(load_user, fetch_teams):
    fetch_teams.return_value = None
    assert await mint_proxy_credential("u1", "team-a") == "unavailable"


@pytest.mark.asyncio
async def test_mint_reports_unavailable_without_a_database(load_user, monkeypatch):
    monkeypatch.setattr(_PRISMA, None, raising=False)
    assert await mint_proxy_credential("u1", "team-a") == "unavailable"


@pytest.mark.asyncio
async def test_mint_for_a_teamless_user_is_unscoped(load_user, fetch_teams):
    load_user.return_value = _user(teams=[])
    minted = await mint_proxy_credential("u1", None)
    assert isinstance(minted, MintedProxyCredential)
    assert minted.team_id is None
    fetch_teams.assert_not_awaited()
    decoded = _decoded(minted)
    assert decoded.team_id is None
    assert decoded.models == ["gpt-5.4"]


@pytest.mark.asyncio
async def test_lookup_consent_teams_lists_the_users_teams_with_aliases(load_user, fetch_teams):
    teams = await lookup_consent_teams("u1")
    assert teams == (ConsentTeam(team_id="team-a", team_alias="Team A"), ConsentTeam(team_id="team-b"))
    fetch_teams.assert_awaited_once_with(ANY, ["team-a", "team-b"])


@pytest.mark.asyncio
async def test_lookup_consent_teams_drops_details_without_a_team_id(load_user, fetch_teams):
    fetch_teams.return_value = (CliSsoTeamDetail(team_models=()), *TEAM_DETAILS[1:])
    assert await lookup_consent_teams("u1") == (ConsentTeam(team_id="team-b"),)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["unavailable", "unresolvable", "no_active_key"])
async def test_lookup_consent_teams_passes_user_lookup_failures_through(failure, load_user, fetch_teams):
    load_user.return_value = failure
    assert await lookup_consent_teams("u1") == failure
    fetch_teams.assert_not_awaited()


@pytest.mark.asyncio
async def test_lookup_consent_teams_reports_unavailable_when_details_cannot_load(load_user, fetch_teams):
    fetch_teams.return_value = None
    assert await lookup_consent_teams("u1") == "unavailable"


def _route_status(minted: MintedProxyCredential, route: str, method: str | None = None) -> int:
    request = None if method is None else MagicMock(method=method)
    try:
        RouteChecks.is_virtual_key_allowed_to_call_route(route=route, valid_token=_decoded(minted), request=request)
    except HTTPException as refused:
        return refused.status_code
    return 200


@pytest.mark.parametrize(
    "allowed_routes, chat_completions, mcp",
    [
        (None, True, True),
        (("mcp_inference_routes",), False, True),
        (("llm_api_routes_without_mcp",), True, False),
        (("mcp_inference_routes", "llm_api_routes_without_mcp"), True, True),
    ],
    ids=["full credential", "mcp only", "llm only", "both"],
)
@pytest.mark.asyncio
async def test_mint_narrows_the_credential_to_the_allowed_routes_the_route_check_enforces(
    load_user, fetch_teams, allowed_routes, chat_completions, mcp
):
    """The allowed_routes a grant's scope resolved to ride inside the credential, so the same route
    check a virtual key goes through on every request refuses /chat/completions for an MCP-only
    credential and /mcp for an LLM-only one, and a credential minted without them is unchanged."""
    minted = await mint_proxy_credential("u1", "team-a", allowed_routes=allowed_routes)
    assert isinstance(minted, MintedProxyCredential)
    assert _decoded(minted).allowed_routes == list(allowed_routes or ())
    chat_status, mcp_status = (200 if chat_completions else 403), (200 if mcp else 403)
    assert _route_status(minted, "/chat/completions") == chat_status
    assert _route_status(minted, "/v1/chat/completions") == chat_status
    assert _route_status(minted, "/mcp") == mcp_status
    assert _route_status(minted, "/mcp/tools/call") == mcp_status


@pytest.mark.asyncio
async def test_lookup_consent_teams_from_database_reads_the_row_the_minter_will_recheck(load_user, fetch_teams):
    """A grant consumed once a team is picked must pick it from the same membership the minter re-reads,
    or a stale cache would burn the assertion on a request the caller could have fixed."""
    teams = await lookup_consent_teams_from_database("u1")
    load_user.assert_awaited_once_with("u1", source="database")
    assert teams == (ConsentTeam(team_id="team-a", team_alias="Team A"), ConsentTeam(team_id="team-b"))


@pytest.mark.parametrize(
    "allowed_routes, registered_pass_through, auto_router_session, mcp_server_listing, agent_registry",
    [
        (("llm_api_routes",), 200, 200, 200, 200),
        (("llm_api_routes_without_mcp",), 200, 200, 403, 403),
        (("mcp_inference_routes",), 403, 403, 403, 403),
    ],
    ids=["llm_api_routes key", "llm only", "mcp only"],
)
@pytest.mark.asyncio
async def test_an_llm_only_credential_keeps_the_llm_api_carve_outs_except_mcp_discovery(
    load_user,
    fetch_teams,
    allowed_routes,
    registered_pass_through,
    auto_router_session,
    mcp_server_listing,
    agent_registry,
):
    """Operator-registered pass-through endpoints and the auto-router session read are LLM API
    surface that only the ``llm_api_routes`` group used to reach, so an LLM-only credential keeps
    them, while the MCP server listing that group also carves out stays MCP surface."""
    minted = await mint_proxy_credential("u1", "team-a", allowed_routes=allowed_routes)
    assert isinstance(minted, MintedProxyCredential)
    with patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.InitPassThroughEndpointHelpers.is_registered_pass_through_route",
        side_effect=lambda route: route.startswith("/my-llm-endpoint"),
    ):
        assert _route_status(minted, "/my-llm-endpoint/v1/chat", "POST") == registered_pass_through
    assert _route_status(minted, "/auto_router/session", "GET") == auto_router_session
    assert _route_status(minted, "/v1/mcp/server", "GET") == mcp_server_listing
    assert _route_status(minted, "/v1/agents", "GET") == agent_registry
