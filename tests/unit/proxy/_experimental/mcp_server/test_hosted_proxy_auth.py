import hashlib
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from pydantic import BaseModel
from starlette.requests import Request

from litellm.proxy import proxy_server
from litellm.proxy._experimental.mcp_server.hosted_proxy_auth import (
    HostedFailure, HostedProxyAuth, authenticate_hosted_request, hosted_proxy_auth, load_hosted_user,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.session_credentials import session_keys_from_master_key
from litellm.proxy._experimental.mcp_server.outbound_credentials.session_token import (
    HostedAppGrant, MintedSessionToken, SessionPrincipal, mint_session_token,
)
from litellm.proxy._types import (
    LiteLLM_TeamMembership, LiteLLM_TeamTableCachedObj, LiteLLM_UserTable, LitellmUserRoles, UserAPIKeyAuth,
)
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.spend_tracking.budget_reservation import team_membership_reservation_cache_key

NOW: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)
CALLBACK: Final = "https://admin.example/oauth/callback"
RESOURCE: Final = "https://gateway.example"


class GrantStore:
    def __init__(self) -> None:
        self.current: str | None = None
        self.available = True

    async def read(self, grant_id: str) -> str | None:
        if not self.available:
            raise ConnectionError("unavailable")
        return self.current

    async def advance(self, grant_id: str, previous: str | None, refresh_jti: str, ttl: int) -> bool:
        await self.read(grant_id)
        if self.current != previous:
            self.current = "revoked" if previous is not None else self.current
            return False
        self.current = refresh_jti
        return True

    async def revoke(self, grant_id: str) -> None:
        await self.read(grant_id)
        self.current = "revoked"


async def admin_user(user_id: str, team_id: str | None) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(user_id=user_id, team_id=team_id, user_role=LitellmUserRoles.PROXY_ADMIN)


@pytest.mark.asyncio
async def test_app_sessions_require_live_family_callback_and_resource(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_PROXY_API_OAUTH_ADMIN_REDIRECT_URIS", CALLBACK)
    store: Final = GrantStore()
    service: Final = HostedProxyAuth(store, admin_user, NOW)
    principal: Final = SessionPrincipal(user_id="user", client_id="client", audience="proxy_api", app_grant=HostedAppGrant(
        grant_id="grant", redirect_uri=CALLBACK, resource=RESOURCE, expires_at=int(NOW.timestamp()) + 86400,
    ))
    assert isinstance(await service.validate(principal, RESOURCE), HostedFailure)
    assert await service.advance(principal, None, "first") is None
    assert isinstance(await service.validate(principal, RESOURCE), UserAPIKeyAuth)
    assert isinstance(await service.validate(principal, "https://other.example"), HostedFailure)
    assert isinstance(await service.validate(principal, RESOURCE, refresh_jti="spent"), HostedFailure)
    monkeypatch.delenv("LITELLM_PROXY_API_OAUTH_ADMIN_REDIRECT_URIS")
    assert isinstance(await service.validate(principal, RESOURCE), HostedFailure)
    monkeypatch.setenv("LITELLM_PROXY_API_OAUTH_ADMIN_REDIRECT_URIS", CALLBACK)
    store.available = False
    for result in (
        await service.validate(principal, RESOURCE), await service.advance(principal, "first", "next"),
        await service.revoke(principal),
    ):
        assert isinstance(result, HostedFailure) and result.error == "temporarily_unavailable"
    store.available = True
    assert await service.revoke(principal) is None
    assert isinstance(await service.validate(principal, RESOURCE), HostedFailure)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["custom_auth", "enable_oauth2_auth", "enable_oauth2_proxy_auth", "database", "redis"])
async def test_app_authorization_refuses_exclusive_modes_and_missing_authority(
    monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    monkeypatch.setattr(proxy_server, "user_custom_auth", object() if mode == "custom_auth" else None)
    monkeypatch.setattr(proxy_server, "general_settings", {mode: True})
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "redis_usage_cache", None)
    result: Final = hosted_proxy_auth() if mode == "redis" else await load_hosted_user("user", None)
    assert isinstance(result, HostedFailure)
    assert result.error == ("temporarily_unavailable" if mode in ("redis", "database") else "invalid_grant")


class DatabaseTable:
    def __init__(self, row: BaseModel | None) -> None:
        self.row = row

    async def find_unique(self, **kwargs: object) -> BaseModel | None:
        return self.row


@pytest.mark.asyncio
async def test_hosted_identity_reloads_user_team_and_member_limits_from_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache: Final = UserApiKeyCache()
    user: Final = LiteLLM_UserTable(user_id="user", user_role="proxy_admin", teams=["team"], organization_id="home-org")
    team: Final = LiteLLM_TeamTableCachedObj(team_id="team", organization_id="team-org", rpm_limit=20, models=["allowed"])
    member: Final = LiteLLM_TeamMembership.model_validate({
        "user_id": "user", "team_id": "team", "spend": 1,
        "litellm_budget_table": {"rpm_limit": 3, "tpm_limit": 70},
    })
    users: Final = DatabaseTable(user)
    teams: Final = DatabaseTable(team)
    members: Final = DatabaseTable(member)
    writer: Final = SimpleNamespace(litellm_usertable=users, litellm_teamtable=teams, litellm_teammembership=members)
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace(writer_db=writer, db=None))
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    monkeypatch.setattr(proxy_server, "user_custom_auth", None)
    monkeypatch.setattr(proxy_server, "general_settings", {})
    await cache.async_set_cache(key="user", value=user.model_copy(update={"user_role": "internal_user"}))
    await cache.async_set_cache(key="team_id:team", value=team.model_copy(update={"rpm_limit": 999}))
    await cache.async_set_cache(key=team_membership_reservation_cache_key("user", "team"), value=member.model_copy(
        update={"litellm_budget_table": None}
    ))
    monkeypatch.setenv("PROXY_BASE_URL", RESOURCE)
    monkeypatch.setenv("LITELLM_PROXY_API_OAUTH_ADMIN_REDIRECT_URIS", CALLBACK)
    monkeypatch.setattr(proxy_server, "master_key", "sk-hosted-test")
    monkeypatch.setattr(proxy_server, "redis_usage_cache", SimpleNamespace(
        check_and_fix_namespace=str, async_eval=AsyncMock(return_value="refresh"),
    ))
    now: Final = datetime.now(timezone.utc)
    principal: Final = SessionPrincipal(
        user_id="user", team_id="team", client_id="client", audience="proxy_api", app_grant=HostedAppGrant(
            grant_id="grant", redirect_uri=CALLBACK, resource=RESOURCE, expires_at=int(now.timestamp()) + 300,
        ),
    )
    minted: Final = mint_session_token(principal, session_keys_from_master_key("sk-hosted-test"), now)
    assert isinstance(minted, MintedSessionToken)
    token: Final = minted.token.get_secret_value()
    request: Final = Request({"type": "http", "headers": []})
    for route in ("/key/info", "/budget/update", "/v1/chat/completions", "/spend/logs"):
        loaded: Final = await authenticate_hosted_request(request, token, route)
        assert loaded.api_key == loaded.token == hashlib.sha256(token.encode()).hexdigest()
        assert (loaded.user_role, loaded.team_rpm_limit, loaded.team_member_rpm_limit, loaded.team_member_tpm_limit) == (
            LitellmUserRoles.PROXY_ADMIN, 20, 3, 70
        )
        assert loaded.org_id == "team-org"
        assert loaded.team_models == ["allowed"]
        assert loaded.object_permission is None
    for route in (
        "/mcp", "/sso/key/generate", "/config/yaml", "/user/password/change",
        "/v1/mcp/server/server-id/user-credentials", "/v1/models/../key/info", "/v1/realtime/client_secrets",
        "/v1/mcp/server/server-id/user-credential\n", "/v1/realtime/client_secrets\n",
        "/key/generate", "/key/regenerate", "/key/sk-1/regenerate", "/key/service-account/generate",
        "/user/new", "/invitation/new",
    ):
        with pytest.raises(HTTPException) as denied:
            await authenticate_hosted_request(request, token, route)
        assert denied.value.status_code == 403
    teams.row = team.model_copy(update={"blocked": True})
    assert isinstance(await load_hosted_user("user", "team"), HostedFailure)
    users.row = user.model_copy(update={"teams": [], "organization_id": "home-org"})
    assert isinstance(await load_hosted_user("user", "team"), HostedFailure)
    teamless: Final = await load_hosted_user("user", None)
    assert isinstance(teamless, UserAPIKeyAuth) and teamless.org_id == "home-org"
    users.row = user.model_copy(update={"teams": [], "metadata": {"scim_active": False}})
    assert isinstance(await load_hosted_user("user", None), HostedFailure)
