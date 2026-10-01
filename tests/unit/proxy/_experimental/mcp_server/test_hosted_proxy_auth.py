import asyncio
import hashlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Final

import pytest
from fastapi import HTTPException, Request
from pydantic import BaseModel

from litellm.caching.redis_cache import RedisCache
from litellm.proxy import proxy_server
from litellm.proxy._experimental.mcp_server.hosted_proxy_auth import (
    HOSTED_ACCESS_TTL,
    HOSTED_ADMIN_SCOPE,
    HOSTED_GRANT_TTL,
    MAX_HOSTED_REFRESHES,
    HostedFailure,
    HostedGrant,
    HostedProxyAuth,
    HostedScope,
    HostedTokens,
    RedisHostedGrantStore,
    authenticate_hosted_request,
    hosted_proxy_auth,
    hosted_redirect_is_allowed,
    hosted_supported_scopes,
    load_hosted_user,
)
from litellm.proxy._types import (
    LiteLLM_TeamMembership,
    LiteLLM_TeamTableCachedObj,
    LiteLLM_UserTable,
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.spend_tracking.budget_reservation import team_membership_reservation_cache_key

NOW: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)
CALLBACK: Final = "https://admin.example/oauth/callback"
RESOURCE: Final = "https://gateway.example"


class GrantStore:
    def __init__(self) -> None:
        self.grant: HostedGrant | None = None
        self.issued: bool = False
        self.spent: frozenset[str] = frozenset()
        self.available: bool = True
        self.revocation_available: bool = True

    async def read(self, grant_id: str) -> HostedGrant | HostedFailure:
        if not self.available:
            raise ConnectionError("store unavailable")
        return self.grant if self.grant is not None and self.grant.grant_id == grant_id else HostedFailure()

    async def replace(self, grant: HostedGrant, previous: HostedGrant | None, ttl: int) -> bool:
        if not self.available:
            raise ConnectionError("store unavailable")
        if self.grant != previous or (previous is None and self.issued):
            return False
        if previous is not None:
            self.spent = self.spent | frozenset({previous.refresh_hash, previous.access_hash})
        self.grant = grant
        self.issued = True
        return True

    async def delete(self, grant_id: str) -> None:
        if not self.available or not self.revocation_available:
            raise ConnectionError("store unavailable")
        if self.grant is not None and self.grant.grant_id == grant_id:
            self.grant = None

    async def revoke_replayed(self, grant_id: str, token_hash: str) -> None:
        if not self.available or not self.revocation_available:
            raise ConnectionError("store unavailable")
        if token_hash in self.spent:
            await self.delete(grant_id)


class UserLoader:
    def __init__(self) -> None:
        self.role = LitellmUserRoles.PROXY_ADMIN
        self.active = True

    async def __call__(self, user_id: str, team_id: str | None) -> UserAPIKeyAuth | HostedFailure:
        if not self.active:
            return HostedFailure()
        return UserAPIKeyAuth(user_id=user_id, team_id=team_id, user_role=self.role)


@pytest.fixture(autouse=True)
def trusted_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_PROXY_API_OAUTH_REDIRECT_URIS", CALLBACK)
    monkeypatch.delenv("LITELLM_PROXY_API_OAUTH_ADMIN_REDIRECT_URIS", raising=False)


async def issued(service: HostedProxyAuth, scope: HostedScope = "proxy:read") -> HostedTokens:
    result: Final = await service.issue("user", "team", "client", CALLBACK, RESOURCE, "code-id", scope)
    assert isinstance(result, HostedTokens), result
    return result


@pytest.mark.parametrize(
    "callback",
    [
        "http://admin.example/callback",
        "https://user:password@admin.example/callback",
        "https://[bad",
        "https://admin.example/#x",
    ],
)
def test_allowlisting_does_not_make_an_unsafe_callback_trusted(monkeypatch: pytest.MonkeyPatch, callback: str) -> None:
    monkeypatch.setenv("LITELLM_PROXY_API_OAUTH_REDIRECT_URIS", callback)
    assert not hosted_redirect_is_allowed(callback)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["custom_auth", "enable_oauth2_auth", "enable_oauth2_proxy_auth"])
async def test_exclusive_authorization_configuration_denies_hosted_identity(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    async def custom_policy(request: Request, api_key: str) -> UserAPIKeyAuth:
        raise HTTPException(status_code=403, detail="custom policy denied access")

    monkeypatch.setattr(proxy_server, "user_custom_auth", custom_policy if mode == "custom_auth" else None)
    monkeypatch.setattr(proxy_server, "general_settings", {mode: True} if mode != "custom_auth" else {})
    result: Final = await load_hosted_user("user", None)
    assert isinstance(result, HostedFailure)
    assert result.error == "invalid_grant"
    assert "custom or external auth is configured" in result.description


@pytest.mark.asyncio
async def test_missing_database_prevents_loading_hosted_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "user_custom_auth", None)
    monkeypatch.setattr(proxy_server, "general_settings", {})
    result: Final = await load_hosted_user("user", None)
    assert isinstance(result, HostedFailure)
    assert result.error == "temporarily_unavailable"


@pytest.mark.asyncio
async def test_missing_shared_redis_prevents_service_and_api_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proxy_server, "redis_usage_cache", None)
    service: Final = hosted_proxy_auth()
    assert isinstance(service, HostedFailure)
    assert service.error == "temporarily_unavailable"
    request: Final = Request({"type": "http", "method": "GET", "path": "/v1/models", "headers": []})
    with pytest.raises(HTTPException) as denied:
        await authenticate_hosted_request(request, "llm_hosted_invalid", "/v1/models")
    assert denied.value.status_code == 503


@pytest.mark.asyncio
async def test_tokens_are_opaque_and_only_hashes_are_stored() -> None:
    store: Final = GrantStore()
    tokens: Final = await issued(HostedProxyAuth(store, UserLoader(), NOW))
    assert store.grant is not None
    assert store.grant.access_hash == hashlib.sha256(tokens.access_token.encode()).hexdigest()
    assert store.grant.refresh_hash == hashlib.sha256(tokens.refresh_token.encode()).hexdigest()
    assert tokens.access_token not in store.grant.model_dump_json()
    assert tokens.refresh_token not in store.grant.model_dump_json()
    assert tokens.expires_in == HOSTED_ACCESS_TTL
    assert tokens.scope == "proxy:read"
    assert tokens.access_token not in repr(tokens)
    assert tokens.refresh_token not in repr(tokens)


@pytest.mark.asyncio
async def test_issuance_rejects_removed_callback_and_inactive_user(monkeypatch: pytest.MonkeyPatch) -> None:
    store: Final = GrantStore()
    loader: Final = UserLoader()
    service: Final = HostedProxyAuth(store, loader, NOW)
    monkeypatch.delenv("LITELLM_PROXY_API_OAUTH_REDIRECT_URIS")
    assert isinstance(await service.issue("user", "team", "client", CALLBACK, RESOURCE, "code-id"), HostedFailure)
    monkeypatch.setenv("LITELLM_PROXY_API_OAUTH_REDIRECT_URIS", CALLBACK)
    loader.active = False
    assert isinstance(await service.issue("user", "team", "client", CALLBACK, RESOURCE, "code-id"), HostedFailure)
    assert store.grant is None


@pytest.mark.asyncio
@pytest.mark.parametrize("replay", [False, True])
async def test_failed_revocation_is_retryable_and_never_reports_success(replay: bool) -> None:
    store: Final = GrantStore()
    service: Final = HostedProxyAuth(store, UserLoader(), NOW)
    tokens: Final = await issued(service)
    rotated: Final = await service.refresh(tokens.refresh_token, "client", RESOURCE)
    assert isinstance(rotated, HostedTokens)
    store.revocation_available = False
    failure: Final = (
        await service.refresh(tokens.refresh_token, "client", RESOURCE)
        if replay
        else await service.revoke(rotated.refresh_token, "client")
    )
    assert isinstance(failure, HostedFailure)
    assert failure.error == "temporarily_unavailable"
    store.revocation_available = True
    assert await service.revoke(rotated.refresh_token, "client") is None
    assert isinstance(await service.refresh(rotated.refresh_token, "client", RESOURCE), HostedFailure)


@pytest.mark.asyncio
async def test_store_outage_fails_closed_and_does_not_consume_authorization_code() -> None:
    store: Final = GrantStore()
    service: Final = HostedProxyAuth(store, UserLoader(), NOW)
    store.available = False
    failed: Final = await service.issue("user", "team", "client", CALLBACK, RESOURCE, "code-id")
    assert isinstance(failed, HostedFailure)
    assert failed.error == "temporarily_unavailable"
    store.available = True
    tokens: Final = await issued(service)
    store.available = False
    access: Final = await service.authenticate(tokens.access_token, RESOURCE, "/v1/models", "GET")
    refresh: Final = await service.refresh(tokens.refresh_token, "client", RESOURCE)
    revoke: Final = await service.revoke(tokens.refresh_token, "client")
    assert isinstance(access, HostedFailure) and access.error == "temporarily_unavailable"
    assert isinstance(refresh, HostedFailure) and refresh.error == "temporarily_unavailable"
    assert isinstance(revoke, HostedFailure) and revoke.error == "temporarily_unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "method"),
    [
        ("/key/generate", "POST"),
        ("/key/info", "GET"),
        ("/model/info", "GET"),
        ("/config/yaml", "GET"),
        ("/spend/logs", "GET"),
        ("/v1/chat/completions", "POST"),
        ("/mcp", "GET"),
        ("/sso/key/generate", "GET"),
        ("/global/spend", "POST"),
        ("/v1/models/../key/info", "GET"),
    ],
)
async def test_read_only_admin_token_cannot_write_read_secrets_or_call_llms(route: str, method: str) -> None:
    service: Final = HostedProxyAuth(GrantStore(), UserLoader(), NOW)
    tokens: Final = await issued(service)
    result: Final = await service.authenticate(tokens.access_token, RESOURCE, route, method)
    assert isinstance(result, HostedFailure)
    assert result.error == "insufficient_scope"


@pytest.mark.asyncio
async def test_read_only_authority_tracks_live_user_role_and_deactivation() -> None:
    loader: Final = UserLoader()
    service: Final = HostedProxyAuth(GrantStore(), loader, NOW)
    tokens: Final = await issued(service)
    assert isinstance(await service.authenticate(tokens.access_token, RESOURCE, "/global/spend", "GET"), UserAPIKeyAuth)
    loader.role = LitellmUserRoles.INTERNAL_USER
    denied: Final = await service.authenticate(tokens.access_token, RESOURCE, "/global/spend", "GET")
    assert isinstance(denied, HostedFailure)
    assert denied.error == "insufficient_scope"
    models: Final = await service.authenticate(tokens.access_token, RESOURCE, "/v1/models", "GET")
    assert isinstance(models, UserAPIKeyAuth)
    assert models.user_role == LitellmUserRoles.INTERNAL_USER
    loader.active = False
    assert isinstance(await service.authenticate(tokens.access_token, RESOURCE, "/v1/models", "GET"), HostedFailure)
    assert isinstance(await service.refresh(tokens.refresh_token, "client", RESOURCE), HostedFailure)


@pytest.mark.asyncio
async def test_removed_callback_stops_active_access_and_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    service: Final = HostedProxyAuth(GrantStore(), UserLoader(), NOW)
    tokens: Final = await issued(service)
    monkeypatch.delenv("LITELLM_PROXY_API_OAUTH_REDIRECT_URIS")
    assert isinstance(await service.authenticate(tokens.access_token, RESOURCE, "/v1/models", "GET"), HostedFailure)
    assert isinstance(await service.refresh(tokens.refresh_token, "client", RESOURCE), HostedFailure)


@pytest.mark.asyncio
async def test_rotation_does_not_extend_absolute_grant_lifetime() -> None:
    store: Final = GrantStore()
    service: Final = HostedProxyAuth(store, UserLoader(), NOW)
    tokens: Final = await issued(service)
    later: Final = HostedProxyAuth(store, UserLoader(), NOW + timedelta(seconds=HOSTED_GRANT_TTL - 10))
    rotated: Final = await later.refresh(tokens.refresh_token, "client", RESOURCE)
    assert isinstance(rotated, HostedTokens)
    assert rotated.expires_in == 10
    expired: Final = HostedProxyAuth(store, UserLoader(), NOW + timedelta(seconds=HOSTED_GRANT_TTL))
    assert isinstance(await expired.refresh(rotated.refresh_token, "client", RESOURCE), HostedFailure)
    assert isinstance(await expired.authenticate(rotated.access_token, RESOURCE, "/v1/models", "GET"), HostedFailure)


@pytest.mark.asyncio
async def test_concurrent_refresh_is_single_use_and_revokes_the_compromised_family() -> None:
    store: Final = GrantStore()
    service: Final = HostedProxyAuth(store, UserLoader(), NOW)
    tokens: Final = await issued(service)
    first_loaded: Final = asyncio.Event()
    both_loaded: Final = asyncio.Event()

    async def simultaneous_user_load(user_id: str, team_id: str | None) -> UserAPIKeyAuth:
        if first_loaded.is_set():
            both_loaded.set()
        else:
            first_loaded.set()
        await both_loaded.wait()
        return UserAPIKeyAuth(user_id=user_id, team_id=team_id, user_role=LitellmUserRoles.PROXY_ADMIN)

    racer: Final = HostedProxyAuth(store, simultaneous_user_load, NOW)
    results: Final = await asyncio.gather(
        racer.refresh(tokens.refresh_token, "client", RESOURCE),
        racer.refresh(tokens.refresh_token, "client", RESOURCE),
    )
    successes: Final = tuple(result for result in results if isinstance(result, HostedTokens))
    assert len(successes) == 1
    assert sum(isinstance(result, HostedFailure) for result in results) == 1
    assert isinstance(
        await service.authenticate(successes[0].access_token, RESOURCE, "/v1/models", "GET"), HostedFailure
    )


@pytest.mark.asyncio
async def test_refresh_replay_revokes_family_but_random_secret_does_not() -> None:
    service: Final = HostedProxyAuth(GrantStore(), UserLoader(), NOW)
    tokens: Final = await issued(service)
    rotated: Final = await service.refresh(tokens.refresh_token, "client", RESOURCE)
    assert isinstance(rotated, HostedTokens)
    forged: Final = rotated.refresh_token[:-43] + "x" * 43
    assert isinstance(await service.refresh(forged, "client", RESOURCE), HostedFailure)
    assert isinstance(await service.authenticate(rotated.access_token, RESOURCE, "/v1/models", "GET"), UserAPIKeyAuth)
    assert isinstance(await service.refresh(tokens.refresh_token, "client", RESOURCE), HostedFailure)
    assert isinstance(await service.authenticate(rotated.access_token, RESOURCE, "/v1/models", "GET"), HostedFailure)


@pytest.mark.asyncio
async def test_revoking_grant_does_not_allow_authorization_code_reuse() -> None:
    service: Final = HostedProxyAuth(GrantStore(), UserLoader(), NOW)
    tokens: Final = await issued(service)
    assert await service.revoke(tokens.refresh_token, "client") is None
    assert isinstance(await service.issue("user", "team", "client", CALLBACK, RESOURCE, "code-id"), HostedFailure)


@pytest.mark.asyncio
async def test_refresh_history_is_bounded() -> None:
    store: Final = GrantStore()
    service: Final = HostedProxyAuth(store, UserLoader(), NOW)
    tokens: Final = await issued(service)
    assert store.grant is not None
    store.grant = store.grant.model_copy(update={"refresh_count": MAX_HOSTED_REFRESHES - 1})
    rotated: Final = await service.refresh(tokens.refresh_token, "client", RESOURCE)
    assert isinstance(rotated, HostedTokens)
    assert store.grant.refresh_count == MAX_HOSTED_REFRESHES
    response: Final = await service.refresh(rotated.refresh_token, "client", RESOURCE)
    assert isinstance(response, HostedFailure)
    assert response.error == "invalid_grant"
    assert store.spent == frozenset(
        hashlib.sha256(token.encode()).hexdigest() for token in (tokens.refresh_token, tokens.access_token)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("rotated", [False, True])
async def test_expired_or_rotated_access_can_revoke_the_entire_grant(rotated: bool) -> None:
    store: Final = GrantStore()
    service: Final = HostedProxyAuth(store, UserLoader(), NOW)
    tokens: Final = await issued(service)
    current: Final = await service.refresh(tokens.refresh_token, "client", RESOURCE) if rotated else tokens
    assert isinstance(current, HostedTokens)
    later: Final = HostedProxyAuth(store, UserLoader(), NOW + timedelta(seconds=HOSTED_ACCESS_TTL))
    forged: Final = tokens.access_token[:-43] + "x" * 43
    assert await later.revoke(forged, "client") is None
    assert store.grant is not None
    assert await later.revoke(tokens.access_token, "different-client") is None
    assert store.grant is not None
    assert await later.revoke(tokens.access_token, "client") is None
    assert store.grant is None
    assert isinstance(await later.refresh(current.refresh_token, "client", RESOURCE), HostedFailure)


@pytest.mark.asyncio
@pytest.mark.parametrize("use_access_token", [False, True])
async def test_revocation_removes_refresh_and_active_api_access(use_access_token: bool) -> None:
    service: Final = HostedProxyAuth(GrantStore(), UserLoader(), NOW)
    tokens: Final = await issued(service)
    token: Final = tokens.access_token if use_access_token else tokens.refresh_token
    assert await service.revoke(token, "different-client") is None
    assert isinstance(await service.authenticate(tokens.access_token, RESOURCE, "/v1/models", "GET"), UserAPIKeyAuth)
    assert await service.revoke(token, "client") is None
    assert isinstance(await service.authenticate(tokens.access_token, RESOURCE, "/v1/models", "GET"), HostedFailure)
    assert isinstance(await service.refresh(tokens.refresh_token, "client", RESOURCE), HostedFailure)


@pytest.mark.asyncio
async def test_tokens_cannot_cross_client_gateway_or_token_kind_boundaries() -> None:
    service: Final = HostedProxyAuth(GrantStore(), UserLoader(), NOW)
    tokens: Final = await issued(service)
    assert isinstance(await service.refresh(tokens.refresh_token, "other-client", RESOURCE), HostedFailure)
    assert isinstance(await service.refresh(tokens.refresh_token, "client", "https://other.example"), HostedFailure)
    assert isinstance(await service.refresh(tokens.access_token, "client", RESOURCE), HostedFailure)
    assert isinstance(await service.authenticate(tokens.refresh_token, RESOURCE, "/v1/models", "GET"), HostedFailure)
    assert isinstance(
        await service.authenticate(tokens.access_token, "https://other.example", "/v1/models", "GET"), HostedFailure
    )


@pytest.mark.asyncio
async def test_expired_access_requires_refresh_and_missing_store_grant_fails_closed() -> None:
    store: Final = GrantStore()
    service: Final = HostedProxyAuth(store, UserLoader(), NOW)
    tokens: Final = await issued(service)
    expired: Final = HostedProxyAuth(store, UserLoader(), NOW + timedelta(seconds=HOSTED_ACCESS_TTL))
    assert isinstance(await expired.authenticate(tokens.access_token, RESOURCE, "/v1/models", "GET"), HostedFailure)
    rotated: Final = await expired.refresh(tokens.refresh_token, "client", RESOURCE)
    assert isinstance(rotated, HostedTokens)
    assert isinstance(await expired.authenticate(rotated.access_token, RESOURCE, "/v1/models", "GET"), UserAPIKeyAuth)
    store.grant = None
    assert isinstance(await expired.authenticate(rotated.access_token, RESOURCE, "/v1/models", "GET"), HostedFailure)
    assert isinstance(await expired.refresh(rotated.refresh_token, "client", RESOURCE), HostedFailure)


def test_discovery_and_callback_approval_distinguish_admin_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    assert hosted_supported_scopes() == ("proxy:read",)
    assert not hosted_redirect_is_allowed(CALLBACK, HOSTED_ADMIN_SCOPE)
    monkeypatch.setenv("LITELLM_PROXY_API_OAUTH_ADMIN_REDIRECT_URIS", CALLBACK)
    assert hosted_supported_scopes() == ("proxy:read", "proxy:admin")
    assert hosted_redirect_is_allowed(CALLBACK, HOSTED_ADMIN_SCOPE)
    assert not hosted_redirect_is_allowed(CALLBACK + "/other", HOSTED_ADMIN_SCOPE)
    monkeypatch.setenv("LITELLM_PROXY_API_OAUTH_ADMIN_REDIRECT_URIS", "http://unsafe.example/callback")
    monkeypatch.delenv("LITELLM_PROXY_API_OAUTH_REDIRECT_URIS")
    assert hosted_supported_scopes() == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("role", [LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY])
async def test_admin_scope_requires_approved_callback_and_current_full_admin(
    monkeypatch: pytest.MonkeyPatch, role: LitellmUserRoles
) -> None:
    loader: Final = UserLoader()
    store: Final = GrantStore()
    service: Final = HostedProxyAuth(store, loader, NOW)
    unapproved: Final = await service.issue("user", "team", "client", CALLBACK, RESOURCE, "code-id", HOSTED_ADMIN_SCOPE)
    assert isinstance(unapproved, HostedFailure)
    monkeypatch.setenv("LITELLM_PROXY_API_OAUTH_ADMIN_REDIRECT_URIS", CALLBACK)
    loader.role = role
    denied: Final = await service.issue("user", "team", "client", CALLBACK, RESOURCE, "code-id", HOSTED_ADMIN_SCOPE)
    assert isinstance(denied, HostedFailure) and denied.error == "insufficient_scope"
    assert store.grant is None
    loader.role = LitellmUserRoles.PROXY_ADMIN
    tokens: Final = await issued(service, HOSTED_ADMIN_SCOPE)
    assert tokens.scope == HOSTED_ADMIN_SCOPE
    assert isinstance(await service.authenticate(tokens.access_token, RESOURCE, "/key/generate", "POST"), UserAPIKeyAuth)
    loader.role = role
    assert isinstance(await service.authenticate(tokens.access_token, RESOURCE, "/key/generate", "POST"), HostedFailure)
    assert isinstance(await service.refresh(tokens.refresh_token, "client", RESOURCE), HostedFailure)


@pytest.mark.asyncio
async def test_callback_removal_revokes_admin_but_read_grant_never_upgrades(monkeypatch: pytest.MonkeyPatch) -> None:
    read_service: Final = HostedProxyAuth(GrantStore(), UserLoader(), NOW)
    read: Final = await issued(read_service)
    monkeypatch.setenv("LITELLM_PROXY_API_OAUTH_ADMIN_REDIRECT_URIS", CALLBACK)
    admin_service: Final = HostedProxyAuth(GrantStore(), UserLoader(), NOW)
    admin: Final = await issued(admin_service, HOSTED_ADMIN_SCOPE)
    rotated: Final = await read_service.refresh(read.refresh_token, "client", RESOURCE)
    assert isinstance(rotated, HostedTokens) and rotated.scope == "proxy:read"
    assert isinstance(await read_service.authenticate(rotated.access_token, RESOURCE, "/key/generate", "POST"), HostedFailure)
    monkeypatch.delenv("LITELLM_PROXY_API_OAUTH_ADMIN_REDIRECT_URIS")
    assert isinstance(await admin_service.authenticate(admin.access_token, RESOURCE, "/user/info", "GET"), HostedFailure)
    assert isinstance(await admin_service.refresh(admin.refresh_token, "client", RESOURCE), HostedFailure)
    assert isinstance(await read_service.authenticate(rotated.access_token, RESOURCE, "/models", "GET"), UserAPIKeyAuth)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "allowed"),
    [
        ("/key/generate", True), ("/user/info", True), ("/team/member_add", True),
        ("/team/model/add", True), ("/team/team-a/disable_logging", True),
        ("/budget/new", True), ("/budget/info", True), ("/budget/list", True),
        ("/organization/info", True), ("/organization/daily/activity", True),
        ("/spend/logs/ui/request-a", True), ("/v1/chat/completions", True),
        ("/v1beta/models/gemini:generateContent", True), ("/v1/messages", True),
        ("/authorize", False), ("/token", False), ("/introspect", False),
        ("/sso/key/generate", False), ("/session/logout", False), ("/user/password/change", False),
        ("/jwt/key/mapping/new", False), ("/global/spend/reset", False),
        ("/v1/mcp/server/oauth/server-a/token", False),
        ("/v1/mcp/server/server-a/oauth-user-credential/status", False),
        ("/v1/mcp/server/server-a/user-env-vars", False),
        ("/v1/realtime/client_secrets", False), ("/anthropic/v1/messages", False),
        ("/mcp", False), ("/key/../token", False),
    ],
)
async def test_admin_scope_only_reaches_supported_operations(
    monkeypatch: pytest.MonkeyPatch, route: str, allowed: bool
) -> None:
    monkeypatch.setenv("LITELLM_PROXY_API_OAUTH_ADMIN_REDIRECT_URIS", CALLBACK)
    service: Final = HostedProxyAuth(GrantStore(), UserLoader(), NOW)
    tokens: Final = await issued(service, HOSTED_ADMIN_SCOPE)
    result: Final = await service.authenticate(tokens.access_token, RESOURCE, route, "POST")
    assert isinstance(result, UserAPIKeyAuth) == allowed
    if not allowed:
        assert isinstance(result, HostedFailure) and result.error == "insufficient_scope"


@pytest.mark.asyncio
async def test_refresh_preserves_unexpired_access_and_family_revocation_reaches_every_generation() -> None:
    store: Final = GrantStore()
    service: Final = HostedProxyAuth(store, UserLoader(), NOW)
    original: Final = await issued(service)
    later: Final = HostedProxyAuth(store, UserLoader(), NOW + timedelta(seconds=10))
    refreshed: Final = await later.refresh(original.refresh_token, "client", RESOURCE)
    assert isinstance(refreshed, HostedTokens)
    assert refreshed.refresh_expires_in == HOSTED_GRANT_TTL - 10
    for token in (original.access_token, refreshed.access_token):
        assert isinstance(await later.authenticate(token, RESOURCE, "/models", "GET"), UserAPIKeyAuth)
    expired: Final = HostedProxyAuth(store, UserLoader(), NOW + timedelta(seconds=HOSTED_ACCESS_TTL))
    assert isinstance(await expired.authenticate(original.access_token, RESOURCE, "/models", "GET"), HostedFailure)
    assert isinstance(await expired.authenticate(refreshed.access_token, RESOURCE, "/models", "GET"), UserAPIKeyAuth)
    assert isinstance(await later.refresh(original.refresh_token, "client", RESOURCE), HostedFailure)
    for token in (original.access_token, refreshed.access_token):
        assert isinstance(await later.authenticate(token, RESOURCE, "/models", "GET"), HostedFailure)


class StoredGrantRedis(RedisCache):
    def __init__(self, value: str) -> None:
        self.value: Final = value

    def check_and_fix_namespace(self, key: str) -> str:
        return key

    async def async_eval(self, script: str, numkeys: int, *keys_and_args: str | bytes | float) -> object:
        return self.value


@pytest.mark.asyncio
async def test_pre_scope_redis_grant_requires_reconnection() -> None:
    store: Final = GrantStore()
    await issued(HostedProxyAuth(store, UserLoader(), NOW))
    assert store.grant is not None
    old_json: Final = store.grant.model_dump_json(exclude={"scope", "previous_access"})
    shared: Final = RedisHostedGrantStore(StoredGrantRedis(old_json), "test-master")
    assert isinstance(await shared.read(store.grant.grant_id), HostedFailure)


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
    loaded: Final = await load_hosted_user("user", "team")
    assert isinstance(loaded, UserAPIKeyAuth), loaded
    assert (loaded.user_role, loaded.team_rpm_limit, loaded.team_member_rpm_limit, loaded.team_member_tpm_limit) == (
        LitellmUserRoles.PROXY_ADMIN, 20, 3, 70
    )
    assert loaded.org_id == "team-org"
    assert loaded.team_models == ["allowed"]
    assert loaded.object_permission is None
    teams.row = team.model_copy(update={"blocked": True})
    assert isinstance(await load_hosted_user("user", "team"), HostedFailure)
    users.row = user.model_copy(update={"teams": [], "organization_id": "home-org"})
    assert isinstance(await load_hosted_user("user", "team"), HostedFailure)
    teamless: Final = await load_hosted_user("user", None)
    assert isinstance(teamless, UserAPIKeyAuth) and teamless.org_id == "home-org"
    users.row = user.model_copy(update={"teams": [], "metadata": {"scim_active": False}})
    assert isinstance(await load_hosted_user("user", None), HostedFailure)
