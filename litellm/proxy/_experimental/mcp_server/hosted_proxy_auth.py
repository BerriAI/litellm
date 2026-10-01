from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from functools import partial
from types import MappingProxyType
from typing import Final, Literal, Protocol
from urllib.parse import urlparse

from fastapi import HTTPException, Request
from pydantic import BaseModel, ConfigDict

from litellm.caching.redis_cache import RedisCache
from litellm.proxy._experimental.mcp_server.oauth_utils import get_request_base_url, validate_redirect_uri_shape
from litellm.proxy._experimental.mcp_server.outbound_credentials.session_credentials import (
    SessionSigningConfigError,
    active_session_signing_keys,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.session_token import (
    OpenedSessionToken,
    SessionPrincipal,
    open_session_token,
)
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth, hash_token
from litellm.proxy.auth.route_checks import RouteChecks

HOSTED_ADMIN_SCOPE: Final = "proxy:admin"


def hosted_redirect_is_allowed(redirect_uri: str) -> bool:
    try:
        parsed: Final = urlparse(redirect_uri)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            return False
        if "?" in redirect_uri or "#" in redirect_uri:
            return False
        validate_redirect_uri_shape(parsed)
    except (HTTPException, ValueError):
        return False
    return redirect_uri in tuple(
        entry.strip() for entry in os.environ.get("LITELLM_PROXY_API_OAUTH_ADMIN_REDIRECT_URIS", "").split(",")
    )


def hosted_supported_scopes() -> tuple[str, ...]:
    callbacks: Final = os.environ.get("LITELLM_PROXY_API_OAUTH_ADMIN_REDIRECT_URIS", "").split(",")
    return (HOSTED_ADMIN_SCOPE,) if any(hosted_redirect_is_allowed(uri.strip()) for uri in callbacks) else ()


class HostedFailure(BaseModel):
    model_config = ConfigDict(frozen=True)
    error: Literal["invalid_grant", "temporarily_unavailable", "insufficient_scope"] = "invalid_grant"
    description: str = "the application session is no longer valid; sign in again"


def _unavailable() -> HostedFailure:
    return HostedFailure(error="temporarily_unavailable", description="application authorization is unavailable; retry")


class HostedGrantStore(Protocol):
    async def read(self, grant_id: str) -> str | None: ...
    async def advance(self, grant_id: str, previous: str | None, refresh_jti: str, ttl: int) -> bool: ...
    async def revoke(self, grant_id: str) -> None: ...


_ADVANCE_GRANT: Final = """
if ARGV[1] == '' then
    return redis.call('SET', KEYS[1], ARGV[2], 'EX', ARGV[3], 'NX') and 1 or 0
end
local current = redis.call('GET', KEYS[1])
if not current or current == 'revoked' then return 0 end
if current ~= ARGV[1] then
    redis.call('SET', KEYS[1], 'revoked', 'KEEPTTL')
    return 0
end
redis.call('SET', KEYS[1], ARGV[2], 'KEEPTTL')
return 1
"""


class RedisHostedGrantStore:
    def __init__(self, cache: RedisCache, master_key: str) -> None:
        self._cache: Final = cache
        self._gateway: Final = hashlib.sha256(master_key.encode()).hexdigest()

    def _key(self, grant_id: str) -> str:
        return self._cache.check_and_fix_namespace(f"hosted_session:{self._gateway}:{grant_id}")

    async def read(self, grant_id: str) -> str | None:
        value: Final = await self._cache.async_eval("return redis.call('GET', KEYS[1])", 1, self._key(grant_id))
        return value.decode() if isinstance(value, bytes) else value if isinstance(value, str) else None

    async def advance(self, grant_id: str, previous: str | None, refresh_jti: str, ttl: int) -> bool:
        result: Final = await self._cache.async_eval(
            _ADVANCE_GRANT, 1, self._key(grant_id), previous or "", refresh_jti, ttl
        )
        return result == 1

    async def revoke(self, grant_id: str) -> None:
        await self._cache.async_eval(
            "return redis.call('SET', KEYS[1], 'revoked', 'XX', 'KEEPTTL')", 1, self._key(grant_id)
        )


class LoadHostedUser(Protocol):
    async def __call__(self, user_id: str, team_id: str | None, /) -> UserAPIKeyAuth | HostedFailure: ...


class HostedProxyAuth:
    def __init__(self, store: HostedGrantStore, load_user: LoadHostedUser, now: datetime) -> None:
        self._store: Final = store
        self._load_user: Final = load_user
        self._now: Final = int(now.timestamp())

    async def validate(
        self, principal: SessionPrincipal, resource: str, *, require_active: bool = True, refresh_jti: str | None = None
    ) -> UserAPIKeyAuth | HostedFailure:
        grant: Final = principal.app_grant
        if grant is None or grant.resource != resource or grant.expires_at <= self._now:
            return HostedFailure()
        if not hosted_redirect_is_allowed(grant.redirect_uri):
            return HostedFailure()
        if require_active:
            try:
                current: Final = await self._store.read(grant.grant_id)
            except Exception:  # noqa: BLE001  # shared authority errors must deny access
                return _unavailable()
            if current in (None, "revoked") or (refresh_jti is not None and current != refresh_jti):
                return HostedFailure()
        identity: Final = await self._load_user(principal.user_id, principal.team_id)
        if isinstance(identity, HostedFailure):
            return identity
        if identity.user_role != LitellmUserRoles.PROXY_ADMIN:
            return HostedFailure(
                error="insufficient_scope", description="application access requires a current proxy admin"
            )
        return identity

    async def advance(
        self, principal: SessionPrincipal, previous_refresh_jti: str | None, refresh_jti: str
    ) -> HostedFailure | None:
        grant: Final = principal.app_grant
        if grant is None or grant.expires_at <= self._now:
            return HostedFailure()
        try:
            accepted: Final = await self._store.advance(
                grant.grant_id, previous_refresh_jti, refresh_jti, grant.expires_at - self._now
            )
        except Exception:  # noqa: BLE001  # never issue a credential whose grant transition was not recorded
            return _unavailable()
        return None if accepted else HostedFailure()

    async def revoke(self, principal: SessionPrincipal) -> HostedFailure | None:
        if principal.app_grant is None:
            return HostedFailure()
        try:
            await self._store.revoke(principal.app_grant.grant_id)
        except Exception:  # noqa: BLE001  # report uncertain revocation so the client can retry
            return _unavailable()
        return None


class _HostedAuthSettings(BaseModel):
    model_config = ConfigDict(frozen=True)
    general_settings: Mapping[str, object]
    user_custom_auth: object


class _UserModelBudget(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)
    model_max_budget: Mapping[str, object] | None


async def load_hosted_user(user_id: str, team_id: str | None) -> UserAPIKeyAuth | HostedFailure:
    from litellm.proxy import proxy_server  # noqa: PLC0415  # startup-owned dependencies
    from litellm.proxy._experimental.mcp_server.bridge_token_flow import (
        active_user_record,  # noqa: PLC0415  # shared offboarding policy, proxy import cycle
    )
    from litellm.proxy.auth.auth_checks import (  # noqa: PLC0415  # proxy import cycle
        effective_user_role,
        get_team_membership,
        get_team_object,
        get_user_object,
    )
    from litellm.proxy.auth.resolvers.grants import (  # noqa: PLC0415  # shared live identity resolution
        GrantResolver,
        LookupDegraded,
        ResolvedGrants,
        UserLookup,
        user_models,
    )
    from litellm.proxy.auth.team_grants import team_grants  # noqa: PLC0415  # shared team permission projection
    from litellm.proxy.management_endpoints.ui_sso import (
        fetch_cli_sso_team_details,  # noqa: PLC0415  # existing live team selection rules
    )
    from litellm.proxy.proxy_server import (  # noqa: PLC0415  # startup-owned dependencies
        prisma_client,
        user_api_key_cache,
    )

    settings: Final = _HostedAuthSettings.model_validate(vars(proxy_server))
    if (
        settings.user_custom_auth is not None
        or settings.general_settings.get("enable_oauth2_auth") is True
        or settings.general_settings.get("enable_oauth2_proxy_auth") is True
    ):
        return HostedFailure(
            description="hosted application authorization is disabled while custom or external auth is configured"
        )
    if prisma_client is None:
        return _unavailable()
    resolved: Final = await GrantResolver(
        prisma_client,
        user_api_key_cache,
        load_user=partial(get_user_object, check_db_only=True),
        load_team=partial(get_team_object, check_db_only=True),
        load_membership=partial(get_team_membership, check_db_only=True),
    ).resolve(UserLookup(user_id=user_id), team_id)
    if isinstance(resolved, LookupDegraded):
        return _unavailable()
    if not isinstance(resolved, ResolvedGrants) or resolved.user_object is None:
        return HostedFailure()
    user: Final = active_user_record(resolved.user_object)
    if isinstance(user, str):
        return HostedFailure()
    if team_id is None and user.teams:
        teams: Final = await fetch_cli_sso_team_details(prisma_client, user.teams)
        if teams is None:
            return _unavailable()
        if any(team.team_id is not None for team in teams):
            return HostedFailure(description="select a current team and sign in again")
    team: Final = resolved.team_object
    if team is not None and team.blocked:
        return HostedFailure()
    return UserAPIKeyAuth.model_validate(
        MappingProxyType(
            {
                "user_id": user.user_id,
                "user_role": effective_user_role(user.user_role),
                "user_email": user.user_email,
                "user_tpm_limit": user.tpm_limit,
                "user_rpm_limit": user.rpm_limit,
                "user_spend": user.spend,
                "user_max_budget": user.max_budget,
                "user_model_max_budget": _UserModelBudget.model_validate(user).model_max_budget,
                "team_id": team_id,
                "org_id": team.organization_id if team is not None else user.organization_id,
                "models": user_models(user) if team is None else (),
                **team_grants(team, resolved.team_membership, user.user_id),
            }
        )
    )


def hosted_proxy_auth() -> HostedProxyAuth | HostedFailure:
    from litellm.proxy.proxy_server import master_key, redis_usage_cache  # noqa: PLC0415  # startup-owned authority

    if redis_usage_cache is None or not isinstance(master_key, str) or not master_key:
        return _unavailable()
    return HostedProxyAuth(
        RedisHostedGrantStore(redis_usage_cache, master_key), load_hosted_user, datetime.now(timezone.utc)
    )


async def authenticate_hosted_request(request: Request, token: str, route: str) -> UserAPIKeyAuth:
    from litellm.proxy.proxy_server import master_key  # noqa: PLC0415  # startup-owned signing configuration

    keys: Final = active_session_signing_keys(master_key) if isinstance(master_key, str) else None
    if keys is None or isinstance(keys, SessionSigningConfigError):
        raise HTTPException(status_code=503, detail="application authorization is unavailable")
    opened: Final = open_session_token(token, keys, datetime.now(timezone.utc))
    if not isinstance(opened, OpenedSessionToken) or opened.principal.app_grant is None:
        raise HTTPException(status_code=401, detail="a valid application access token is required")
    service: Final = hosted_proxy_auth()
    result: Final = (
        service
        if isinstance(service, HostedFailure)
        else await service.validate(opened.principal, get_request_base_url(request))
    )
    if isinstance(result, HostedFailure):
        status: Final = (
            503 if result.error == "temporarily_unavailable" else 403 if result.error == "insufficient_scope" else 401
        )
        raise HTTPException(status_code=status, detail=result.description)
    if not RouteChecks.hosted_admin_route_allowed(route):
        raise HTTPException(status_code=403, detail="this application grant does not permit this route")
    digest: Final = hash_token(token)
    return result.model_copy(update=MappingProxyType({"token": digest, "api_key": digest}))
