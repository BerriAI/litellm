from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Final, Literal
from urllib.parse import urlparse

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, TypeAdapter

from litellm.proxy._experimental.mcp_server.bridge_token_flow import load_active_user_by_id
from litellm.proxy._experimental.mcp_server.oauth_utils import (
    TOKEN_NO_CACHE_HEADERS,
    canonical_resource_uri,
    get_request_base_url,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.session_credentials import (
    SessionSigningConfigError,
    active_session_signing_keys,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.session_token import (
    MintedSessionToken,
    OpenedSessionToken,
    SessionPrincipal,
    SessionSigningKeys,
    mint_session_refresh_token,
    mint_session_token,
    open_session_refresh_token,
    open_session_token,
)
from litellm.proxy._types import LiteLLM_UserTable, LitellmUserRoles, UserAPIKeyAuth, hash_token
from litellm.proxy.auth.auth_checks import get_object_permission, get_team_membership, get_team_object
from litellm.proxy.auth.resolvers.grants import user_models
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.auth.team_grants import team_grants

_FAMILY_SCRIPT: Final = """
local current = redis.call('GET', KEYS[1])
local revoked = '!revoked'
if ARGV[1] == 'revoke' then
  redis.call('SET', KEYS[1], revoked, 'EX', ARGV[4]); return 1
end
if ARGV[1] == 'active' then
  return current and current ~= revoked and (ARGV[2] == '' or current == ARGV[2]) and 1 or 0
end
if ARGV[1] == 'create' then
  if current then return 0 end
elseif not current or current == revoked then
  return 0
elseif current ~= ARGV[2] then
  redis.call('SET', KEYS[1], revoked, 'EX', ARGV[4]); return 0
end
redis.call('SET', KEYS[1], ARGV[3], 'EX', ARGV[4])
return 1
"""


class _AuthenticationPolicy(BaseModel):
    general_settings: Mapping[str, object] = Field(default_factory=dict)
    user_custom_auth: object = None


class _UserBudget(BaseModel):
    model_max_budget: Mapping[str, object] | None = None


def is_delegated_callback(redirect_uri: str) -> bool:
    try:
        parsed: Final = urlparse(redirect_uri)
    except ValueError:
        return False
    return (
        redirect_uri.isprintable()
        and parsed.scheme == "https"
        and bool(parsed.hostname)
        and not parsed.username
        and not parsed.password
        and not parsed.fragment
        and redirect_uri
        in tuple(value.strip() for value in os.getenv("LITELLM_OAUTH_ADMIN_REDIRECT_URIS", "").split(","))
    )


async def _family_transition(
    principal: SessionPrincipal,
    operation: Literal["create", "rotate", "active", "revoke"],
    previous: str = "",
    replacement: str = "",
) -> bool:
    from litellm.proxy.proxy_server import redis_usage_cache

    delegation: Final = principal.delegation
    if delegation is None or principal.audience != "proxy_api":
        return False
    ttl: Final = delegation.expires_at - int(datetime.now(timezone.utc).timestamp())
    if ttl <= 0:
        return False
    if redis_usage_cache is None:
        raise HTTPException(503, "Delegated OAuth requires shared Redis coordination")
    key: Final = "oauth:delegation:" + delegation.grant_id
    try:
        result: Final = TypeAdapter(int).validate_python(
            await redis_usage_cache.async_register_script(_FAMILY_SCRIPT)(
                keys=(key,), args=(operation, previous, replacement, ttl)
            ),
            strict=True,
        )
    except Exception as exc:
        raise HTTPException(503, "Delegated OAuth coordination is unavailable") from exc
    return result == 1


async def delegation_active(principal: SessionPrincipal, refresh_jti: str | None = None) -> bool:
    return (
        principal.delegation is not None
        and is_delegated_callback(principal.delegation.redirect_uri)
        and await _family_transition(principal, "active", refresh_jti or "")
    )


async def revoke_delegation(principal: SessionPrincipal) -> None:
    await _family_transition(principal, "revoke")


async def delegated_user(user_id: str) -> LiteLLM_UserTable:
    from litellm.proxy import proxy_server

    policy: Final = _AuthenticationPolicy.model_validate(vars(proxy_server))
    if policy.user_custom_auth is not None or any(
        policy.general_settings.get(flag) is True for flag in ("enable_oauth2_auth", "enable_oauth2_proxy_auth")
    ):
        raise HTTPException(503, "Delegated OAuth requires the gateway database authentication policy")
    user: Final = await load_active_user_by_id(user_id, source="database")
    if isinstance(user, str):
        raise HTTPException(503 if user in ("unavailable", "faulted", "unresolvable") else 401, "User is unavailable")
    if user.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(403, "Delegated admin access requires an active proxy admin")
    return user


async def delegated_identity(principal: SessionPrincipal) -> UserAPIKeyAuth:
    from litellm.proxy.proxy_server import prisma_client, user_api_key_cache

    try:
        user: Final = await delegated_user(principal.user_id)
        if (principal.team_id is None and user.teams) or (
            principal.team_id is not None and principal.team_id not in user.teams
        ):
            raise HTTPException(403, "The selected team is no longer available; authorize again")
        team: Final = (
            await get_team_object(principal.team_id, prisma_client, user_api_key_cache, check_db_only=True)
            if principal.team_id is not None
            else None
        )
        if team is not None and (
            team.blocked or not any(member.user_id == user.user_id for member in team.members_with_roles)
        ):
            raise HTTPException(403, "The selected team is blocked or its membership has been removed")
        membership: Final = (
            await get_team_membership(user.user_id, team.team_id, prisma_client, user_api_key_cache, check_db_only=True)
            if team is not None
            else None
        )
        permission: Final = (
            await get_object_permission(
                user.object_permission_id, prisma_client, user_api_key_cache, check_db_only=True
            )
            if user.object_permission_id is not None
            else None
        )
        identity: Final = UserAPIKeyAuth.model_validate(
            {
                **team_grants(team, membership, user.user_id),
                "user_id": user.user_id,
                "user_role": user.user_role,
                "user_email": user.user_email,
                "team_id": principal.team_id,
                "org_id": team.organization_id if team is not None else user.organization_id,
                "models": () if team is not None else user_models(user),
                "user_tpm_limit": user.tpm_limit,
                "user_rpm_limit": user.rpm_limit,
                "user_max_budget": user.max_budget,
                "user_model_max_budget": _UserBudget.model_validate(user.model_dump()).model_max_budget,
                "user_spend": user.spend,
                "object_permission": permission,
                "object_permission_id": user.object_permission_id,
            }
        )
        identity.requires_fresh_policy = True
        return identity
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(503, "Delegated user policy is unavailable") from exc


async def issue_delegated_tokens(
    principal: SessionPrincipal,
    keys: SessionSigningKeys,
    now: datetime,
    previous_refresh_jti: str | None = None,
) -> Response:
    try:
        if principal.delegation is None or not is_delegated_callback(principal.delegation.redirect_uri):
            raise HTTPException(400, "The delegated callback is not approved")
        await delegated_identity(principal)
        access: Final = mint_session_token(principal, keys, now)
        refresh: Final = mint_session_refresh_token(principal, keys, now)
        if not isinstance(access, MintedSessionToken) or not isinstance(refresh, MintedSessionToken):
            raise HTTPException(500, "Could not issue delegated tokens")
        opened: Final = open_session_refresh_token(refresh.token.get_secret_value(), keys, now)
        if not isinstance(opened, OpenedSessionToken):
            raise HTTPException(500, "Could not issue delegated tokens")
        if not await _family_transition(
            principal, "create" if previous_refresh_jti is None else "rotate", previous_refresh_jti or "", opened.jti
        ):
            raise HTTPException(400, "The grant is expired, revoked, or reused; authorize again")
        return JSONResponse(
            {
                "access_token": access.token.get_secret_value(),
                "token_type": "Bearer",
                "expires_in": int((access.expires_at - now).total_seconds()),
                "refresh_token": refresh.token.get_secret_value(),
                "scope": principal.delegation.scope,
            },
            headers=TOKEN_NO_CACHE_HEADERS,
        )
    except HTTPException as exc:
        return JSONResponse(
            {
                "error": "temporarily_unavailable" if exc.status_code >= 500 else "invalid_grant",
                "error_description": exc.detail,
            },
            status_code=503 if exc.status_code >= 500 else 400,
            headers=TOKEN_NO_CACHE_HEADERS,
        )


async def authenticate_delegated_request(request: Request, token: str, route: str) -> UserAPIKeyAuth:
    from litellm.proxy.proxy_server import master_key

    keys: Final = active_session_signing_keys(master_key) if master_key else None
    if keys is None or isinstance(keys, SessionSigningConfigError):
        raise HTTPException(503, "The gateway token signing configuration is unavailable")
    opened: Final = open_session_token(token, keys, datetime.now(timezone.utc))
    if not isinstance(opened, OpenedSessionToken) or opened.principal.delegation is None:
        raise HTTPException(401, "Invalid delegated access token")
    if opened.principal.delegation.resource != canonical_resource_uri(get_request_base_url(request)):
        raise HTTPException(401, "The delegated token belongs to a different gateway")
    if not RouteChecks.is_delegated_admin_route(route, request):
        raise HTTPException(403, "This route is outside the approved application scope")
    if not await delegation_active(opened.principal):
        raise HTTPException(401, "The application grant has expired or been revoked")
    identity: Final = await delegated_identity(opened.principal)
    identity.token = hash_token(token)
    identity.key_alias = "oauth-admin"
    return identity
