"""Bidirectional bridge between ``IdentityContext`` and ``UserAPIKeyAuth``.

The legacy Pydantic model stays the universal carrier. These two pure
functions let new code work in terms of ``IdentityContext`` without
forcing call sites to migrate today.

Invariants:
- ``identity_context_to_user_api_key_auth(uak.to_identity_context())``
  preserves every identity-relevant field on ``uak``.
- ``ApiKeyPrincipal.token_hash`` is treated as already-hashed; the
  Pydantic ``check_api_key`` validator does not re-hash it.
"""

from typing import TYPE_CHECKING

from litellm.constants import (
    LITELLM_INTERNAL_JOBS_SERVICE_ACCOUNT_NAME,
    LITTELM_CLI_SERVICE_ACCOUNT_NAME,
    LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME,
)
from litellm.identity.context import AuditInfo, ClientInfo, IdentityContext, RequestIds
from litellm.identity.principal import (
    AnonymousPrincipal,
    ApiKeyPrincipal,
    JWTPrincipal,
    Principal,
    ServiceAccountPrincipal,
)

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth

_SERVICE_ACCOUNT_NAMES = frozenset(
    {
        LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME,
        LITTELM_CLI_SERVICE_ACCOUNT_NAME,
        LITELLM_INTERNAL_JOBS_SERVICE_ACCOUNT_NAME,
    }
)


def _principal_from_uak(uak: "UserAPIKeyAuth") -> Principal:
    if uak.api_key in _SERVICE_ACCOUNT_NAMES or uak.key_alias in _SERVICE_ACCOUNT_NAMES:
        return ServiceAccountPrincipal(
            name=uak.api_key if uak.api_key in _SERVICE_ACCOUNT_NAMES else uak.key_alias  # type: ignore[arg-type]
        )

    if uak.jwt_claims:
        claims = uak.jwt_claims
        scope_claim = claims.get("scope") or claims.get("scp") or ""
        if isinstance(scope_claim, list):
            scopes = [str(s) for s in scope_claim if s]
        elif isinstance(scope_claim, str):
            scopes = [s for s in scope_claim.split(" ") if s]
        else:
            scopes = []
        return JWTPrincipal(
            sub=claims.get("sub"),
            iss=claims.get("iss"),
            aud=claims.get("aud"),
            scopes=scopes,
            claims=dict(claims),
            mapped_user_id=uak.user_id,
            mapped_team_id=uak.team_id,
            mapped_org_id=uak.org_id,
        )

    if uak.token:
        return ApiKeyPrincipal(
            token_hash=uak.token,
            key_alias=uak.key_alias,
            user_id=uak.user_id,
            team_id=uak.team_id,
            org_id=uak.org_id,
            project_id=uak.project_id,
            agent_id=uak.agent_id,
        )

    return AnonymousPrincipal()


def user_api_key_auth_to_identity_context(
    uak: "UserAPIKeyAuth",
) -> IdentityContext:
    principal = _principal_from_uak(uak)
    return IdentityContext(
        principal=principal,
        end_user_id=uak.end_user_id,
        tags=[],
        access_group_ids=list(uak.access_group_ids or []),
        request=RequestIds(),
        client=ClientInfo(),
        audit=AuditInfo(),
    )


def identity_context_to_user_api_key_auth(
    ctx: IdentityContext,
) -> "UserAPIKeyAuth":
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

    principal = ctx.principal
    kwargs: dict = {
        "end_user_id": ctx.end_user_id,
        "access_group_ids": list(ctx.access_group_ids) if ctx.access_group_ids else None,
    }

    if isinstance(principal, ApiKeyPrincipal):
        kwargs.update(
            {
                "token": principal.token_hash,
                "key_alias": principal.key_alias,
                "user_id": principal.user_id,
                "team_id": principal.team_id,
                "org_id": principal.org_id,
                "project_id": principal.project_id,
                "agent_id": principal.agent_id,
            }
        )
    elif isinstance(principal, JWTPrincipal):
        kwargs.update(
            {
                "jwt_claims": dict(principal.claims),
                "user_id": principal.mapped_user_id,
                "team_id": principal.mapped_team_id,
                "org_id": principal.mapped_org_id,
            }
        )
    elif isinstance(principal, ServiceAccountPrincipal):
        if principal.name == LITELLM_INTERNAL_JOBS_SERVICE_ACCOUNT_NAME:
            kwargs.update(
                {
                    "api_key": principal.name,
                    "team_id": "system",
                    "key_alias": principal.name,
                    "team_alias": "system",
                    "user_id": "system",
                    "user_role": LitellmUserRoles.PROXY_ADMIN,
                }
            )
        else:
            kwargs.update(
                {
                    "api_key": principal.name,
                    "team_id": principal.name,
                    "key_alias": principal.name,
                    "team_alias": principal.name,
                }
            )

    return UserAPIKeyAuth(**{k: v for k, v in kwargs.items() if v is not None})
