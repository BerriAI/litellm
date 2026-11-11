"""Turns an authenticated `RustAuthOutcome` into the `UserAPIKeyAuth` the proxy hands to its routes."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

import litellm
from litellm.constants import LITELLM_PROXY_MASTER_KEY_ALIAS
from litellm.models.user import LiteLLM_UserTable
from litellm.proxy._types import JWTAuthBuilderResult, ProxyErrorTypes, ProxyException
from litellm.proxy.auth.auth_checks import cache_key_object
from litellm.proxy.auth.auth_exception_handler import db_unavailable_identity
from litellm.proxy.auth.host.facts import (
    EndUserRow,
    with_owner,
    with_project,
    with_team,
)
from litellm.proxy.auth.team_grants import team_grants
from litellm.rust_bridge._native import (
    RustAuthOutcome,
    RustIdentity,
    RustKeyContext,
    RustTeamGrant,
)
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.types.proxy.auth.user_roles import LitellmUserRoles

if TYPE_CHECKING:
    from opentelemetry.trace import Span

    from litellm.proxy.auth.host.python import PythonAuthHost


def jwt_result(identity: RustIdentity.Jwt, token: str | None) -> JWTAuthBuilderResult:
    """The `JWTAuthBuilderResult` `authorize_jwt` returned for `token`, from the core's ids and rows."""
    end_user: Final = identity.end_user
    return JWTAuthBuilderResult(
        is_proxy_admin=identity.is_proxy_admin,
        team_id=identity.team_id,
        team_object=identity.team,
        user_id=identity.user_id,
        user_email=identity.user_email,
        user_object=identity.user,
        org_id=identity.org_id,
        org_object=identity.org,
        end_user_id=identity.end_user_id,
        end_user_object=None if end_user is None else end_user.end_user,
        token=token or "",
        team_membership=identity.membership,
        jwt_claims=dict(identity.claims),
        agent_id=identity.agent_id,
        managed_agent_context=identity.managed_agent,
    )


def auth_from_result(result: JWTAuthBuilderResult, parent_otel_span: Span | None = None) -> UserAPIKeyAuth:
    """Keep JWT identity and permission attribution identical across consumers."""
    user: Final = result["user_object"]
    admin: Final = result["is_proxy_admin"]
    auth: Final = UserAPIKeyAuth(
        api_key=None,
        user_role=(
            LitellmUserRoles.PROXY_ADMIN
            if admin
            else LitellmUserRoles.INTERNAL_USER
            if result.get("managed_agent_context") is not None
            else LitellmUserRoles(user.user_role)
            if user is not None and user.user_role is not None
            else LitellmUserRoles.INTERNAL_USER
        ),
        user_id=result["user_id"],
        user_email=result["user_email"],
        team_id=result["team_id"],
        org_id=result["org_id"],
        end_user_id=result["end_user_id"],
        parent_otel_span=parent_otel_span,
        jwt_claims=result["jwt_claims"],
        agent_id=result.get("agent_id"),
        user_tpm_limit=user.tpm_limit if user is not None and not admin else None,
        user_rpm_limit=user.rpm_limit if user is not None and not admin else None,
        user_model_max_budget=user.model_max_budget if user is not None and not admin else None,
        **team_grants(
            team_object=result["team_object"],
            team_membership=result.get("team_membership"),
            user_id=result["user_id"],
        ),
    )
    auth.managed_agent_context = result.get("managed_agent_context")
    auth._managed_delegation_verified = (  # pyright: ignore[reportPrivateUsage]  # JWT admission produces the one-shot proof consumed by managed authorization
        auth.managed_agent_context is not None and auth.managed_agent_context.mode == "delegated"
    )
    return auth


def _jwt_mapped_key(
    token: UserAPIKeyAuth,
    claims: Mapping[str, object],
    end_user_id: str | None,
    user_email: str | None,
    org_id: str | None,
    user: LiteLLM_UserTable | None,
) -> UserAPIKeyAuth:
    """A key auto-registered for a JWT, as `_auto_register_jwt_mapping` and the builder stamped it."""
    return token.model_copy(
        update={
            **({} if org_id is None else {"org_id": org_id}),
            "end_user_id": end_user_id,
            "api_key": token.token,
            "jwt_claims": dict(claims),
            "user_email": user_email,
            "user_model_max_budget": None if user is None else user.model_max_budget,
        }
    )


def finished_jwt_token(identity: RustIdentity, token: str | None, parent_otel_span: Span | None) -> UserAPIKeyAuth:
    match identity:
        case RustIdentity.Jwt() as jwt_identity:
            return auth_from_result(jwt_result(jwt_identity, token), parent_otel_span)
        case RustIdentity.AutoRegisteredJwtKey(
            claims=claims,
            end_user_id=end_user_id,
            user_email=user_email,
            minted=minted,
            org_id=org_id,
            user=user,
            key=key,
        ):
            return _jwt_mapped_key(key, claims, end_user_id, user_email, org_id if minted else None, user)
        case _:
            raise RuntimeError(f"rust auth finished a JWT with an identity this host does not know: {identity!r}")


def _with_team_grant(token: UserAPIKeyAuth, grant: RustTeamGrant) -> UserAPIKeyAuth:
    match grant:
        case RustTeamGrant.Loaded(team=team):
            return with_team(token, team.object_permission, team.metadata)
        case RustTeamGrant.KeyCopy(object_permission=object_permission):
            return with_team(token, object_permission, token.team_metadata)
        case _:
            return token


def _context_token(context: RustKeyContext) -> UserAPIKeyAuth:
    """The key with what its checks loaded: the owner's per-model budget, the team's values and the project's."""
    return with_project(_with_team_grant(with_owner(context.key, context.owner), context.team), context.project)


async def project(outcome: RustAuthOutcome, host: PythonAuthHost) -> UserAPIKeyAuth:
    """The one place an authenticated outcome becomes a UserAPIKeyAuth."""
    match outcome.identity:
        case RustIdentity.UserAuthRoute():
            return UserAPIKeyAuth()
        case RustIdentity.NoMasterKey(token=token):
            return UserAPIKeyAuth(
                api_key=token,
                user_role=LitellmUserRoles.INTERNAL_USER,
                parent_otel_span=host.parent_otel_span,
            )
        case RustIdentity.CachedAdmin(key=key, end_user=end_user):
            return _cached_admin(host, key, end_user)
        case RustIdentity.MasterKey(hash=master_hash, end_user=end_user):
            return await _master_key(host, master_hash, end_user)
        case RustIdentity.VirtualKey(context=context):
            return await _virtual_key(host, _context_token(context), context)
        case RustIdentity.MappedJwtKey(
            claims=claims,
            end_user_id=end_user_id,
            user_email=user_email,
            auto_registered=auto_registered,
            minted=minted,
            org_id=org_id,
            user=policy_user,
            context=context,
        ):
            key_token: Final = _context_token(context)
            user: Final = context.owner if key_token.user_id is not None else policy_user
            mapped: Final = (
                _jwt_mapped_key(key_token, claims, end_user_id, user_email, org_id if minted else None, user)
                if auto_registered
                else key_token.model_copy(
                    update={"jwt_claims": dict(claims), "end_user_id": end_user_id, "user_email": user_email}
                )
            )
            return await _virtual_key(host, mapped, context)
        case RustIdentity.FinishedJwt(token=token):
            return token
        case RustIdentity.Jwt() as jwt_identity:
            return auth_from_result(jwt_result(jwt_identity, host.api_key), host.parent_otel_span)
        case RustIdentity.DbUnavailable(message=message):
            error: Final = ProxyException(message=message, type=ProxyErrorTypes.no_db_connection, param=None, code=503)
            return db_unavailable_identity(error, host.route)
        case None:
            raise RuntimeError("rust auth projected an outcome with no identity")
        case unknown:
            raise RuntimeError(f"rust auth returned an identity this host does not know: {unknown!r}")


def _cached_admin(host: PythonAuthHost, cached: UserAPIKeyAuth, end_user: EndUserRow) -> UserAPIKeyAuth:
    from litellm.proxy.auth.user_api_key_auth import (  # noqa: PLC0415  # user_api_key_auth imports this module
        update_valid_token_with_end_user_params,
    )

    token: Final = update_valid_token_with_end_user_params(valid_token=cached, end_user_params=dict(end_user.params))
    token.parent_otel_span = host.parent_otel_span
    if end_user.end_user is not None:
        token.end_user_object_permission = end_user.end_user.object_permission
    return token


async def _master_key(host: PythonAuthHost, master_hash: str, end_user: EndUserRow) -> UserAPIKeyAuth:
    from litellm.proxy.auth.user_api_key_auth import (  # noqa: PLC0415  # user_api_key_auth imports this module
        return_user_api_key_auth_obj,
        update_valid_token_with_end_user_params,
    )

    token: Final = await return_user_api_key_auth_obj(
        user_obj=None,
        user_role=LitellmUserRoles.PROXY_ADMIN,
        api_key=LITELLM_PROXY_MASTER_KEY_ALIAS,
        parent_otel_span=host.parent_otel_span,
        valid_token_dict={**end_user.params, "user_id": host.litellm_proxy_admin_name},
        route=host.route,
        start_time=host.call().start_time,
    )
    _ = asyncio.create_task(
        cache_key_object(
            hashed_token=master_hash,
            user_api_key_obj=token,
            user_api_key_cache=host.user_api_key_cache,
            proxy_logging_obj=host.proxy_logging_obj,
        )
    )
    authenticated: Final = update_valid_token_with_end_user_params(
        valid_token=token, end_user_params=dict(end_user.params)
    )
    authenticated.via_virtual_key = True
    return authenticated


async def _virtual_key(host: PythonAuthHost, token: UserAPIKeyAuth, context: RustKeyContext) -> UserAPIKeyAuth:
    from litellm.proxy.auth.user_api_key_auth import (  # noqa: PLC0415  # user_api_key_auth imports this module
        return_user_api_key_auth_obj,
    )

    if token.token is None:
        raise RuntimeError("rust auth accepted a key with no token")
    end_user_row: Final[EndUserRow] = context.end_user
    end_user: Final = end_user_row.end_user
    dumped: Final[Mapping[str, object]] = token.model_dump(
        exclude_none=True, exclude={"token"} if token.team_id is not None else {"token", "team_object_permission"}
    )
    valid_token_dict: Final = {
        **dumped,
        **({} if token.budget_throttle_pct is None else {"budget_throttle_pct": token.budget_throttle_pct}),
        **(end_user_row.params if end_user is not None or litellm.max_end_user_budget_id is not None else {}),
        **({} if end_user is None else {"end_user_object_permission": end_user.object_permission}),
    }
    authenticated: Final = await return_user_api_key_auth_obj(
        user_obj=context.owner,
        api_key=token.token,
        parent_otel_span=host.parent_otel_span,
        valid_token_dict=valid_token_dict,
        route=host.route,
        start_time=host.call().start_time,
    )
    authenticated.via_virtual_key = True
    host.call().request.state.requires_key_model_check = (
        True  # rebind-ok: defers the key's model check to central grants
    )
    return authenticated
