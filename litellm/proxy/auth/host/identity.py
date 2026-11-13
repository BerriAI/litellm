"""Turns an authenticated `RustAuthOutcome` into the `UserAPIKeyAuth` the proxy hands to its routes."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

from typing_extensions import assert_never

import litellm
from litellm.constants import LITELLM_PROXY_MASTER_KEY_ALIAS
from litellm.models.user import LiteLLM_UserTable
from litellm.proxy.auth.auth_checks import cache_key_object
from litellm.proxy.auth.host.facts import (
    EndUserParams,
    EndUserRow,
    with_end_user,
    with_owner,
    with_project,
    with_team,
)
from litellm.proxy.auth.jwt import auth_from_result, jwt_result
from litellm.rust_bridge._native import (
    RustAuthOutcome,
    RustIdentity,
    RustKeyContext,
    RustTeamGrant,
)
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.types.proxy.auth.user_roles import LitellmUserRoles

if TYPE_CHECKING:
    from litellm.proxy.auth.host.python import PythonAuthHost


def auto_registered_key(
    key: UserAPIKeyAuth,
    claims: Mapping[str, object],
    end_user_id: str | None,
    user_email: str | None,
    org_id: str | None,
    user: LiteLLM_UserTable | None,
) -> UserAPIKeyAuth:
    """A key auto-registered for a JWT, as `_auto_register_jwt_mapping` and the builder stamped it.
    `org_id` is the JWT's, which the core passes only for a key created for it."""
    return key.model_copy(
        update={
            **({} if org_id is None else {"org_id": org_id}),
            "end_user_id": end_user_id,
            "api_key": key.token,
            "jwt_claims": dict(claims),
            "user_email": user_email,
            "user_model_max_budget": None if user is None else user.model_max_budget,
        }
    )


def _linked_key(
    key: UserAPIKeyAuth, claims: Mapping[str, object], end_user_id: str | None, user_email: str | None
) -> UserAPIKeyAuth:
    """An existing key a JWT maps onto, with what the JWT keeps once the key takes over."""
    return key.model_copy(update={"jwt_claims": dict(claims), "end_user_id": end_user_id, "user_email": user_email})


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
        case RustIdentity.MappedJwtKey(claims=claims, end_user_id=end_user_id, user_email=user_email, context=context):
            return await _virtual_key(
                host, _linked_key(_context_token(context), claims, end_user_id, user_email), context
            )
        case RustIdentity.AutoRegisteredJwtKey(
            claims=claims,
            end_user_id=end_user_id,
            user_email=user_email,
            org_id=org_id,
            user=policy_user,
            context=context,
        ):
            key_token: Final = _context_token(context)
            user: Final = context.owner if key_token.user_id is not None else policy_user
            return await _virtual_key(
                host, auto_registered_key(key_token, claims, end_user_id, user_email, org_id, user), context
            )
        case RustIdentity.FinishedJwt(token=token):
            return token
        case RustIdentity.Jwt() as jwt_identity:
            return auth_from_result(jwt_result(jwt_identity, host.api_key), host.parent_otel_span)
        case None:
            raise RuntimeError("rust auth projected an outcome with no identity")
        case unreachable:
            assert_never(unreachable)


def _cached_admin(host: PythonAuthHost, cached: UserAPIKeyAuth, end_user: EndUserRow) -> UserAPIKeyAuth:
    row: Final = end_user.end_user
    return with_end_user(cached, end_user.params).model_copy(
        update={
            "parent_otel_span": host.parent_otel_span,
            **({} if row is None else {"end_user_object_permission": row.object_permission}),
        }
    )


async def _master_key(host: PythonAuthHost, master_hash: str, end_user: EndUserRow) -> UserAPIKeyAuth:
    from litellm.proxy.auth.user_api_key_auth import (  # noqa: PLC0415  # user_api_key_auth imports this module
        return_user_api_key_auth_obj,
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
    authenticated: Final = with_end_user(token, end_user.params).model_copy(update={"via_virtual_key": True})
    _ = asyncio.create_task(
        cache_key_object(
            hashed_token=master_hash,
            user_api_key_obj=authenticated,
            user_api_key_cache=host.user_api_key_cache,
            proxy_logging_obj=host.proxy_logging_obj,
        )
    )
    return authenticated


async def _virtual_key(host: PythonAuthHost, token: UserAPIKeyAuth, context: RustKeyContext) -> UserAPIKeyAuth:
    from litellm.proxy.auth.user_api_key_auth import (  # noqa: PLC0415  # user_api_key_auth imports this module
        return_user_api_key_auth_obj,
    )

    if token.token is None:
        raise RuntimeError("rust auth accepted a key with no token")
    end_user: Final = context.end_user.end_user
    dumped: Final[Mapping[str, object]] = token.model_dump(
        exclude_none=True, exclude={"token"} if token.team_id is not None else {"token", "team_object_permission"}
    )
    end_user_params: Final[EndUserParams] = (
        context.end_user.params if end_user is not None or litellm.max_end_user_budget_id is not None else {}
    )
    valid_token_dict: Final = {
        **dumped,
        **({} if token.budget_throttle_pct is None else {"budget_throttle_pct": token.budget_throttle_pct}),
        **end_user_params,
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
    host.call().request.state.requires_key_model_check = True
    return authenticated
