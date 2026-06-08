"""JWT identity construction: ``auth_builder`` result -> ``UserAPIKeyAuth``.

Validation (signature, RBAC, scope, email-domain, ``custom_validate``) is
done upstream by ``JWTAuthManager.auth_builder``; this module only builds
the carrier.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth


def parse_jwt_scopes(claims: dict) -> Tuple[str, ...]:
    """Normalize the ``scope``/``scp`` claim into a tuple of scope strings.

    Accepts the space-delimited string form (``"read write"``) and the JSON
    array form (``["read", "write"]``); anything else yields ``()``.
    """
    scope_claim = claims.get("scope") or claims.get("scp") or ""
    if isinstance(scope_claim, list):
        return tuple(str(s) for s in scope_claim if s)
    if isinstance(scope_claim, str):
        return tuple(s for s in scope_claim.split(" ") if s)
    return ()


def build_user_api_key_auth_from_jwt_result(
    *,
    result: dict,
    parent_otel_span: Any = None,
    is_proxy_admin: bool,
) -> "UserAPIKeyAuth":
    """Build a ``UserAPIKeyAuth`` carrier from a JWT auth-builder result.

    ``result`` is the dict returned by
    ``JWTAuthManager.auth_builder``; its shape (``team_id``,
    ``team_object``, ``user_id``, ``user_object``, ``end_user_id``,
    ``org_id``, ``team_membership``, ``jwt_claims``) is the public
    contract between the validation layer and this construction layer.

    The ``is_proxy_admin`` flag is the caller's responsibility to pass
    because the auth_builder result reports it but the call site already
    branched on it; threading it through avoids re-checking.
    """
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

    team_id = result["team_id"]
    team_object = result["team_object"]
    user_id = result["user_id"]
    user_object = result["user_object"]
    end_user_id = result["end_user_id"]
    org_id = result["org_id"]
    team_membership = result.get("team_membership")
    jwt_claims = result.get("jwt_claims")

    team_alias = team_object.team_alias if team_object is not None else None
    team_tpm_limit = team_object.tpm_limit if team_object is not None else None
    team_rpm_limit = team_object.rpm_limit if team_object is not None else None
    team_models = team_object.models if team_object is not None else []
    team_metadata = team_object.metadata if team_object is not None else None
    team_object_permission = (
        team_object.object_permission if team_object is not None else None
    )

    if is_proxy_admin:
        return UserAPIKeyAuth(
            api_key=None,
            user_role=LitellmUserRoles.PROXY_ADMIN,
            user_id=user_id,
            team_id=team_id,
            team_alias=team_alias,
            team_tpm_limit=team_tpm_limit,
            team_rpm_limit=team_rpm_limit,
            team_models=team_models,
            team_metadata=team_metadata,
            org_id=org_id,
            end_user_id=end_user_id,
            parent_otel_span=parent_otel_span,
            jwt_claims=jwt_claims,
        )

    user_role = (
        LitellmUserRoles(user_object.user_role)
        if user_object is not None and user_object.user_role is not None
        else LitellmUserRoles.INTERNAL_USER
    )
    user_tpm_limit = user_object.tpm_limit if user_object is not None else None
    user_rpm_limit = user_object.rpm_limit if user_object is not None else None
    team_member_rpm_limit = (
        team_membership.safe_get_team_member_rpm_limit()
        if team_membership is not None
        else None
    )
    team_member_tpm_limit = (
        team_membership.safe_get_team_member_tpm_limit()
        if team_membership is not None
        else None
    )

    valid_token = UserAPIKeyAuth(
        api_key=None,
        team_id=team_id,
        team_alias=team_alias,
        team_tpm_limit=team_tpm_limit,
        team_rpm_limit=team_rpm_limit,
        team_models=team_models,
        user_role=user_role,
        user_id=user_id,
        org_id=org_id,
        parent_otel_span=parent_otel_span,
        end_user_id=end_user_id,
        user_tpm_limit=user_tpm_limit,
        user_rpm_limit=user_rpm_limit,
        team_member_rpm_limit=team_member_rpm_limit,
        team_member_tpm_limit=team_member_tpm_limit,
        team_metadata=team_metadata,
        jwt_claims=jwt_claims,
    )
    valid_token.team_object_permission = team_object_permission
    return valid_token
