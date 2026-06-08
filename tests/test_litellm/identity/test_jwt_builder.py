import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath("../.."))

from litellm.identity import build_user_api_key_auth_from_jwt_result
from litellm.identity.jwt import parse_jwt_scopes
from litellm.identity.principal import JWTPrincipal
from litellm.proxy._types import (
    LiteLLM_TeamTableCachedObj,
    LiteLLM_UserTable,
    LitellmUserRoles,
    UserAPIKeyAuth,
)


def _team(team_id="t-jwt"):
    return LiteLLM_TeamTableCachedObj(
        team_id=team_id,
        team_alias=f"alias-{team_id}",
        tpm_limit=100,
        rpm_limit=10,
        models=["gpt-4o"],
        metadata={"env": "prod"},
    )


def _user(user_id="u-jwt", role="internal_user"):
    return LiteLLM_UserTable(
        user_id=user_id,
        user_email="jwt@litellm.io",
        user_role=role,
        tpm_limit=50,
        rpm_limit=5,
    )


def _membership(tpm=42, rpm=7):
    return SimpleNamespace(
        safe_get_team_member_tpm_limit=lambda: tpm,
        safe_get_team_member_rpm_limit=lambda: rpm,
    )


def _auth_builder_result(
    *,
    is_proxy_admin: bool = False,
    team=None,
    user=None,
    team_membership=None,
    jwt_claims=None,
):
    return {
        "is_proxy_admin": is_proxy_admin,
        "team_id": team.team_id if team is not None else None,
        "team_object": team,
        "user_id": user.user_id if user is not None else None,
        "user_object": user,
        "end_user_id": "eu-jwt",
        "org_id": "org-jwt",
        "team_membership": team_membership,
        "jwt_claims": jwt_claims
        or {"sub": "u-jwt", "iss": "https://idp", "scope": "read write"},
    }


def test_parse_jwt_scopes_handles_str_list_garbage_empty():
    assert parse_jwt_scopes({"scope": "read write"}) == ("read", "write")
    assert parse_jwt_scopes({"scope": ["read", "", "write"]}) == ("read", "write")
    assert parse_jwt_scopes({"scp": "admin"}) == ("admin",)
    assert parse_jwt_scopes({"scope": 123}) == ()
    assert parse_jwt_scopes({"scope": {"nested": "obj"}}) == ()
    assert parse_jwt_scopes({}) == ()
    assert parse_jwt_scopes({"scope": ""}) == ()
    assert parse_jwt_scopes({"scope": "   "}) == ()


def test_proxy_admin_path_marks_role_and_drops_membership_fields():
    result = _auth_builder_result(is_proxy_admin=True, team=_team(), user=_user())
    uak = build_user_api_key_auth_from_jwt_result(
        result=result, parent_otel_span=None, is_proxy_admin=True
    )
    assert isinstance(uak, UserAPIKeyAuth)
    assert uak.user_role == LitellmUserRoles.PROXY_ADMIN
    assert uak.team_id == "t-jwt"
    assert uak.team_alias == "alias-t-jwt"
    assert uak.team_tpm_limit == 100
    assert uak.team_metadata == {"env": "prod"}
    assert uak.team_member_rpm_limit is None
    assert uak.team_member_tpm_limit is None
    assert uak.user_tpm_limit is None
    assert uak.jwt_claims == result["jwt_claims"]


def test_regular_path_layers_team_user_membership_limits():
    membership = _membership(tpm=42, rpm=7)
    result = _auth_builder_result(
        team=_team(), user=_user(role="internal_user"), team_membership=membership
    )
    uak = build_user_api_key_auth_from_jwt_result(
        result=result, parent_otel_span=None, is_proxy_admin=False
    )
    assert uak.user_role == LitellmUserRoles.INTERNAL_USER
    assert uak.user_tpm_limit == 50
    assert uak.user_rpm_limit == 5
    assert uak.team_member_tpm_limit == 42
    assert uak.team_member_rpm_limit == 7
    assert uak.org_id == "org-jwt"
    assert uak.end_user_id == "eu-jwt"


def test_missing_team_object_defaults_team_fields_safely():
    result = _auth_builder_result(team=None, user=_user())
    uak = build_user_api_key_auth_from_jwt_result(
        result=result, parent_otel_span=None, is_proxy_admin=False
    )
    assert uak.team_alias is None
    assert uak.team_tpm_limit is None
    assert uak.team_models == []
    assert uak.team_metadata is None


def test_missing_user_object_defaults_user_role_to_internal():
    result = _auth_builder_result(team=_team(), user=None)
    uak = build_user_api_key_auth_from_jwt_result(
        result=result, parent_otel_span=None, is_proxy_admin=False
    )
    assert uak.user_role == LitellmUserRoles.INTERNAL_USER
    assert uak.user_tpm_limit is None


def test_adapter_roundtrip_produces_jwt_principal():
    result = _auth_builder_result(team=_team(), user=_user())
    uak = build_user_api_key_auth_from_jwt_result(
        result=result, parent_otel_span=None, is_proxy_admin=False
    )
    ctx = uak.to_identity_context()
    assert isinstance(ctx.principal, JWTPrincipal)
    assert ctx.principal.sub == "u-jwt"
    assert ctx.principal.iss == "https://idp"
    assert ctx.principal.scopes == ("read", "write")
    assert ctx.principal.mapped_user_id == "u-jwt"
    assert ctx.principal.mapped_team_id == "t-jwt"
    assert ctx.principal.mapped_org_id == "org-jwt"


def test_team_object_permission_propagates_when_present():
    team = _team()
    team.object_permission = SimpleNamespace(object_permission_id="perm-1")
    result = _auth_builder_result(team=team, user=_user())
    uak = build_user_api_key_auth_from_jwt_result(
        result=result, parent_otel_span=None, is_proxy_admin=False
    )
    assert uak.team_object_permission is not None
    assert uak.team_object_permission.object_permission_id == "perm-1"
