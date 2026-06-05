import pytest

from litellm.proxy.auth.v2.oauth2_introspection import (
    IntrospectionSettings,
    OAuth2IntrospectionError,
    parse_introspection_response,
)

SETTINGS = IntrospectionSettings(endpoint="https://idp/introspect")


def test_active_token_yields_identity():
    ident = parse_introspection_response({"active": True, "sub": "u1"}, SETTINGS)
    assert ident.user_id == "u1"


def test_inactive_token_is_rejected():
    with pytest.raises(OAuth2IntrospectionError):
        parse_introspection_response({"active": False, "sub": "u1"}, SETTINGS)


def test_missing_active_flag_is_rejected():
    # Absence of active=True must not be treated as valid.
    with pytest.raises(OAuth2IntrospectionError):
        parse_introspection_response({"sub": "u1"}, SETTINGS)


def test_missing_subject_is_rejected():
    with pytest.raises(OAuth2IntrospectionError):
        parse_introspection_response({"active": True}, SETTINGS)


def test_scope_string_maps_to_role():
    settings = IntrospectionSettings(
        endpoint="x", role_map={"litellm:admin": "proxy_admin"}
    )
    ident = parse_introspection_response(
        {"active": True, "sub": "u1", "scope": "openid litellm:admin"}, settings
    )
    assert ident.role == "proxy_admin"


def test_scope_list_maps_to_role():
    settings = IntrospectionSettings(endpoint="x", role_map={"a": "internal_user"})
    ident = parse_introspection_response(
        {"active": True, "sub": "u1", "scope": ["x", "a"]}, settings
    )
    assert ident.role == "internal_user"


def test_unmapped_scopes_yield_no_role():
    settings = IntrospectionSettings(endpoint="x", role_map={"a": "proxy_admin"})
    ident = parse_introspection_response(
        {"active": True, "sub": "u1", "scope": "b c"}, settings
    )
    assert ident.role is None


def test_team_claim_is_extracted():
    settings = IntrospectionSettings(endpoint="x", team_claim="team_id")
    ident = parse_introspection_response(
        {"active": True, "sub": "u1", "team_id": "eng"}, settings
    )
    assert ident.team_id == "eng"
