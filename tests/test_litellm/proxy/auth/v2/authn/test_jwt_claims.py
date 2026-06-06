import pytest

from litellm.proxy.auth.v2.authn.jwt_claims import (
    JWTClaimError,
    JWTSettings,
    extract_identity,
)

BASE = JWTSettings(jwks_uri="https://idp/jwks")


def test_extracts_user_id_from_default_sub_claim():
    ident = extract_identity({"sub": "u1"}, BASE)
    assert ident.user_id == "u1"
    assert ident.team_id is None
    assert ident.role is None


def test_custom_user_id_claim():
    settings = JWTSettings(jwks_uri="x", user_id_claim="oid")
    assert extract_identity({"oid": "abc"}, settings).user_id == "abc"


def test_team_and_role_are_mapped():
    settings = JWTSettings(
        jwks_uri="x",
        team_claim="team",
        role_claim="groups",
        role_map={"admins": "proxy_admin"},
    )
    ident = extract_identity({"sub": "u1", "team": "eng", "groups": "admins"}, settings)
    assert ident.team_id == "eng"
    assert ident.role == "proxy_admin"


def test_unmapped_role_value_yields_no_role():
    # An arbitrary IdP role string must not be trusted as a litellm role.
    settings = JWTSettings(
        jwks_uri="x", role_claim="role", role_map={"a": "proxy_admin"}
    )
    assert (
        extract_identity({"sub": "u1", "role": "totally-unknown"}, settings).role
        is None
    )


def test_missing_user_id_raises():
    with pytest.raises(JWTClaimError):
        extract_identity({"email": "x@y.z"}, BASE)


def test_non_string_ids_are_coerced():
    settings = JWTSettings(jwks_uri="x", team_claim="team")
    ident = extract_identity({"sub": 12345, "team": 67}, settings)
    assert ident.user_id == "12345"
    assert ident.team_id == "67"
