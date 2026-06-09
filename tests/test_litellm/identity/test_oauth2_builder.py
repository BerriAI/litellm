import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.identity import build_user_api_key_auth_from_oauth2_response
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth


def test_default_field_names_extract_from_introspection_response():
    response = {"sub": "u-1", "role": "internal_user", "team_id": "t-1"}
    uak = build_user_api_key_auth_from_oauth2_response(
        token="opaque-token", response_data=response
    )
    assert isinstance(uak, UserAPIKeyAuth)
    assert uak.user_id == "u-1"
    assert uak.user_role == "internal_user"
    assert uak.team_id == "t-1"


def test_custom_field_names_override_defaults():
    response = {
        "preferred_username": "alice",
        "groups": "proxy_admin",
        "tenant_id": "tenant-9",
    }
    uak = build_user_api_key_auth_from_oauth2_response(
        token="t",
        response_data=response,
        user_id_field_name="preferred_username",
        user_role_field_name="groups",
        user_team_id_field_name="tenant_id",
    )
    assert uak.user_id == "alice"
    assert uak.user_role == "proxy_admin"
    assert uak.team_id == "tenant-9"


def test_missing_fields_default_to_none():
    uak = build_user_api_key_auth_from_oauth2_response(token="t", response_data={})
    assert uak.user_id is None
    assert uak.user_role is None
    assert uak.team_id is None


def test_unknown_idp_role_rejected_fail_closed():
    with pytest.raises(ValueError, match="Invalid OAuth2 role"):
        build_user_api_key_auth_from_oauth2_response(
            token="t", response_data={"sub": "u", "role": "definitely-not-a-role"}
        )


def test_known_idp_role_passes_through():
    uak = build_user_api_key_auth_from_oauth2_response(
        token="t", response_data={"sub": "u", "role": "proxy_admin"}
    )
    assert uak.user_role == LitellmUserRoles.PROXY_ADMIN


def test_missing_role_field_stays_none():
    uak = build_user_api_key_auth_from_oauth2_response(
        token="t", response_data={"sub": "u"}
    )
    assert uak.user_role is None


def test_unknown_idp_role_uses_env_fallback_when_set(monkeypatch):
    monkeypatch.setenv("LITELLM_OAUTH2_UNKNOWN_ROLE_DEFAULT", "internal_user")
    uak = build_user_api_key_auth_from_oauth2_response(
        token="t", response_data={"sub": "u", "role": "custom-idp-role"}
    )
    assert uak.user_role == LitellmUserRoles.INTERNAL_USER


def test_unknown_idp_role_fallback_with_invalid_env_value_fails(monkeypatch):
    monkeypatch.setenv("LITELLM_OAUTH2_UNKNOWN_ROLE_DEFAULT", "not-a-real-role")
    with pytest.raises(ValueError):
        build_user_api_key_auth_from_oauth2_response(
            token="t", response_data={"sub": "u", "role": "custom-idp-role"}
        )


def test_known_idp_role_ignores_env_fallback(monkeypatch):
    monkeypatch.setenv("LITELLM_OAUTH2_UNKNOWN_ROLE_DEFAULT", "internal_user")
    uak = build_user_api_key_auth_from_oauth2_response(
        token="t", response_data={"sub": "u", "role": "proxy_admin"}
    )
    assert uak.user_role == LitellmUserRoles.PROXY_ADMIN


def test_token_is_hashed_into_token_field():
    """The api_key is hashed by the UserAPIKeyAuth validator; the
    OAuth2 builder must not bypass that path."""
    uak = build_user_api_key_auth_from_oauth2_response(
        token="sk-oauth2-test", response_data={"sub": "u"}
    )
    assert uak.token is not None
    assert uak.token != "sk-oauth2-test"
