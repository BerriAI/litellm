"""Tests for custom-role RBAC: litellm.auth.rbac and its enforcement seams."""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from litellm.auth.rbac import RBACEngine, get_rbac_engine
from litellm.auth.route_checks import RouteChecks
from litellm.proxy._types import (
    LiteLLM_UserTable,
    UserAPIKeyAuth,
    validate_assignable_user_role,
)

POLICY = {
    "roles": [
        {"name": "base-reader", "allowed_routes": ["info_routes"]},
        {
            "name": "data-scientist",
            "allowed_routes": ["llm_api_routes", "/key/generate"],
            "inherits": ["base-reader"],
        },
        {"name": "ops", "allowed_routes": ["*"]},
        {"name": "loop-a", "allowed_routes": ["/a"], "inherits": ["loop-b"]},
        {"name": "loop-b", "allowed_routes": ["/b"], "inherits": ["loop-a"]},
    ]
}


@pytest.fixture
def engine():
    return RBACEngine.from_config(POLICY)


class TestRBACEngine:
    def test_governed_role_detection(self, engine):
        assert engine.is_governed_role("data-scientist") is True
        assert engine.is_governed_role("internal_user") is False
        assert engine.is_governed_role(None) is False
        assert engine.is_governed_role("") is False

    def test_route_group_name_grants_member_routes(self, engine):
        # llm_api_routes grants chat/completions; info_routes (inherited) grants /user/info
        assert engine.is_route_allowed("data-scientist", "/chat/completions") is True
        assert engine.is_route_allowed("data-scientist", "/v1/chat/completions") is True
        assert engine.is_route_allowed("data-scientist", "/user/info") is True

    def test_explicit_route_grant(self, engine):
        assert engine.is_route_allowed("data-scientist", "/key/generate") is True

    def test_default_deny_for_ungranted_route(self, engine):
        # /user/new is never granted to data-scientist -> denied
        assert engine.is_route_allowed("data-scientist", "/user/new") is False
        assert engine.is_route_allowed("data-scientist", "/key/delete") is False

    def test_inheritance_is_transitive_and_one_directional(self, engine):
        # base-reader does NOT inherit data-scientist's grants
        assert engine.is_route_allowed("base-reader", "/user/info") is True
        assert engine.is_route_allowed("base-reader", "/chat/completions") is False

    def test_wildcard_grants_everything(self, engine):
        assert engine.is_route_allowed("ops", "/anything/at/all") is True
        assert engine.is_route_allowed("ops", "/user/new") is True

    def test_inheritance_cycle_is_safe(self, engine):
        # loop-a <-> loop-b cycle must not hang and must union both grants
        assert engine.is_route_allowed("loop-a", "/a") is True
        assert engine.is_route_allowed("loop-a", "/b") is True
        assert engine.is_route_allowed("loop-b", "/a") is True

    def test_unknown_inherit_is_ignored_default_deny(self):
        e = RBACEngine.from_config(
            {"roles": [{"name": "r", "allowed_routes": ["/x"], "inherits": ["ghost"]}]}
        )
        assert e.is_route_allowed("r", "/x") is True
        assert e.is_route_allowed("r", "/y") is False


class TestEngineCache:
    def test_same_config_returns_same_engine(self):
        e1 = get_rbac_engine(POLICY)
        e2 = get_rbac_engine(dict(POLICY))
        assert e1 is e2

    def test_changed_config_rebuilds(self):
        e1 = get_rbac_engine({"roles": [{"name": "a", "allowed_routes": ["/a"]}]})
        e2 = get_rbac_engine({"roles": [{"name": "b", "allowed_routes": ["/b"]}]})
        assert e1 is not e2
        assert e2.is_route_allowed("b", "/b") is True

    def test_empty_config_returns_none(self):
        assert get_rbac_engine(None) is None
        assert get_rbac_engine({}) is None


@pytest.fixture
def rbac_general_settings(monkeypatch):
    from litellm.proxy import proxy_server

    monkeypatch.setitem(proxy_server.general_settings, "rbac", POLICY)
    yield
    proxy_server.general_settings.pop("rbac", None)


def _call_route_check(user_role, route):
    RouteChecks.non_proxy_admin_allowed_routes_check(
        user_obj=LiteLLM_UserTable(user_id="u1", user_role=user_role),
        _user_role=None,
        route=route,
        request=MagicMock(),
        valid_token=UserAPIKeyAuth(api_key="sk-test", user_role=user_role),
        request_data={},
    )


class TestRBACEnforcementWiring:
    def test_governed_role_allowed_route_passes(self, rbac_general_settings):
        # does not raise
        _call_route_check("data-scientist", "/chat/completions")
        _call_route_check("data-scientist", "/key/generate")

    def test_governed_role_denied_route_raises_403(self, rbac_general_settings):
        with pytest.raises(HTTPException) as exc:
            _call_route_check("data-scientist", "/user/new")
        assert exc.value.status_code == 403
        assert "data-scientist" in exc.value.detail

    def test_without_policy_custom_role_is_not_governed(self):
        # No rbac in general_settings -> custom role falls through to the built-in
        # chain, which denies management writes for a non-admin custom role.
        with pytest.raises((HTTPException, Exception)):
            _call_route_check("data-scientist", "/user/new")


class TestAssignableUserRoleValidator:
    def test_builtin_assignable_role_accepted(self):
        assert validate_assignable_user_role("internal_user") is not None

    def test_non_assignable_builtin_rejected(self):
        with pytest.raises(ValueError, match="cannot be assigned"):
            validate_assignable_user_role("org_admin")

    def test_custom_role_rejected_without_policy(self):
        with pytest.raises(ValueError, match="Invalid user_role"):
            validate_assignable_user_role("data-scientist")

    def test_custom_role_accepted_with_policy(self, rbac_general_settings):
        assert validate_assignable_user_role("data-scientist") == "data-scientist"

    def test_typo_role_rejected_with_policy(self, rbac_general_settings):
        with pytest.raises(ValueError, match="Invalid user_role"):
            validate_assignable_user_role("data-scientistt")


class TestNewUserRequestRoleField:
    def test_new_user_request_accepts_custom_role_with_policy(
        self, rbac_general_settings
    ):
        from litellm.proxy._types import NewUserRequest

        req = NewUserRequest(user_role="data-scientist")
        assert req.user_role == "data-scientist"

    def test_new_user_request_rejects_custom_role_without_policy(self):
        from pydantic import ValidationError

        from litellm.proxy._types import NewUserRequest

        with pytest.raises(ValidationError):
            NewUserRequest(user_role="data-scientist")
