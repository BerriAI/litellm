import pytest
from unittest.mock import MagicMock

from fastapi import HTTPException, Request

from litellm.proxy._types import (
    LiteLLM_UserTable,
    LiteLLMRoutes,
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.route_checks import RouteChecks


def test_info_route_identification():
    """Test that info routes are correctly identified"""
    for route in LiteLLMRoutes.info_routes.value:
        assert RouteChecks.is_info_route(route) is True

    # Non-info routes should return False
    assert RouteChecks.is_info_route("/chat/completions") is False
    assert RouteChecks.is_info_route("/key/generate") is False


def test_key_info_route_access():
    """Test access control for /key/info route"""
    # This route handles its own access control, so it should pass for any user
    user_obj = LiteLLM_UserTable(
        user_id="test_user",
        user_email="test@example.com",
        user_role=LitellmUserRoles.INTERNAL_USER,
    )
    valid_token = UserAPIKeyAuth(user_id="test_user")
    request = MagicMock(spec=Request)
    request.query_params = {}

    # Should not raise exception as /key/info handles its own logic
    RouteChecks.non_proxy_admin_allowed_routes_check(
        user_obj=user_obj,
        _user_role=LitellmUserRoles.INTERNAL_USER,
        route="/key/info",
        request=request,
        valid_token=valid_token,
        request_data={},
    )


def test_user_info_route_access():
    """Test access control for /user/info route"""
    user_obj = LiteLLM_UserTable(
        user_id="test_user",
        user_email="test@example.com",
        user_role=LitellmUserRoles.INTERNAL_USER,
    )
    valid_token = UserAPIKeyAuth(user_id="test_user")
    request = MagicMock(spec=Request)
    request.query_params = {"user_id": "test_user"}

    # Should not raise exception when user_id matches token's user_id
    RouteChecks.non_proxy_admin_allowed_routes_check(
        user_obj=user_obj,
        _user_role=LitellmUserRoles.INTERNAL_USER,
        route="/user/info",
        request=request,
        valid_token=valid_token,
        request_data={},
    )

    # Should raise exception when user_id does not match
    request.query_params = {"user_id": "different_user"}
    with pytest.raises(HTTPException) as exc_info:
        RouteChecks.non_proxy_admin_allowed_routes_check(
            user_obj=user_obj,
            _user_role=LitellmUserRoles.INTERNAL_USER,
            route="/user/info",
            request=request,
            valid_token=valid_token,
            request_data={},
        )
    assert exc_info.value.status_code == 403


def test_model_info_route_access():
    """Test access control for /model/info route"""
    user_obj = LiteLLM_UserTable(
        user_id="test_user",
        user_email="test@example.com",
        user_role=LitellmUserRoles.INTERNAL_USER,
    )
    valid_token = UserAPIKeyAuth(user_id="test_user")
    request = MagicMock(spec=Request)
    request.query_params = {}

    # Should not raise exception as /model/info is accessible to show user's models
    RouteChecks.non_proxy_admin_allowed_routes_check(
        user_obj=user_obj,
        _user_role=LitellmUserRoles.INTERNAL_USER,
        route="/model/info",
        request=request,
        valid_token=valid_token,
        request_data={},
    )


def test_team_info_route_access():
    """Test access control for /team/info route"""
    user_obj = LiteLLM_UserTable(
        user_id="test_user",
        user_email="test@example.com",
        user_role=LitellmUserRoles.INTERNAL_USER,
    )
    valid_token = UserAPIKeyAuth(user_id="test_user")
    request = MagicMock(spec=Request)
    request.query_params = {}

    # Should not raise exception as /team/info handles its own logic
    RouteChecks.non_proxy_admin_allowed_routes_check(
        user_obj=user_obj,
        _user_role=LitellmUserRoles.INTERNAL_USER,
        route="/team/info",
        request=request,
        valid_token=valid_token,
        request_data={},
    )


def test_v2_user_info_route_in_info_routes():
    """Test that /v2/user/info is in the info_routes list"""
    assert "/v2/user/info" in LiteLLMRoutes.info_routes.value


def test_v2_user_info_route_access():
    """Test access control for /v2/user/info route - handled by endpoint itself"""
    user_obj = LiteLLM_UserTable(
        user_id="test_user",
        user_email="test@example.com",
        user_role=LitellmUserRoles.INTERNAL_USER,
    )
    valid_token = UserAPIKeyAuth(user_id="test_user")
    request = MagicMock(spec=Request)
    request.query_params = {"user_id": "other_user"}

    # Should not raise exception as /v2/user/info handles its own RBAC logic in the handler
    RouteChecks.non_proxy_admin_allowed_routes_check(
        user_obj=user_obj,
        _user_role=LitellmUserRoles.INTERNAL_USER,
        route="/v2/user/info",
        request=request,
        valid_token=valid_token,
        request_data={},
    )


@pytest.mark.parametrize(
    "role",
    [
        LitellmUserRoles.INTERNAL_USER,
        LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
        LitellmUserRoles.TEAM,
        LitellmUserRoles.CUSTOMER,
        None,
    ],
)
def test_templated_info_route_is_as_reachable_as_a_plain_one(role):
    """
    A resolved path parameter must not make an info route less reachable than one without.

    `is_info_route` compared the request path against the route list by exact string, so
    `/key/{key_id}/budgets` never matched a real request and callers who could read /key/info
    were refused on the equivalent budgets route.
    """
    user_obj = LiteLLM_UserTable(
        user_id="test_user",
        user_email="test@example.com",
        user_role=role.value if role is not None else None,
        max_budget=None,
        spend=0.0,
        models=[],
    )

    for route in ("/key/info", "/key/some-resolved-key-hash/budgets", "/key/budgets"):
        RouteChecks.non_proxy_admin_allowed_routes_check(
            user_obj=user_obj,
            _user_role=role,
            route=route,
            request=MagicMock(spec=Request),
            valid_token=UserAPIKeyAuth(api_key="sk-test"),
            request_data={},
        )


def test_templated_route_matching_does_not_widen_unrelated_key_routes():
    """Pattern matching must stay scoped to the entries in the list, not every /key/<x>/<y> path."""
    assert RouteChecks.is_info_route("/key/some-hash/regenerate") is False
    assert RouteChecks.is_info_route("/key/some-hash/delete") is False
    assert RouteChecks.is_info_route("/key/some-hash/budgets") is True
