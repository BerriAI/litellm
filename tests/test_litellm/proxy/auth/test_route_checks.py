import asyncio
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import pytest
from fastapi import HTTPException, Request

from litellm.proxy._types import LiteLLM_UserTable, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.route_checks import RouteChecks


def test_non_admin_config_update_route_rejected():
    """Test that non-admin users are rejected when trying to call /config/update"""

    # Create a non-admin user object
    user_obj = LiteLLM_UserTable(
        user_id="test_user",
        user_email="test@example.com",
        user_role=LitellmUserRoles.INTERNAL_USER.value,  # Non-admin role
    )

    # Create a non-admin user API key auth
    valid_token = UserAPIKeyAuth(
        user_id="test_user",
        user_role=LitellmUserRoles.INTERNAL_USER.value,  # Non-admin role
    )

    # Create a mock request
    request = MagicMock(spec=Request)
    request.query_params = {}

    # Test that calling /config/update route raises HTTPException with 403 status
    with pytest.raises(Exception) as exc_info:
        RouteChecks.non_proxy_admin_allowed_routes_check(
            user_obj=user_obj,
            _user_role=LitellmUserRoles.INTERNAL_USER.value,
            route="/config/update",
            request=request,
            valid_token=valid_token,
            request_data={},
        )

    # Verify the exception is raised with the correct message
    assert (
        "Only proxy admin can be used to generate, delete, update info for new keys/users/teams"
        in str(exc_info.value)
    )
    assert "Route=/config/update" in str(exc_info.value)
    assert "Your role=internal_user" in str(exc_info.value)


def test_proxy_admin_viewer_config_update_route_rejected():
    """Test that proxy admin viewer users are rejected when trying to call /config/update"""

    # Create a proxy admin viewer user object (read-only admin)
    user_obj = LiteLLM_UserTable(
        user_id="viewer_user",
        user_email="viewer@example.com",
        user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value,
    )

    # Create a proxy admin viewer user API key auth
    valid_token = UserAPIKeyAuth(
        user_id="viewer_user",
        user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value,
    )

    # Create a mock request
    request = MagicMock(spec=Request)
    request.query_params = {}

    # Test that calling /config/update route raises HTTPException with 403 status
    with pytest.raises(HTTPException) as exc_info:
        RouteChecks.non_proxy_admin_allowed_routes_check(
            user_obj=user_obj,
            _user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value,
            route="/config/update",
            request=request,
            valid_token=valid_token,
            request_data={},
        )

    # Verify the exception is HTTPException with 403 status
    assert exc_info.value.status_code == 403
    assert "user not allowed to access this route" in str(exc_info.value.detail)
    assert "role= proxy_admin_viewer" in str(exc_info.value.detail)


def test_virtual_key_allowed_routes_with_litellm_routes_member_name_allowed():
    """Test that virtual key is allowed to call routes when allowed_routes contains LiteLLMRoutes member name"""

    # Create a UserAPIKeyAuth with allowed_routes containing a LiteLLMRoutes member name
    valid_token = UserAPIKeyAuth(
        user_id="test_user",
        allowed_routes=["openai_routes"],  # This is a member name in LiteLLMRoutes enum
    )

    # Test that a route from the openai_routes group is allowed
    result = RouteChecks.is_virtual_key_allowed_to_call_route(
        route="/chat/completions",  # This is in LiteLLMRoutes.openai_routes.value
        valid_token=valid_token,
    )

    assert result is True


def test_virtual_key_allowed_routes_with_litellm_routes_member_name_denied():
    """Test that virtual key is denied when route is not in the allowed LiteLLMRoutes group"""

    # Create a UserAPIKeyAuth with allowed_routes containing a LiteLLMRoutes member name
    valid_token = UserAPIKeyAuth(
        user_id="test_user",
        allowed_routes=["info_routes"],  # This is a member name in LiteLLMRoutes enum
    )

    # Test that a route NOT in the info_routes group raises an exception
    with pytest.raises(Exception) as exc_info:
        RouteChecks.is_virtual_key_allowed_to_call_route(
            route="/chat/completions",  # This is NOT in LiteLLMRoutes.info_routes.value
            valid_token=valid_token,
        )

    # Verify the exception message
    assert "Virtual key is not allowed to call this route" in str(exc_info.value)
    assert "Only allowed to call routes: ['info_routes']" in str(exc_info.value)
    assert "Tried to call route: /chat/completions" in str(exc_info.value)

@pytest.mark.parametrize("route", [
    "/anthropic/v1/messages",
    "/anthropic/v1/count_tokens",
    "/gemini/v1/models",
    "/gemini/countTokens",
])
def test_virtual_key_llm_api_route_includes_passthrough_prefix(route):
    """
    Virtual key with llm_api_routes should allow passthrough routes like /anthropic/v1/messages
    
    Relevant issue: https://github.com/BerriAI/litellm/issues/14017
    """

    valid_token = UserAPIKeyAuth(
        user_id="test_user", allowed_routes=["llm_api_routes"]
    )

    result = RouteChecks.is_virtual_key_allowed_to_call_route(
        route=route, valid_token=valid_token
    )

    assert result is True


def test_virtual_key_allowed_routes_with_multiple_litellm_routes_member_names():
    """Test that virtual key works with multiple LiteLLMRoutes member names in allowed_routes"""

    # Create a UserAPIKeyAuth with multiple LiteLLMRoutes member names
    valid_token = UserAPIKeyAuth(
        user_id="test_user", allowed_routes=["openai_routes", "info_routes"]
    )

    # Test that routes from both groups are allowed
    result1 = RouteChecks.is_virtual_key_allowed_to_call_route(
        route="/chat/completions", valid_token=valid_token  # This is in openai_routes
    )

    result2 = RouteChecks.is_virtual_key_allowed_to_call_route(
        route="/user/info", valid_token=valid_token  # This is in info_routes
    )

    assert result1 is True
    assert result2 is True


def test_virtual_key_allowed_routes_with_mixed_member_names_and_explicit_routes():
    """Test that virtual key works with both LiteLLMRoutes member names and explicit routes"""

    # Create a UserAPIKeyAuth with both member names and explicit routes
    valid_token = UserAPIKeyAuth(
        user_id="test_user",
        allowed_routes=[
            "info_routes",
            "/custom/route",
        ],  # Mix of member name and explicit route
    )

    # Test that both info routes and explicit custom route are allowed
    result1 = RouteChecks.is_virtual_key_allowed_to_call_route(
        route="/user/info", valid_token=valid_token  # This is in info_routes
    )

    result2 = RouteChecks.is_virtual_key_allowed_to_call_route(
        route="/custom/route", valid_token=valid_token  # This is explicitly listed
    )

    assert result1 is True
    assert result2 is True


def test_virtual_key_allowed_routes_with_no_member_names_only_explicit():
    """Test that virtual key works when allowed_routes contains only explicit routes (no member names)"""

    # Create a UserAPIKeyAuth with only explicit routes (no LiteLLMRoutes member names)
    valid_token = UserAPIKeyAuth(
        user_id="test_user",
        allowed_routes=["/chat/completions", "/custom/route"],  # Only explicit routes
    )

    # Test that explicit routes are allowed
    result1 = RouteChecks.is_virtual_key_allowed_to_call_route(
        route="/chat/completions", valid_token=valid_token
    )

    result2 = RouteChecks.is_virtual_key_allowed_to_call_route(
        route="/custom/route", valid_token=valid_token
    )

    assert result1 is True
    assert result2 is True

    # Test that non-allowed route raises exception
    with pytest.raises(Exception) as exc_info:
        RouteChecks.is_virtual_key_allowed_to_call_route(
            route="/user/info", valid_token=valid_token  # Not in allowed routes
        )

    assert "Virtual key is not allowed to call this route" in str(exc_info.value)


def test_anthropic_count_tokens_route_is_llm_api_route():
    """Test that /v1/messages/count_tokens is recognized as an LLM API route for Anthropic"""
    
    # Test the core anthropic routes
    assert RouteChecks.is_llm_api_route("/v1/messages") is True
    assert RouteChecks.is_llm_api_route("/v1/messages/count_tokens") is True


def test_anthropic_count_tokens_route_accessible_to_internal_users():
    """Test that internal users can access the Anthropic count_tokens route"""
    
    # Test that the route is recognized as an LLM API route (which means it's accessible to internal users)
    # This is the core check that was failing in the original issue
    assert RouteChecks.is_llm_api_route("/v1/messages/count_tokens") is True
    
    # Also test that the regular messages route still works
    assert RouteChecks.is_llm_api_route("/v1/messages") is True
