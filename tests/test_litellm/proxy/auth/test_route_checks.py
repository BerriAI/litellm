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


def test_virtual_key_mcp_routes_allows_v1_mcp_server():
    """Regression test for #20325: allow virtual keys to list MCP servers."""

    valid_token = UserAPIKeyAuth(
        user_id="test_user",
        allowed_routes=["mcp_routes"],
    )

    result = RouteChecks.is_virtual_key_allowed_to_call_route(
        route="/v1/mcp/server",
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

    # Test that a route NOT in the info_routes group raises an HTTPException
    with pytest.raises(HTTPException) as exc_info:
        RouteChecks.is_virtual_key_allowed_to_call_route(
            route="/chat/completions",  # This is NOT in LiteLLMRoutes.info_routes.value
            valid_token=valid_token,
        )

    # Verify the exception has correct status and message
    assert exc_info.value.status_code == 403
    assert "Virtual key is not allowed to call this route" in str(exc_info.value.detail)
    assert "Only allowed to call routes: ['info_routes']" in str(exc_info.value.detail)
    assert "Tried to call route: /chat/completions" in str(exc_info.value.detail)


@pytest.mark.parametrize(
    "route",
    [
        "/anthropic/v1/messages",
        "/anthropic/v1/count_tokens",
        "/gemini/v1/models",
        "/gemini/countTokens",
    ],
)
def test_virtual_key_llm_api_route_includes_passthrough_prefix(route):
    """
    Virtual key with llm_api_routes should allow passthrough routes like /anthropic/v1/messages

    Relevant issue: https://github.com/BerriAI/litellm/issues/14017
    """

    valid_token = UserAPIKeyAuth(user_id="test_user", allowed_routes=["llm_api_routes"])

    result = RouteChecks.is_virtual_key_allowed_to_call_route(
        route=route, valid_token=valid_token
    )

    assert result is True


@pytest.mark.parametrize(
    "route",
    [
        "/v1beta/models/gemini-2.5-flash:countTokens",
        "/v1beta/models/gemini-2.0-flash:generateContent",
        "/v1beta/models/bedrock/claude-sonnet-3.7:generateContent",
        "/v1beta/models/gemini-1.5-pro:streamGenerateContent",
        "/models/gemini-2.5-flash:countTokens",
        "/models/gemini-2.0-flash:generateContent",
        "/models/bedrock/claude-sonnet-3.7:generateContent",
        "/models/gemini-1.5-pro:streamGenerateContent",
    ],
)
def test_virtual_key_llm_api_routes_allows_google_routes(route):
    """
    Test that virtual keys with llm_api_routes permission can access Google AI Studio routes.
    """

    valid_token = UserAPIKeyAuth(user_id="test_user", allowed_routes=["llm_api_routes"])

    result = RouteChecks.is_virtual_key_allowed_to_call_route(
        route=route, valid_token=valid_token
    )

    assert result is True


@pytest.mark.parametrize(
    "route",
    [
        "/v1beta/models/google-gemini-2-5-pro-code-reviewer-k8s:generateContent",
        "/v1beta/models/gemini-2.5-flash-exp:countTokens",
        "/v1beta/models/custom-model-name-123:streamGenerateContent",
        "/v1beta/models/bedrock/claude-sonnet-3.7:generateContent",
        "/models/google-gemini-2-5-pro-code-reviewer-k8s:generateContent",
        "/models/gemini-2.5-flash-exp:countTokens",
        "/models/custom-model-name-123:streamGenerateContent",
        "/models/bedrock/claude-sonnet-3.7:generateContent",
    ],
)
def test_google_routes_with_dynamic_model_names_recognized_as_llm_api_route(route):
    """
    Test that Google routes with dynamic model names (including custom names) are recognized as LLM API routes.

    This test verifies the fix for the issue where routes like:
    /v1beta/models/google-gemini-2-5-pro-code-reviewer-k8s:generateContent
    were incorrectly classified as "custom admin only route" instead of LLM API routes.

    The fix adds pattern matching for Google routes with placeholders like {model_name}.
    """

    # Test that the route is recognized as an LLM API route
    assert RouteChecks.is_llm_api_route(route) is True


def test_google_routes_with_dynamic_model_names_accessible_to_internal_users():
    """
    Test that internal users can access Google routes with dynamic model names.

    This ensures that routes like /v1beta/models/{model_name}:generateContent
    are properly accessible to internal users and not blocked as admin-only routes.
    """

    # Create an internal user object
    user_obj = LiteLLM_UserTable(
        user_id="test_user",
        user_email="test@example.com",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )

    # Create an internal user API key auth
    valid_token = UserAPIKeyAuth(
        user_id="test_user",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )

    # Create a mock request
    request = MagicMock(spec=Request)
    request.query_params = {}

    # Test that calling Google route with dynamic model name does NOT raise an exception
    try:
        RouteChecks.non_proxy_admin_allowed_routes_check(
            user_obj=user_obj,
            _user_role=LitellmUserRoles.INTERNAL_USER.value,
            route="/v1beta/models/google-gemini-2-5-pro-code-reviewer-k8s:generateContent",
            request=request,
            valid_token=valid_token,
            request_data={"contents": [{"parts": [{"text": "test"}]}]},
        )
        # If no exception is raised, the test passes
    except Exception as e:
        pytest.fail(
            f"Internal user should be able to access Google generateContent route. Got error: {str(e)}"
        )


def test_virtual_key_allowed_routes_with_multiple_litellm_routes_member_names():
    """Test that virtual key works with multiple LiteLLMRoutes member names in allowed_routes"""

    # Create a UserAPIKeyAuth with multiple LiteLLMRoutes member names
    valid_token = UserAPIKeyAuth(
        user_id="test_user", allowed_routes=["openai_routes", "info_routes"]
    )

    # Test that routes from both groups are allowed
    result1 = RouteChecks.is_virtual_key_allowed_to_call_route(
        route="/chat/completions",
        valid_token=valid_token,  # This is in openai_routes
    )

    result2 = RouteChecks.is_virtual_key_allowed_to_call_route(
        route="/user/info",
        valid_token=valid_token,  # This is in info_routes
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
        route="/user/info",
        valid_token=valid_token,  # This is in info_routes
    )

    result2 = RouteChecks.is_virtual_key_allowed_to_call_route(
        route="/custom/route",
        valid_token=valid_token,  # This is explicitly listed
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

    # Test that non-allowed route raises HTTPException
    with pytest.raises(HTTPException) as exc_info:
        RouteChecks.is_virtual_key_allowed_to_call_route(
            route="/user/info",
            valid_token=valid_token,  # Not in allowed routes
        )

    assert exc_info.value.status_code == 403
    assert "Virtual key is not allowed to call this route" in str(exc_info.value.detail)


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


def test_virtual_key_llm_api_routes_allows_registered_pass_through_endpoints():
    """
    Test that virtual keys with llm_api_routes permission can access registered pass-through endpoints.

    This tests the scenario where a pass-through endpoint is registered from the DB
    (e.g., /azure-assistant) and a virtual key with llm_api_routes permission should be able to access
    both the exact path and subpaths (e.g., /azure-assistant/openai/assistants).
    """

    # Mock the registered pass-through routes
    mock_registered_routes = {
        "test-uuid-1:exact:/azure-assistant": {
            "endpoint_id": "test-uuid-1",
            "path": "/azure-assistant",
            "type": "exact",
        },
        "test-uuid-2:subpath:/custom-endpoint": {
            "endpoint_id": "test-uuid-2",
            "path": "/custom-endpoint",
            "type": "subpath",
        },
    }

    with (
        patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints._registered_pass_through_routes",
            mock_registered_routes,
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_server_root_path",
            return_value="/",
        ),
    ):
        # Create a virtual key with llm_api_routes permission
        valid_token = UserAPIKeyAuth(
            user_id="test_user",
            allowed_routes=["llm_api_routes"],
        )

        # Test exact match for registered pass-through endpoint
        result1 = RouteChecks.is_virtual_key_allowed_to_call_route(
            route="/azure-assistant",
            valid_token=valid_token,
        )
        assert result1 is True

        # Test subpath for registered pass-through endpoint with subpath type
        result2 = RouteChecks.is_virtual_key_allowed_to_call_route(
            route="/custom-endpoint/openai/assistants",
            valid_token=valid_token,
        )
        assert result2 is True

        # Test exact match for subpath type
        result3 = RouteChecks.is_virtual_key_allowed_to_call_route(
            route="/custom-endpoint",
            valid_token=valid_token,
        )
        assert result3 is True


def test_virtual_key_without_llm_api_routes_cannot_access_pass_through():
    """
    Test that virtual keys without llm_api_routes permission cannot access registered pass-through endpoints.
    """

    # Mock the registered pass-through routes
    mock_registered_routes = {
        "test-uuid-1:exact:/azure-assistant": {
            "endpoint_id": "test-uuid-1",
            "path": "/azure-assistant",
            "type": "exact",
        },
    }

    with (
        patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints._registered_pass_through_routes",
            mock_registered_routes,
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_server_root_path",
            return_value="/",
        ),
    ):
        # Create a virtual key without llm_api_routes permission
        valid_token = UserAPIKeyAuth(
            user_id="test_user",
            allowed_routes=["info_routes"],
        )

        # Test that access is denied
        with pytest.raises(HTTPException) as exc_info:
            RouteChecks.is_virtual_key_allowed_to_call_route(
                route="/azure-assistant",
                valid_token=valid_token,
            )

        assert exc_info.value.status_code == 403
        assert "Virtual key is not allowed to call this route" in str(
            exc_info.value.detail
        )


def test_check_passthrough_route_access_key_metadata_exact_match():
    """Test that key metadata allowed_passthrough_routes allows exact match"""

    # Create a UserAPIKeyAuth with allowed_passthrough_routes in metadata
    valid_token = UserAPIKeyAuth(
        user_id="test_user",
        metadata={"allowed_passthrough_routes": ["/custom-endpoint"]},
    )

    # Test exact match
    result = RouteChecks.check_passthrough_route_access(
        route="/custom-endpoint",
        user_api_key_dict=valid_token,
    )

    assert result is True


def test_check_passthrough_route_access_key_metadata_prefix_match():
    """Test that key metadata allowed_passthrough_routes allows prefix match"""

    # Create a UserAPIKeyAuth with allowed_passthrough_routes in metadata
    valid_token = UserAPIKeyAuth(
        user_id="test_user",
        metadata={"allowed_passthrough_routes": ["/custom-endpoint"]},
    )

    # Test prefix match
    result = RouteChecks.check_passthrough_route_access(
        route="/custom-endpoint/v1/chat/completions",
        user_api_key_dict=valid_token,
    )

    assert result is True


def test_check_passthrough_route_access_key_metadata_no_match():
    """Test that key metadata allowed_passthrough_routes denies non-matching routes"""

    # Create a UserAPIKeyAuth with allowed_passthrough_routes in metadata
    valid_token = UserAPIKeyAuth(
        user_id="test_user",
        metadata={"allowed_passthrough_routes": ["/custom-endpoint"]},
    )

    # Test non-matching route
    result = RouteChecks.check_passthrough_route_access(
        route="/other-endpoint",
        user_api_key_dict=valid_token,
    )

    assert result is False


def test_check_passthrough_route_access_team_metadata_exact_match():
    """Test that team metadata allowed_passthrough_routes allows exact match"""

    # Create a UserAPIKeyAuth with allowed_passthrough_routes in team_metadata
    valid_token = UserAPIKeyAuth(
        user_id="test_user",
        metadata={},
        team_metadata={"allowed_passthrough_routes": ["/team-endpoint"]},
    )

    # Test exact match
    result = RouteChecks.check_passthrough_route_access(
        route="/team-endpoint",
        user_api_key_dict=valid_token,
    )

    assert result is True


def test_check_passthrough_route_access_team_metadata_prefix_match():
    """Test that team metadata allowed_passthrough_routes allows prefix match"""

    # Create a UserAPIKeyAuth with allowed_passthrough_routes in team_metadata
    valid_token = UserAPIKeyAuth(
        user_id="test_user",
        metadata={},
        team_metadata={"allowed_passthrough_routes": ["/team-endpoint"]},
    )

    # Test prefix match
    result = RouteChecks.check_passthrough_route_access(
        route="/team-endpoint/v1/messages",
        user_api_key_dict=valid_token,
    )

    assert result is True


def test_check_passthrough_route_access_team_metadata_no_match():
    """Test that team metadata allowed_passthrough_routes denies non-matching routes"""

    # Create a UserAPIKeyAuth with allowed_passthrough_routes in team_metadata
    valid_token = UserAPIKeyAuth(
        user_id="test_user",
        metadata={},
        team_metadata={"allowed_passthrough_routes": ["/team-endpoint"]},
    )

    # Test non-matching route
    result = RouteChecks.check_passthrough_route_access(
        route="/other-endpoint",
        user_api_key_dict=valid_token,
    )

    assert result is False


def test_check_passthrough_route_access_key_metadata_takes_precedence():
    """Test that key metadata takes precedence over team metadata"""

    # Create a UserAPIKeyAuth with different allowed_passthrough_routes in both metadata
    valid_token = UserAPIKeyAuth(
        user_id="test_user",
        metadata={"allowed_passthrough_routes": ["/key-endpoint"]},
        team_metadata={"allowed_passthrough_routes": ["/team-endpoint"]},
    )

    # Test that key endpoint is allowed
    result1 = RouteChecks.check_passthrough_route_access(
        route="/key-endpoint",
        user_api_key_dict=valid_token,
    )

    # Test that team endpoint is NOT allowed (key metadata takes precedence)
    result2 = RouteChecks.check_passthrough_route_access(
        route="/team-endpoint",
        user_api_key_dict=valid_token,
    )

    assert result1 is True
    assert result2 is False


def test_check_passthrough_route_access_no_metadata():
    """Test that route is denied when metadata and team_metadata don't have allowed_passthrough_routes"""

    # Create a UserAPIKeyAuth without allowed_passthrough_routes
    valid_token = UserAPIKeyAuth(
        user_id="test_user",
    )

    # Test that route is denied
    result = RouteChecks.check_passthrough_route_access(
        route="/any-endpoint",
        user_api_key_dict=valid_token,
    )

    assert result is False


def test_check_passthrough_route_access_no_allowed_passthrough_routes_key():
    """Test that route is denied when allowed_passthrough_routes is not in metadata"""

    # Create a UserAPIKeyAuth with metadata but no allowed_passthrough_routes
    valid_token = UserAPIKeyAuth(
        user_id="test_user",
        metadata={"other_field": "value"},
        team_metadata={},
    )

    # Test that route is denied
    result = RouteChecks.check_passthrough_route_access(
        route="/any-endpoint",
        user_api_key_dict=valid_token,
    )

    assert result is False


def test_check_passthrough_route_access_allowed_passthrough_routes_is_none():
    """Test that route is denied when allowed_passthrough_routes is None"""

    # Create a UserAPIKeyAuth with allowed_passthrough_routes set to None
    valid_token = UserAPIKeyAuth(
        user_id="test_user",
        metadata={"allowed_passthrough_routes": None},
        team_metadata={"allowed_passthrough_routes": None},
    )

    # Test that route is denied
    result = RouteChecks.check_passthrough_route_access(
        route="/any-endpoint",
        user_api_key_dict=valid_token,
    )

    assert result is False


def test_check_passthrough_route_access_multiple_routes():
    """Test that multiple allowed_passthrough_routes work correctly"""

    # Create a UserAPIKeyAuth with multiple allowed_passthrough_routes
    valid_token = UserAPIKeyAuth(
        user_id="test_user",
        metadata={
            "allowed_passthrough_routes": [
                "/endpoint-1",
                "/endpoint-2",
                "/endpoint-3",
            ]
        },
    )

    # Test that all allowed routes work
    result1 = RouteChecks.check_passthrough_route_access(
        route="/endpoint-1/v1/chat",
        user_api_key_dict=valid_token,
    )
    result2 = RouteChecks.check_passthrough_route_access(
        route="/endpoint-2",
        user_api_key_dict=valid_token,
    )
    result3 = RouteChecks.check_passthrough_route_access(
        route="/endpoint-3/completions",
        user_api_key_dict=valid_token,
    )

    # Test that non-allowed route fails
    result4 = RouteChecks.check_passthrough_route_access(
        route="/endpoint-4",
        user_api_key_dict=valid_token,
    )

    assert result1 is True
    assert result2 is True
    assert result3 is True
    assert result4 is False


def test_check_passthrough_route_access_prevents_false_prefix_match():
    """Test that prefix matching doesn't allow false matches like /endpoint vs /endpoint-2"""

    # Create a UserAPIKeyAuth with allowed_passthrough_routes
    valid_token = UserAPIKeyAuth(
        user_id="test_user",
        metadata={"allowed_passthrough_routes": ["/endpoint"]},
    )

    # Test that /endpoint-2 is NOT allowed (not a valid prefix match)
    result = RouteChecks.check_passthrough_route_access(
        route="/endpoint-2",
        user_api_key_dict=valid_token,
    )

    assert result is False

    # Test that /endpoint/something IS allowed (valid prefix match)
    result2 = RouteChecks.check_passthrough_route_access(
        route="/endpoint/something",
        user_api_key_dict=valid_token,
    )

    assert result2 is True


def test_check_passthrough_route_access_empty_list():
    """Test that empty allowed_passthrough_routes list denies all routes"""

    # Create a UserAPIKeyAuth with empty allowed_passthrough_routes
    valid_token = UserAPIKeyAuth(
        user_id="test_user",
        metadata={"allowed_passthrough_routes": []},
    )

    # Test that route is denied
    result = RouteChecks.check_passthrough_route_access(
        route="/any-endpoint",
        user_api_key_dict=valid_token,
    )

    assert result is False


@pytest.mark.parametrize(
    "route",
    [
        "/videos",
        "/v1/videos",
        "/videos/video_123",
        "/v1/videos/video_123",
        "/videos/video_123/content",
        "/v1/videos/video_123/content",
        "/videos/video_123/remix",
        "/v1/videos/video_123/remix",
    ],
)
def test_videos_route_is_llm_api_route(route):
    """Test that video routes are recognized as LLM API routes"""

    # Test that all video routes are recognized as LLM API routes
    assert RouteChecks.is_llm_api_route(route) is True


@pytest.mark.parametrize(
    "route",
    [
        "/containers",
        "/v1/containers",
        "/containers/container_123",
        "/v1/containers/container_123",
        "/containers/container_123/files",
        "/v1/containers/container_123/files",
        "/containers/container_123/files/file_456",
        "/v1/containers/container_123/files/file_456",
    ],
)
def test_containers_routes_are_llm_api_routes(route):
    """Test that container routes are recognized as LLM API routes"""

    assert RouteChecks.is_llm_api_route(route) is True


def test_videos_route_accessible_to_internal_users():
    """
    Test that internal users can access the videos routes.

    This test verifies the fix for issue #16470:
    https://github.com/BerriAI/litellm/issues/16470

    Videos routes should be accessible to internal_user role since video generation
    is a legitimate user feature, not a management/admin-only feature.
    """

    # Create an internal user object
    user_obj = LiteLLM_UserTable(
        user_id="test_user",
        user_email="test@example.com",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )

    # Create an internal user API key auth
    valid_token = UserAPIKeyAuth(
        user_id="test_user",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )

    # Create a mock request
    request = MagicMock(spec=Request)
    request.query_params = {}

    # Test that calling /v1/videos route does NOT raise an exception
    # Since videos is now in openai_routes, it should be accessible to internal users
    try:
        RouteChecks.non_proxy_admin_allowed_routes_check(
            user_obj=user_obj,
            _user_role=LitellmUserRoles.INTERNAL_USER.value,
            route="/v1/videos",
            request=request,
            valid_token=valid_token,
            request_data={"model": "sora-2", "prompt": "test video"},
        )
        # If no exception is raised, the test passes
    except Exception as e:
        pytest.fail(
            f"Internal user should be able to access /v1/videos route. Got error: {str(e)}"
        )


def test_videos_route_with_virtual_key_llm_api_routes():
    """Test that virtual keys with llm_api_routes permission can access videos endpoints"""

    # Create a virtual key with llm_api_routes permission
    valid_token = UserAPIKeyAuth(
        user_id="test_user",
        allowed_routes=["llm_api_routes"],
    )

    # Test that all video routes are accessible
    test_routes = [
        "/v1/videos",
        "/videos",
        "/v1/videos/video_123",
        "/videos/video_123/content",
        "/v1/videos/video_123/remix",
    ]

    for route in test_routes:
        result = RouteChecks.is_virtual_key_allowed_to_call_route(
            route=route, valid_token=valid_token
        )
        assert (
            result is True
        ), f"Virtual key with llm_api_routes should be able to access {route}"


def test_non_proxy_admin_wildcard_allowed_routes():
    """Test that nonproxy admin users can still use wildcard routes"""

    user_obj = LiteLLM_UserTable(
        user_id="test_user",
        user_email="test@example.com",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )

    valid_token = UserAPIKeyAuth(
        user_id="test_user",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
        allowed_routes=["/scim/*"],
    )

    request = MagicMock(spec=Request)
    request.query_params = {}

    RouteChecks.non_proxy_admin_allowed_routes_check(
        user_obj=user_obj,
        _user_role=LitellmUserRoles.INTERNAL_USER.value,
        route="/scim/v2/Users",
        request=request,
        valid_token=valid_token,
        request_data={},
    )


def test_proxy_admin_viewer_can_access_global_spend_tags():
    """
    Test that proxy_admin_viewer can access /global/spend/tags endpoint.

    This test verifies the fix for the issue where proxy_admin_viewer was getting
    403 errors when trying to access /global/spend/tags endpoint.

    Related: Slack thread from 10/9/2025 - Erik Kristensen reported this issue.
    proxy_admin_viewer role should have access to "view all spend" endpoints.
    """

    # Create a proxy admin viewer user object
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
    request.query_params = {"start_date": "2025-05-12", "end_date": "2025-10-09"}

    # Test that calling /global/spend/tags route does NOT raise an exception
    try:
        RouteChecks.non_proxy_admin_allowed_routes_check(
            user_obj=user_obj,
            _user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value,
            route="/global/spend/tags",
            request=request,
            valid_token=valid_token,
            request_data={},
        )
        # If no exception is raised, the test passes
    except Exception as e:
        pytest.fail(
            f"proxy_admin_viewer should be able to access /global/spend/tags route. Got error: {str(e)}"
        )


class TestModelsRouteExemptFromDisableLLMEndpoints:
    """
    Test that /models and /v1/models are exempt from DISABLE_LLM_API_ENDPOINTS.

    When DISABLE_LLM_API_ENDPOINTS is set, inference routes like /v1/chat/completions
    should be blocked, but /models and /v1/models should remain accessible because
    they are read-only model listing routes needed by the Admin UI.

    Relevant issue: https://github.com/BerriAI/litellm/issues/new (UI breaks with DISABLE_LLM_ENDPOINTS)
    """

    def _get_enterprise_route_checks(self):
        """Import EnterpriseRouteChecks from the local enterprise source file."""
        import importlib.util

        local_file = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..", "..", "enterprise",
            "litellm_enterprise", "proxy", "auth", "route_checks.py",
        )
        local_file = os.path.abspath(local_file)

        spec = importlib.util.spec_from_file_location(
            "local_enterprise_route_checks", local_file
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.EnterpriseRouteChecks

    @patch("litellm.proxy.proxy_server.premium_user", True)
    def test_should_models_route_allowed_when_llm_api_disabled(self):
        """Test that /models is allowed even when LLM API routes are disabled"""
        EnterpriseRouteChecks = self._get_enterprise_route_checks()

        with patch.object(
            EnterpriseRouteChecks, "is_llm_api_route_disabled", return_value=True
        ), patch.object(
            EnterpriseRouteChecks, "is_management_routes_disabled", return_value=False
        ):
            # /models should NOT raise - it's exempt
            EnterpriseRouteChecks.should_call_route("/models")

    @patch("litellm.proxy.proxy_server.premium_user", True)
    def test_should_v1_models_route_allowed_when_llm_api_disabled(self):
        """Test that /v1/models is allowed even when LLM API routes are disabled"""
        EnterpriseRouteChecks = self._get_enterprise_route_checks()

        with patch.object(
            EnterpriseRouteChecks, "is_llm_api_route_disabled", return_value=True
        ), patch.object(
            EnterpriseRouteChecks, "is_management_routes_disabled", return_value=False
        ):
            # /v1/models should NOT raise - it's exempt
            EnterpriseRouteChecks.should_call_route("/v1/models")

    @patch("litellm.proxy.proxy_server.premium_user", True)
    def test_should_chat_completions_still_blocked_when_llm_api_disabled(self):
        """Test that non-exempt LLM routes like /v1/chat/completions are still blocked"""
        EnterpriseRouteChecks = self._get_enterprise_route_checks()

        with patch.object(
            EnterpriseRouteChecks, "is_llm_api_route_disabled", return_value=True
        ), patch.object(
            EnterpriseRouteChecks, "is_management_routes_disabled", return_value=False
        ):
            with pytest.raises(HTTPException) as exc_info:
                EnterpriseRouteChecks.should_call_route("/v1/chat/completions")

            assert exc_info.value.status_code == 403
            assert "LLM API routes are disabled for this instance." in str(
                exc_info.value.detail
            )

    @patch("litellm.proxy.proxy_server.premium_user", True)
    def test_should_embeddings_still_blocked_when_llm_api_disabled(self):
        """Test that /v1/embeddings is still blocked when LLM API routes are disabled"""
        EnterpriseRouteChecks = self._get_enterprise_route_checks()

        with patch.object(
            EnterpriseRouteChecks, "is_llm_api_route_disabled", return_value=True
        ), patch.object(
            EnterpriseRouteChecks, "is_management_routes_disabled", return_value=False
        ):
            with pytest.raises(HTTPException) as exc_info:
                EnterpriseRouteChecks.should_call_route("/v1/embeddings")

            assert exc_info.value.status_code == 403

    @patch("litellm.proxy.proxy_server.premium_user", True)
    def test_should_models_route_allowed_when_llm_api_not_disabled(self):
        """Test that /models works normally when LLM API routes are not disabled"""
        EnterpriseRouteChecks = self._get_enterprise_route_checks()

        with patch.object(
            EnterpriseRouteChecks, "is_llm_api_route_disabled", return_value=False
        ), patch.object(
            EnterpriseRouteChecks, "is_management_routes_disabled", return_value=False
        ):
            # Should not raise
            EnterpriseRouteChecks.should_call_route("/models")
            EnterpriseRouteChecks.should_call_route("/v1/models")


def test_route_in_additional_public_routes_wildcard_match():
    """
    Test that route_in_additonal_public_routes supports wildcard patterns.
    """
    from litellm.proxy.auth.auth_utils import route_in_additonal_public_routes

    with (
        patch(
            "litellm.proxy.proxy_server.general_settings", {"public_routes": ["/api/*"]}
        ),
        patch("litellm.proxy.proxy_server.premium_user", True),
    ):
        # Wildcard should match subpaths
        assert route_in_additonal_public_routes("/api/users") is True
        assert route_in_additonal_public_routes("/api/users/123") is True
        # Should not match different prefix
        assert route_in_additonal_public_routes("/other/path") is False


def test_route_in_additional_public_routes_exact_match():
    """
    Test that route_in_additonal_public_routes supports exact matches.
    """
    from litellm.proxy.auth.auth_utils import route_in_additonal_public_routes

    with (
        patch(
            "litellm.proxy.proxy_server.general_settings",
            {"public_routes": ["/health", "/status"]},
        ),
        patch("litellm.proxy.proxy_server.premium_user", True),
    ):
        # Exact matches should work
        assert route_in_additonal_public_routes("/health") is True
        assert route_in_additonal_public_routes("/status") is True
        # Non-matching routes should fail
        assert route_in_additonal_public_routes("/other") is False
