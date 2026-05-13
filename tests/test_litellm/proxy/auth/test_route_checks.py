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


@pytest.mark.parametrize(
    "role",
    [
        LitellmUserRoles.INTERNAL_USER.value,
        LitellmUserRoles.INTERNAL_USER_VIEW_ONLY.value,
    ],
)
@pytest.mark.parametrize(
    "route",
    ["/compliance/eu-ai-act", "/compliance/gdpr"],
)
def test_compliance_routes_open_to_non_admin_roles(role, route):
    """Compliance routes are stateless validators on caller-supplied log data
    — both non-admin internal_user roles can call them."""
    user_obj = LiteLLM_UserTable(
        user_id="test_user",
        user_email="test@example.com",
        user_role=role,
    )
    valid_token = UserAPIKeyAuth(user_id="test_user", user_role=role)
    request = MagicMock(spec=Request)
    request.query_params = {}

    RouteChecks.non_proxy_admin_allowed_routes_check(
        user_obj=user_obj,
        _user_role=role,
        route=route,
        request=request,
        valid_token=valid_token,
        request_data={},
    )


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


@pytest.mark.parametrize(
    "blocked_route",
    [
        # team write routes that previously fell through the blocklist
        "/team/block",
        "/team/unblock",
        "/team/permissions_update",
        "/team/permissions_bulk_update",
        # JWT key mapping write routes
        "/jwt/key/mapping/new",
        "/jwt/key/mapping/update",
        "/jwt/key/mapping/delete",
        # key write routes
        "/key/bulk_update",
        # path-parameterized key write routes (suffix match)
        "/key/abc123/regenerate",
        "/key/abc123/reset_spend",
        # baseline coverage of routes that were already blocked
        "/team/new",
        "/team/delete",
        "/key/generate",
        "/key/delete",
        "/model/new",
        "/model/delete",
    ],
)
def test_proxy_admin_viewer_blocked_management_writes(blocked_route):
    """View-only admins must be denied on every management write route — the
    fall-through path previously allowed /team/block, /team/unblock,
    /key/bulk_update, /key/{id}/reset_spend, and the JWT key-mapping routes."""
    with pytest.raises(HTTPException) as exc_info:
        RouteChecks._check_proxy_admin_viewer_access(
            route=blocked_route,
            _user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value,
            request_data={},
        )
    assert exc_info.value.status_code == 403
    assert blocked_route in str(exc_info.value.detail)


@pytest.mark.parametrize(
    "allowed_read_route",
    [
        "/team/info",
        "/team/list",
        "/v2/team/list",
        "/team/permissions_list",
        "/team/daily/activity",
        "/user/info",
        "/user/list",
        "/key/info",
        "/key/list",
        "/model/info",
        "/jwt/key/mapping/list",
        "/jwt/key/mapping/info",
    ],
)
def test_proxy_admin_viewer_allowed_management_reads(allowed_read_route):
    """View-only admins must still be allowed to read management routes."""
    # Should not raise
    RouteChecks._check_proxy_admin_viewer_access(
        route=allowed_read_route,
        _user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value,
        request_data={},
    )


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


@pytest.mark.parametrize(
    "route",
    [
        "/v1/mcp/server/register",
        "/v1/mcp/server/health",
        "/v1/mcp/server/submissions",
        "/v1/mcp/server/abc123",
        "/v1/mcp/server/abc123/approve",
        "/v1/mcp/server/oauth/session",
        "/v1/mcp/server/oauth/abc123/authorize",
    ],
)
def test_virtual_key_mcp_routes_allows_v1_mcp_server_subpaths(route):
    """Regression test: mcp_routes must allow /v1/mcp/server sub-paths (register, health, oauth, etc.)."""

    valid_token = UserAPIKeyAuth(
        user_id="test_user",
        allowed_routes=["mcp_routes"],
    )

    result = RouteChecks.is_virtual_key_allowed_to_call_route(
        route=route,
        valid_token=valid_token,
    )

    assert result is True


@pytest.mark.parametrize(
    "route",
    [
        "/v1/mcp/server",
        "/v1/mcp/server/abc-123",
        "/v1/mcp/server/abc-123/approve",
    ],
)
def test_mcp_management_routes_classified_as_management_not_llm_api(route):
    """MCP server CRUD must be management routes, not llm_api routes, so
    DISABLE_LLM_API_ENDPOINTS on admin nodes does not block the Admin UI."""

    assert RouteChecks.is_llm_api_route(route=route) is False
    assert RouteChecks.is_management_route(route=route) is True


@pytest.mark.parametrize(
    "route",
    [
        "/mcp/tools/call",
        "/mcp-rest/tools/call",
        "/mcp/tools/list",
    ],
)
def test_mcp_inference_routes_classified_as_llm_api(route):
    """MCP tool-call / passthrough routes must remain llm_api routes so they
    continue to be blocked by DISABLE_LLM_API_ENDPOINTS on admin nodes."""

    assert RouteChecks.is_llm_api_route(route=route) is True
    assert RouteChecks.is_management_route(route=route) is False


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


@pytest.mark.parametrize(
    "route",
    [
        "/rag/ingest",
        "/v1/rag/ingest",
        "/rag/query",
        "/v1/rag/query",
    ],
)
def test_rag_routes_are_llm_api_routes(route):
    """Test that RAG routes are recognized as LLM API routes (internal_user_viewer can access)"""

    assert RouteChecks.is_llm_api_route(route) is True


def test_rag_routes_accessible_to_internal_user_viewer():
    """
    Test that internal_user_viewer can access RAG routes (/rag/ingest, /rag/query).

    internal_user_viewer should be able to call RAG endpoints like chat/completions
    since they are LLM API routes. For /rag/ingest, they can only add to existing
    vector stores (enforced in the endpoint).
    """

    valid_token = UserAPIKeyAuth(
        user_id="test_user",
        user_role=LitellmUserRoles.INTERNAL_USER_VIEW_ONLY.value,
    )

    for route in ["/rag/ingest", "/v1/rag/ingest", "/rag/query", "/v1/rag/query"]:
        RouteChecks.non_proxy_admin_allowed_routes_check(
            user_obj=None,
            _user_role=LitellmUserRoles.INTERNAL_USER_VIEW_ONLY.value,
            route=route,
            request=MagicMock(spec=Request),
            valid_token=valid_token,
            request_data={},
        )


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


# Routes returning proxy-wide spend across every team / customer / api_key.
# Sourced from `LiteLLMRoutes.global_spend_tracking_routes` so any future
# additions to that list are exercised by these tests automatically.
from litellm.proxy._types import LiteLLMRoutes

GLOBAL_SPEND_ROUTES = LiteLLMRoutes.global_spend_tracking_routes.value


@pytest.mark.parametrize("route", GLOBAL_SPEND_ROUTES)
def test_internal_user_blocked_from_global_spend_routes(route):
    """
    Non-admin INTERNAL_USER role must NOT be able to read proxy-wide spend.
    These routes return spend across every team, customer, and api_key.
    """
    user_obj = LiteLLM_UserTable(
        user_id="internal_user",
        user_email="user@example.com",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )
    valid_token = UserAPIKeyAuth(
        user_id="internal_user",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )
    request = MagicMock(spec=Request)
    request.query_params = {}

    with pytest.raises(Exception) as exc_info:
        RouteChecks.non_proxy_admin_allowed_routes_check(
            user_obj=user_obj,
            _user_role=LitellmUserRoles.INTERNAL_USER.value,
            route=route,
            request=request,
            valid_token=valid_token,
            request_data={},
        )
    assert "Only proxy admin" in str(exc_info.value)


@pytest.mark.parametrize("route", GLOBAL_SPEND_ROUTES)
def test_internal_user_view_only_blocked_from_global_spend_routes(route):
    """
    INTERNAL_USER_VIEW_ONLY must also be blocked from proxy-wide spend routes.
    """
    user_obj = LiteLLM_UserTable(
        user_id="viewer_user",
        user_email="viewer@example.com",
        user_role=LitellmUserRoles.INTERNAL_USER_VIEW_ONLY.value,
    )
    valid_token = UserAPIKeyAuth(
        user_id="viewer_user",
        user_role=LitellmUserRoles.INTERNAL_USER_VIEW_ONLY.value,
    )
    request = MagicMock(spec=Request)
    request.query_params = {}

    with pytest.raises(Exception) as exc_info:
        RouteChecks.non_proxy_admin_allowed_routes_check(
            user_obj=user_obj,
            _user_role=LitellmUserRoles.INTERNAL_USER_VIEW_ONLY.value,
            route=route,
            request=request,
            valid_token=valid_token,
            request_data={},
        )
    assert "Only proxy admin" in str(exc_info.value)


@pytest.mark.parametrize("route", GLOBAL_SPEND_ROUTES)
def test_proxy_admin_viewer_can_access_all_global_spend_routes(route):
    """
    PROXY_ADMIN_VIEW_ONLY ("view all keys, view all spend") must retain access
    to every route in `global_spend_tracking_routes`.
    """
    user_obj = LiteLLM_UserTable(
        user_id="admin_viewer",
        user_email="admin_viewer@example.com",
        user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value,
    )
    valid_token = UserAPIKeyAuth(
        user_id="admin_viewer",
        user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value,
    )
    request = MagicMock(spec=Request)
    request.query_params = {}

    RouteChecks.non_proxy_admin_allowed_routes_check(
        user_obj=user_obj,
        _user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value,
        route=route,
        request=request,
        valid_token=valid_token,
        request_data={},
    )


@pytest.mark.parametrize("route", GLOBAL_SPEND_ROUTES)
def test_get_spend_routes_permission_keeps_access_for_internal_user(route):
    """
    A key minted with the `get_spend_routes` permission is an explicit
    admin opt-in and must continue to grant access even though the caller's
    role would otherwise be blocked.
    """
    user_obj = LiteLLM_UserTable(
        user_id="internal_user_with_permission",
        user_email="user@example.com",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )
    valid_token = UserAPIKeyAuth(
        user_id="internal_user_with_permission",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
        permissions={"get_spend_routes": True},
    )
    request = MagicMock(spec=Request)
    request.query_params = {}

    RouteChecks.non_proxy_admin_allowed_routes_check(
        user_obj=user_obj,
        _user_role=LitellmUserRoles.INTERNAL_USER.value,
        route=route,
        request=request,
        valid_token=valid_token,
        request_data={},
    )


@pytest.mark.parametrize("route", ["/audit", "/audit/some-log-id"])
def test_proxy_admin_viewer_can_access_audit_logs(route):
    """
    Test that proxy_admin_viewer can access /audit endpoints.

    Admin viewers should be able to view audit logs since these are read-only.
    """

    user_obj = LiteLLM_UserTable(
        user_id="viewer_user",
        user_email="viewer@example.com",
        user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value,
    )

    valid_token = UserAPIKeyAuth(
        user_id="viewer_user",
        user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value,
    )

    request = MagicMock(spec=Request)
    request.query_params = {}

    try:
        RouteChecks.non_proxy_admin_allowed_routes_check(
            user_obj=user_obj,
            _user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value,
            route=route,
            request=request,
            valid_token=valid_token,
            request_data={},
        )
    except Exception as e:
        pytest.fail(
            f"proxy_admin_viewer should be able to access {route} route. Got error: {str(e)}"
        )


# ── Admin Viewer parity: Logs page endpoints ──────────────────────────────────
#
# The Admin Viewer (PROXY_ADMIN_VIEW_ONLY) role is documented as
# "view all keys, view all spend" and follows a read-parity-with-Proxy-Admin
# rule. The UI Logs page is the most user-visible failure mode: filtering and
# log details break entirely when these routes are blocked at the route_checks
# layer, even though the underlying handlers already gate on PROXY_ADMIN_VIEW_ONLY.
#
# Each route below corresponds to a network call made by the Logs page
# (ui/litellm-dashboard/src/components/view_logs/) — see the comment on each.
ADMIN_VIEWER_LOGS_PAGE_ROUTES = [
    # Main paginated log list — uiSpendLogsCall in log_filter_logic.tsx & index.tsx
    "/spend/logs/ui",
    # Single-log detail drawer — fetched on row click in LogDetailsDrawer
    "/spend/logs/ui/abc-request-id",
    # Multi-call session drawer — sessionSpendLogsCall in LogDetailsDrawer
    "/spend/logs/session/ui",
    # End User filter dropdown — allEndUsersCall in index.tsx
    "/customer/list",
    "/customer/info",
    # Cost estimation — used by some log views
    "/cost/estimate",
    # Public spend logs / spend tracking routes that admin viewer should read
    "/spend/logs",
    "/spend/keys",
    "/spend/users",
    "/spend/tags",
    "/spend/calculate",
]


@pytest.mark.parametrize("route", ADMIN_VIEWER_LOGS_PAGE_ROUTES)
def test_proxy_admin_viewer_can_access_logs_page_endpoints(route):
    """
    PROXY_ADMIN_VIEW_ONLY must pass route_checks for every endpoint the UI
    Logs page depends on. Without these, the page renders empty / errors.
    """
    user_obj = LiteLLM_UserTable(
        user_id="viewer_user",
        user_email="viewer@example.com",
        user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value,
    )
    valid_token = UserAPIKeyAuth(
        user_id="viewer_user",
        user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value,
    )
    request = MagicMock(spec=Request)
    request.query_params = {}

    try:
        RouteChecks.non_proxy_admin_allowed_routes_check(
            user_obj=user_obj,
            _user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value,
            route=route,
            request=request,
            valid_token=valid_token,
            request_data={},
        )
    except Exception as e:
        pytest.fail(
            f"proxy_admin_viewer should be able to access {route}. Got error: {str(e)}"
        )


@pytest.mark.parametrize("route", ADMIN_VIEWER_LOGS_PAGE_ROUTES)
def test_internal_user_blocked_from_admin_viewer_logs_routes(route):
    """
    The Logs-page route opening above must NOT also widen access for
    INTERNAL_USER. Plain internal users still see only their own logs and
    must be blocked from proxy-wide spend tracking + customer routes.
    """
    user_obj = LiteLLM_UserTable(
        user_id="internal_user",
        user_email="user@example.com",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )
    valid_token = UserAPIKeyAuth(
        user_id="internal_user",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )
    request = MagicMock(spec=Request)
    request.query_params = {}

    # Routes already in `spend_tracking_routes` (which is part of
    # `internal_user_routes`) are intentionally accessible to internal users
    # for their own scoped spend — those handlers enforce per-user filtering.
    # /cost/estimate is similarly per-user. The /customer/* routes are
    # admin-only.
    INTERNAL_USER_BLOCKED_SUBSET = {
        "/customer/list",
        "/customer/info",
    }
    if route not in INTERNAL_USER_BLOCKED_SUBSET:
        return

    with pytest.raises(Exception) as exc_info:
        RouteChecks.non_proxy_admin_allowed_routes_check(
            user_obj=user_obj,
            _user_role=LitellmUserRoles.INTERNAL_USER.value,
            route=route,
            request=request,
            valid_token=valid_token,
            request_data={},
        )
    assert "Only proxy admin" in str(exc_info.value)


# ── Admin Viewer parity: Settings/observability read endpoints ────────────────
#
# These are GET endpoints accessible to PROXY_ADMIN that the UI exposes to
# admin viewers via sidebar items gated by `all_admin_roles` (which includes
# proxy_admin_viewer). Without these, the Logging & Alerts, Caching, Budgets,
# and Admin Settings pages break for admin viewers.
ADMIN_VIEWER_SETTINGS_ROUTES = [
    # Logging & Alerts page
    "/callbacks/list",
    "/callbacks/configs",
    "/get/config/callbacks",
    "/alerting/settings",
    # Admin Settings / Router Settings pages
    "/config/list",
    "/config/field/info",
    # Budgets page
    "/budget/list",
    "/budget/settings",
    # Invitation viewing (admin viewer cannot create/delete; can read)
    "/invitation/info",
    # Guardrails / Policies pages (read-only views)
    "/guardrails/list",
    "/v2/guardrails/list",
    "/guardrails/submissions",
    "/guardrails/submissions/some-guardrail-id",
    "/guardrails/usage/overview",
    "/policies/attachments/list",
    # MCP semantic filter settings (read)
    "/get/mcp_semantic_filter_settings",
    # Model cost map (read-only status / source)
    "/schedule/model_cost_map_reload/status",
    "/model/cost_map/source",
]


@pytest.mark.parametrize("route", ADMIN_VIEWER_SETTINGS_ROUTES)
def test_proxy_admin_viewer_can_access_settings_read_endpoints(route):
    """
    PROXY_ADMIN_VIEW_ONLY must pass route_checks for the read-only
    settings/observability endpoints exposed in admin-only sidebar groups.
    """
    user_obj = LiteLLM_UserTable(
        user_id="viewer_user",
        user_email="viewer@example.com",
        user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value,
    )
    valid_token = UserAPIKeyAuth(
        user_id="viewer_user",
        user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value,
    )
    request = MagicMock(spec=Request)
    request.query_params = {}

    try:
        RouteChecks.non_proxy_admin_allowed_routes_check(
            user_obj=user_obj,
            _user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value,
            route=route,
            request=request,
            valid_token=valid_token,
            request_data={},
        )
    except Exception as e:
        pytest.fail(
            f"proxy_admin_viewer should be able to access {route}. Got error: {str(e)}"
        )


# ── Admin Viewer parity: default-allow GET semantics ─────────────────────────
#
# The route-check layer is structured to default-allow safe HTTP methods
# (GET / HEAD / OPTIONS) for PROXY_ADMIN_VIEW_ONLY. This eliminates the
# whack-a-mole where every newly-added GET endpoint silently 403'd until
# someone remembered to add it to admin_viewer_routes.
#
# These tests pin the new contract:
#   - Any GET endpoint not on the LLM/inference path is readable.
#   - Any unsafe method (POST/PUT/PATCH/DELETE) outside the explicit allow
#     sets is still 403.

# Routes the user reported as broken in production — they're in disparate
# corners of the codebase and represent the long tail of GETs we'd otherwise
# need to enumerate manually. Default-allow makes them all work.
ADMIN_VIEWER_REPORTED_GET_ROUTES = [
    "/in_product_nudges",
    "/health/latest",
    "/credentials",
    "/v1/mcp/network/client-ip",
    "/claude-code/plugins",
    "/policy/templates",
    # Routes we already had to enumerate manually (regression coverage).
    "/spend/logs/ui",
    "/customer/list",
    "/guardrails/list",
    "/policies/attachments/list",
    # Hypothetical future GETs — must not require an allowlist entry.
    "/some/future/read/endpoint",
    "/another/admin-tool/status",
]


@pytest.mark.parametrize("route", ADMIN_VIEWER_REPORTED_GET_ROUTES)
def test_proxy_admin_viewer_default_allows_any_get(route):
    """
    PROXY_ADMIN_VIEW_ONLY must be able to GET any non-inference endpoint.

    This is a structural guarantee: the route-check defaults to allow for
    safe HTTP methods so we don't have to maintain an explicit allowlist.
    """
    user_obj = LiteLLM_UserTable(
        user_id="viewer_user",
        user_email="viewer@example.com",
        user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value,
    )
    valid_token = UserAPIKeyAuth(
        user_id="viewer_user",
        user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value,
    )
    request = MagicMock(spec=Request)
    request.method = "GET"
    request.query_params = {}
    request.url = MagicMock()
    request.url.path = route

    try:
        RouteChecks.non_proxy_admin_allowed_routes_check(
            user_obj=user_obj,
            _user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value,
            route=route,
            request=request,
            valid_token=valid_token,
            request_data={},
        )
    except Exception as e:
        pytest.fail(f"proxy_admin_viewer GET should default-allow {route!r}. Got: {e}")


@pytest.mark.parametrize(
    "route",
    [
        # Random path that isn't in any allowlist — POST must still 403.
        "/some/future/write/endpoint",
        # Hard-blocked write routes.
        "/user/new",
        "/team/new",
        "/key/generate",
        "/model/new",
    ],
)
def test_proxy_admin_viewer_post_blocked_outside_allowlists(route):
    """
    Default-allow only applies to safe HTTP methods. POST/PUT/PATCH/DELETE
    on a route not in any allow set must still 403.
    """
    user_obj = LiteLLM_UserTable(
        user_id="viewer_user",
        user_email="viewer@example.com",
        user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value,
    )
    valid_token = UserAPIKeyAuth(
        user_id="viewer_user",
        user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value,
    )
    request = MagicMock(spec=Request)
    request.method = "POST"
    request.query_params = {}

    with pytest.raises(HTTPException) as exc_info:
        RouteChecks.non_proxy_admin_allowed_routes_check(
            user_obj=user_obj,
            _user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value,
            route=route,
            request=request,
            valid_token=valid_token,
            request_data={},
        )
    assert exc_info.value.status_code == 403


# ── Admin Viewer: management_routes write endpoints stay blocked ─────────────
#
# `management_routes` is a mix of reads (info/list, handled via the safe-method
# branch — GET) and writes. The route_checks layer must NOT blanket-allow the
# whole set on POST — that would let Admin Viewer mutate teams, JWT mappings,
# and bulk-update keys, violating the "no writes, ever" rule.
#
# These cases pin the gap closed (Greptile P1 review, 2026-04-30).
ADMIN_VIEWER_MANAGEMENT_ROUTE_WRITES = [
    # Team writes
    "/team/block",
    "/team/unblock",
    "/team/permissions_update",
    # JWT key mapping writes
    "/jwt/key/mapping/new",
    "/jwt/key/mapping/update",
    "/jwt/key/mapping/delete",
    # Key writes (existing _ADMIN_VIEWER_BLOCKED_WRITE_ROUTES doesn't list bulk
    # update or per-key reset-spend, so the management_routes fallback was the
    # only thing keeping them out — and it was permissive, not restrictive).
    "/key/bulk_update",
    "/key/some-key-id/reset_spend",
]


@pytest.mark.parametrize("route", ADMIN_VIEWER_MANAGEMENT_ROUTE_WRITES)
def test_proxy_admin_viewer_post_blocked_for_management_route_writes(route):
    """
    Admin Viewer must be blocked on POST to write endpoints in
    `management_routes`, even when the specific route is not in
    `_ADMIN_VIEWER_BLOCKED_WRITE_ROUTES`.
    """
    user_obj = LiteLLM_UserTable(
        user_id="viewer_user",
        user_email="viewer@example.com",
        user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value,
    )
    valid_token = UserAPIKeyAuth(
        user_id="viewer_user",
        user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value,
    )
    request = MagicMock(spec=Request)
    request.method = "POST"
    request.query_params = {}

    with pytest.raises(HTTPException) as exc_info:
        RouteChecks.non_proxy_admin_allowed_routes_check(
            user_obj=user_obj,
            _user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value,
            route=route,
            request=request,
            valid_token=valid_token,
            request_data={},
        )
    assert exc_info.value.status_code == 403


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
            "..",
            "..",
            "..",
            "..",
            "enterprise",
            "litellm_enterprise",
            "proxy",
            "auth",
            "route_checks.py",
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

        with (
            patch.object(
                EnterpriseRouteChecks, "is_llm_api_route_disabled", return_value=True
            ),
            patch.object(
                EnterpriseRouteChecks,
                "is_management_routes_disabled",
                return_value=False,
            ),
        ):
            # /models should NOT raise - it's exempt
            EnterpriseRouteChecks.should_call_route("/models")

    @patch("litellm.proxy.proxy_server.premium_user", True)
    def test_should_v1_models_route_allowed_when_llm_api_disabled(self):
        """Test that /v1/models is allowed even when LLM API routes are disabled"""
        EnterpriseRouteChecks = self._get_enterprise_route_checks()

        with (
            patch.object(
                EnterpriseRouteChecks, "is_llm_api_route_disabled", return_value=True
            ),
            patch.object(
                EnterpriseRouteChecks,
                "is_management_routes_disabled",
                return_value=False,
            ),
        ):
            # /v1/models should NOT raise - it's exempt
            EnterpriseRouteChecks.should_call_route("/v1/models")

    @patch("litellm.proxy.proxy_server.premium_user", True)
    def test_should_chat_completions_still_blocked_when_llm_api_disabled(self):
        """Test that non-exempt LLM routes like /v1/chat/completions are still blocked"""
        EnterpriseRouteChecks = self._get_enterprise_route_checks()

        with (
            patch.object(
                EnterpriseRouteChecks, "is_llm_api_route_disabled", return_value=True
            ),
            patch.object(
                EnterpriseRouteChecks,
                "is_management_routes_disabled",
                return_value=False,
            ),
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

        with (
            patch.object(
                EnterpriseRouteChecks, "is_llm_api_route_disabled", return_value=True
            ),
            patch.object(
                EnterpriseRouteChecks,
                "is_management_routes_disabled",
                return_value=False,
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                EnterpriseRouteChecks.should_call_route("/v1/embeddings")

            assert exc_info.value.status_code == 403

    @patch("litellm.proxy.proxy_server.premium_user", True)
    def test_should_models_route_allowed_when_llm_api_not_disabled(self):
        """Test that /models works normally when LLM API routes are not disabled"""
        EnterpriseRouteChecks = self._get_enterprise_route_checks()

        with (
            patch.object(
                EnterpriseRouteChecks, "is_llm_api_route_disabled", return_value=False
            ),
            patch.object(
                EnterpriseRouteChecks,
                "is_management_routes_disabled",
                return_value=False,
            ),
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


def test_internal_user_can_access_key_reset_spend_route():
    """
    Regression test: team admins (role=internal_user) should pass the route-level
    check for /key/{hash}/reset_spend. The endpoint itself enforces team admin status.
    """
    user_obj = LiteLLM_UserTable(
        user_id="team-admin-user",
        user_email="teamadmin@example.com",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )
    valid_token = UserAPIKeyAuth(
        user_id="team-admin-user",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )
    request = MagicMock(spec=Request)
    request.query_params = {}

    key_hash = "baec26d2901589fe9fec76610e6e2be4895cdd8e19b3ada9a4fa2eb85e1901ae"
    route = f"/key/{key_hash}/reset_spend"

    # Should not raise — the route-level check must pass for team admins
    RouteChecks.non_proxy_admin_allowed_routes_check(
        user_obj=user_obj,
        _user_role=LitellmUserRoles.INTERNAL_USER.value,
        route=route,
        request=request,
        valid_token=valid_token,
        request_data={},
    )


def test_non_admin_non_team_admin_cannot_access_config_update_but_can_attempt_reset_spend():
    """
    An internal_user passes the route check for /key/{hash}/reset_spend
    (authorization is deferred to the endpoint), but is still blocked from
    admin-only routes like /config/update.
    """
    user_obj = LiteLLM_UserTable(
        user_id="regular-user",
        user_email="user@example.com",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )
    valid_token = UserAPIKeyAuth(
        user_id="regular-user",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )
    request = MagicMock(spec=Request)
    request.query_params = {}

    key_hash = "baec26d2901589fe9fec76610e6e2be4895cdd8e19b3ada9a4fa2eb85e1901ae"

    # /key/{hash}/reset_spend passes the route check for internal_user
    RouteChecks.non_proxy_admin_allowed_routes_check(
        user_obj=user_obj,
        _user_role=LitellmUserRoles.INTERNAL_USER.value,
        route=f"/key/{key_hash}/reset_spend",
        request=request,
        valid_token=valid_token,
        request_data={},
    )

    # /config/update is still blocked
    with pytest.raises(Exception) as exc_info:
        RouteChecks.non_proxy_admin_allowed_routes_check(
            user_obj=user_obj,
            _user_role=LitellmUserRoles.INTERNAL_USER.value,
            route="/config/update",
            request=request,
            valid_token=valid_token,
            request_data={},
        )
    assert "Only proxy admin can be used to generate" in str(exc_info.value)


@pytest.mark.parametrize(
    "user_role",
    [
        LitellmUserRoles.INTERNAL_USER.value,
        LitellmUserRoles.INTERNAL_USER_VIEW_ONLY.value,
    ],
)
@pytest.mark.parametrize("route", ["/tag/list", "/tag/daily/activity"])
def test_internal_users_can_access_scoped_tag_usage_routes(user_role, route):
    """
    Internal users can read tag usage endpoints because the endpoint handlers
    scope results to the caller's own keys.
    """
    user_obj = LiteLLM_UserTable(
        user_id="test_user",
        user_email="test@example.com",
        user_role=user_role,
    )
    valid_token = UserAPIKeyAuth(
        user_id="test_user",
        user_role=user_role,
    )
    request = MagicMock(spec=Request)
    request.query_params = {}

    RouteChecks.non_proxy_admin_allowed_routes_check(
        user_obj=user_obj,
        _user_role=user_role,
        route=route,
        request=request,
        valid_token=valid_token,
        request_data={},
    )


@pytest.mark.parametrize(
    "user_role",
    [
        LitellmUserRoles.INTERNAL_USER.value,
        LitellmUserRoles.INTERNAL_USER_VIEW_ONLY.value,
        LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value,
    ],
)
def test_available_roles_accessible_to_non_admin_users(user_role):
    """
    /user/available_roles is read-only role metadata that any authenticated user
    (including org admins and team admins) needs when inviting users. It should
    pass the route check for all non-proxy-admin roles without requiring an
    organization_id in the request body.
    """
    user_obj = LiteLLM_UserTable(
        user_id="test_user",
        user_email="test@example.com",
        user_role=user_role,
    )
    valid_token = UserAPIKeyAuth(
        user_id="test_user",
        user_role=user_role,
    )
    request = MagicMock(spec=Request)
    request.query_params = {}

    # Should not raise — /user/available_roles is in self_managed_routes
    RouteChecks.non_proxy_admin_allowed_routes_check(
        user_obj=user_obj,
        _user_role=user_role,
        route="/user/available_roles",
        request=request,
        valid_token=valid_token,
        request_data={},
    )


# ── _user_is_org_admin tests ──────────────────────────────────────────────────

from datetime import datetime

from litellm.proxy._types import LiteLLM_OrganizationMembershipTable
from litellm.proxy.auth.auth_checks_organization import _user_is_org_admin


def _make_org_admin_user(org_id: str) -> LiteLLM_UserTable:
    membership = LiteLLM_OrganizationMembershipTable(
        user_id="org-admin-user",
        organization_id=org_id,
        user_role=LitellmUserRoles.ORG_ADMIN.value,
        created_at=datetime(2024, 1, 1),
        updated_at=datetime(2024, 1, 1),
    )
    return LiteLLM_UserTable(
        user_id="org-admin-user",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
        organization_memberships=[membership],
    )


def test_user_is_org_admin_with_organizations_list():
    """Org admin can be identified via the `organizations` list field (used by /user/new)."""
    user_obj = _make_org_admin_user("org-1")
    assert _user_is_org_admin({"organizations": ["org-1"]}, user_obj) is True


def test_user_is_org_admin_with_singular_organization_id():
    """Backward-compat: org admin can still be identified via singular `organization_id`."""
    user_obj = _make_org_admin_user("org-1")
    assert _user_is_org_admin({"organization_id": "org-1"}, user_obj) is True


def test_user_is_org_admin_organizations_list_wrong_org():
    """Non-member of the requested org is not considered an org admin for it."""
    user_obj = _make_org_admin_user("org-2")
    assert _user_is_org_admin({"organizations": ["org-1"]}, user_obj) is False


def test_user_is_org_admin_no_org_fields():
    """Returns False when neither `organization_id` nor `organizations` is in the request."""
    user_obj = _make_org_admin_user("org-1")
    assert _user_is_org_admin({}, user_obj) is False


def test_non_org_admin_with_organizations_list():
    """A regular internal user is not an org admin even if they are a member of the org."""
    membership = LiteLLM_OrganizationMembershipTable(
        user_id="regular-user",
        organization_id="org-1",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
        created_at=datetime(2024, 1, 1),
        updated_at=datetime(2024, 1, 1),
    )
    user_obj = LiteLLM_UserTable(
        user_id="regular-user",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
        organization_memberships=[membership],
    )
    assert _user_is_org_admin({"organizations": ["org-1"]}, user_obj) is False


def test_org_admin_cannot_escalate_to_other_org():
    """Regression: admin of org-A requesting [org-A, org-B] must be rejected."""
    user_obj = _make_org_admin_user("org-A")
    assert _user_is_org_admin({"organizations": ["org-A", "org-B"]}, user_obj) is False


def test_org_admin_of_multiple_orgs_can_operate_on_both():
    """Admin of both org-A and org-B can operate on both."""
    memberships = [
        LiteLLM_OrganizationMembershipTable(
            user_id="multi-admin",
            organization_id="org-A",
            user_role=LitellmUserRoles.ORG_ADMIN.value,
            created_at=datetime(2024, 1, 1),
            updated_at=datetime(2024, 1, 1),
        ),
        LiteLLM_OrganizationMembershipTable(
            user_id="multi-admin",
            organization_id="org-B",
            user_role=LitellmUserRoles.ORG_ADMIN.value,
            created_at=datetime(2024, 1, 1),
            updated_at=datetime(2024, 1, 1),
        ),
    ]
    user_obj = LiteLLM_UserTable(
        user_id="multi-admin",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
        organization_memberships=memberships,
    )
    assert _user_is_org_admin({"organizations": ["org-A", "org-B"]}, user_obj) is True


@pytest.mark.asyncio
async def test_initialize_pass_through_registers_wildcard_for_auth_subpath():
    """
    Test that initialize_pass_through_endpoints registers both base path and
    wildcard path in openai_routes when auth=true and include_subpath=true,
    and that subpath requests pass is_llm_api_route.

    Also verifies:
    - Dedup: calling init twice does not duplicate entries
    - Cleanup: removing the endpoint cleans up openai_routes
    """
    from litellm.proxy._types import LiteLLMRoutes
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        InitPassThroughEndpointHelpers,
        initialize_pass_through_endpoints,
    )

    base_path = "/v1/ocr/nvidia/community/nemoretriever-ocr-v1"
    wildcard_path = base_path + "/*"

    endpoint_config = {
        "path": base_path,
        "target": "https://httpbin.org/post",
        "include_subpath": True,
        "auth": True,
        "headers": {"content-type": "application/json"},
    }

    original_routes = LiteLLMRoutes.openai_routes.value[:]
    try:
        with (
            patch(
                "litellm.proxy.proxy_server.app",
                MagicMock(),
            ),
            patch(
                "litellm.proxy.proxy_server.premium_user",
                True,
            ),
            patch(
                "litellm.proxy.proxy_server.config_passthrough_endpoints",
                None,
            ),
        ):
            await initialize_pass_through_endpoints([endpoint_config])

            # Both base and wildcard paths should be registered
            assert base_path in LiteLLMRoutes.openai_routes.value
            assert wildcard_path in LiteLLMRoutes.openai_routes.value

            # Subpath requests should pass the auth route check
            assert RouteChecks.is_llm_api_route(base_path) is True
            assert RouteChecks.is_llm_api_route(base_path + "/v1/infer") is True

            # Calling init again should not duplicate entries
            await initialize_pass_through_endpoints([endpoint_config])
            assert LiteLLMRoutes.openai_routes.value.count(base_path) == 1
            assert LiteLLMRoutes.openai_routes.value.count(wildcard_path) == 1

            # Removing the endpoint should clean up openai_routes
            # remove_endpoint_routes takes endpoint_id (UUID portion of
            # the route key "{id}:exact:{path}:{methods}")
            registered = (
                InitPassThroughEndpointHelpers.get_all_registered_pass_through_routes()
            )
            endpoint_ids = {k.split(":")[0] for k in registered}
            for eid in endpoint_ids:
                InitPassThroughEndpointHelpers.remove_endpoint_routes(eid)
            assert base_path not in LiteLLMRoutes.openai_routes.value
            assert wildcard_path not in LiteLLMRoutes.openai_routes.value
    finally:
        LiteLLMRoutes.openai_routes.value[:] = original_routes
        # Clean up any routes registered during this test to avoid
        # polluting the module-level _registered_pass_through_routes
        registered = (
            InitPassThroughEndpointHelpers.get_all_registered_pass_through_routes()
        )
        for k in registered:
            InitPassThroughEndpointHelpers.remove_endpoint_routes(k.split(":")[0])
