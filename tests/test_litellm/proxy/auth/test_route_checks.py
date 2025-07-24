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
