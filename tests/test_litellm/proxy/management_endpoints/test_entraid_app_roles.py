"""
Unit tests for EntraID app roles JWT claim extraction.

This module tests the get_app_roles_from_id_token method to ensure it correctly
extracts app roles from Microsoft EntraID JWT tokens and prevents regressions.
"""

import pytest
import jwt

from litellm.proxy.management_endpoints.ui_sso import MicrosoftSSOHandler


class TestEntraIDAppRoles:
    """Test EntraID app roles extraction from JWT tokens"""

    def test_get_app_roles_from_id_token_works_without_roles(self):
        """Test that JWT token works fine without app_roles claim"""
        # Arrange - Token without app_roles (normal user)
        payload = {
            "sub": "user123",
            "email": "user@company.com",
            "aud": "litellm-app",
            "iss": "https://login.microsoftonline.com/tenant-id/v2.0",
            "exp": 9999999999,
        }
        no_roles_token = jwt.encode(payload, "secret", algorithm="HS256")

        # Act
        result = MicrosoftSSOHandler.get_app_roles_from_id_token(no_roles_token)

        # Assert - Should return empty list, not error
        assert result == []
        assert len(result) == 0

    def test_get_app_roles_from_id_token_assigns_roles_when_present(self):
        """Test that valid app roles are properly assigned when present"""
        # Arrange - Token with valid roles
        payload = {
            "sub": "user123",
            "email": "admin@company.com",
            "app_roles": ["proxy_admin"],
            "aud": "litellm-app",
            "iss": "https://login.microsoftonline.com/tenant-id/v2.0",
            "exp": 9999999999,
        }
        valid_roles_token = jwt.encode(payload, "secret", algorithm="HS256")

        # Act
        result = MicrosoftSSOHandler.get_app_roles_from_id_token(valid_roles_token)

        # Assert - Should extract the role
        assert result == ["proxy_admin"]
        assert len(result) == 1
        assert "proxy_admin" in result
