"""
Unit tests for EntraID app roles JWT claim extraction.

This module tests the get_app_roles_from_id_token method to ensure it correctly
extracts app roles from Microsoft EntraID JWT tokens and prevents regressions.
"""

import pytest
from unittest.mock import patch
import jwt

from litellm.proxy.management_endpoints.ui_sso import MicrosoftSSOHandler


class TestEntraIDAppRoles:
    """Test EntraID app roles extraction from JWT tokens"""

    @pytest.fixture
    def sample_jwt_token(self):
        """Create a sample JWT token with app_roles claim"""
        payload = {
            "sub": "user123",
            "email": "user@company.com",
            "app_roles": ["proxy_admin"],
            "aud": "litellm-app",
            "iss": "https://login.microsoftonline.com/tenant-id/v2.0",
            "exp": 9999999999,
        }
        return jwt.encode(payload, "secret", algorithm="HS256")

    @pytest.fixture
    def sample_jwt_token_single_role(self):
        """Create a sample JWT token with single app role"""
        payload = {
            "sub": "user456",
            "email": "admin@company.com",
            "app_roles": ["proxy_admin_viewer"],
            "aud": "litellm-app",
            "iss": "https://login.microsoftonline.com/tenant-id/v2.0",
            "exp": 9999999999,
        }
        return jwt.encode(payload, "secret", algorithm="HS256")

    @pytest.fixture
    def sample_jwt_token_no_roles(self):
        """Create a sample JWT token without app_roles claim"""
        payload = {
            "sub": "user789",
            "email": "user@company.com",
            "aud": "litellm-app",
            "iss": "https://login.microsoftonline.com/tenant-id/v2.0",
            "exp": 9999999999,
        }
        return jwt.encode(payload, "secret", algorithm="HS256")

    @pytest.fixture
    def sample_jwt_token_empty_roles(self):
        """Create a sample JWT token with empty app_roles array"""
        payload = {
            "sub": "user000",
            "email": "user@company.com",
            "app_roles": [],
            "aud": "litellm-app",
            "iss": "https://login.microsoftonline.com/tenant-id/v2.0",
            "exp": 9999999999,
        }
        return jwt.encode(payload, "secret", algorithm="HS256")

    def test_get_app_roles_from_id_token_single_role(
        self, sample_jwt_token_single_role
    ):
        """Test extracting single app role from JWT token"""
        # Act
        result = MicrosoftSSOHandler.get_app_roles_from_id_token(
            sample_jwt_token_single_role
        )

        # Assert
        assert result == ["proxy_admin_viewer"]
        assert len(result) == 1

    def test_get_app_roles_from_id_token_no_roles_claim(
        self, sample_jwt_token_no_roles
    ):
        """Test handling JWT token without app_roles claim"""
        # Act
        result = MicrosoftSSOHandler.get_app_roles_from_id_token(
            sample_jwt_token_no_roles
        )

        # Assert
        assert result == []
        assert len(result) == 0

    def test_get_app_roles_from_id_token_empty_roles(
        self, sample_jwt_token_empty_roles
    ):
        """Test handling JWT token with empty app_roles array"""
        # Act
        result = MicrosoftSSOHandler.get_app_roles_from_id_token(
            sample_jwt_token_empty_roles
        )

        # Assert
        assert result == []
        assert len(result) == 0

    def test_get_app_roles_from_id_token_none_input(self):
        """Test handling None input"""
        # Act
        result = MicrosoftSSOHandler.get_app_roles_from_id_token(None)

        # Assert
        assert result == []
        assert len(result) == 0

    def test_get_app_roles_from_id_token_empty_string(self):
        """Test handling empty string input"""
        # Act
        result = MicrosoftSSOHandler.get_app_roles_from_id_token("")

        # Assert
        assert result == []
        assert len(result) == 0

    def test_get_app_roles_from_id_token_invalid_jwt(self):
        """Test handling invalid JWT token"""
        # Act
        result = MicrosoftSSOHandler.get_app_roles_from_id_token("invalid.jwt.token")

        # Assert
        assert result == []
        assert len(result) == 0

    def test_get_app_roles_from_id_token_malformed_roles(self):
        """Test handling JWT with malformed app_roles (not a list)"""
        # Arrange
        payload = {
            "sub": "user123",
            "app_roles": "not_a_list",  # Should be a list
            "exp": 9999999999,
        }
        malformed_token = jwt.encode(payload, "secret", algorithm="HS256")

        # Act
        result = MicrosoftSSOHandler.get_app_roles_from_id_token(malformed_token)

        # Assert
        assert result == []
        assert len(result) == 0

    def test_get_app_roles_from_id_token_jwt_decode_exception(self):
        """Test handling JWT decode exceptions gracefully"""
        # Arrange
        invalid_token = "completely.invalid.token"

        # Act
        result = MicrosoftSSOHandler.get_app_roles_from_id_token(invalid_token)

        # Assert
        assert result == []
        assert len(result) == 0

    def test_get_app_roles_from_id_token_import_error(self):
        """Test handling import error for jwt library"""
        # Arrange
        with patch(
            "builtins.__import__", side_effect=ImportError("No module named 'jwt'")
        ):
            # Act
            result = MicrosoftSSOHandler.get_app_roles_from_id_token("any.token")

            # Assert
            assert result == []
            assert len(result) == 0

    def test_get_app_roles_from_id_token_uses_correct_claim_name(
        self, sample_jwt_token
    ):
        """Test that the method uses 'app_roles' claim, not 'roles' claim"""
        # This test ensures we don't regress to the old bug where 'roles' was used

        # Arrange - Create a token with both claims to verify correct one is used
        payload = {
            "sub": "user123",
            "roles": ["old_roles_claim"],  # This should be ignored
            "app_roles": ["proxy_admin"],  # This should be used
            "exp": 9999999999,
        }
        token_with_both_claims = jwt.encode(payload, "secret", algorithm="HS256")

        # Act
        result = MicrosoftSSOHandler.get_app_roles_from_id_token(token_with_both_claims)

        # Assert
        assert result == ["proxy_admin"]  # Should use app_roles, not roles
        assert "old_roles_claim" not in result

    def test_get_app_roles_from_id_token_case_sensitivity(self):
        """Test that app roles are extracted as-is (case sensitive)"""
        # Arrange
        payload = {
            "sub": "user123",
            "app_roles": ["PROXY_ADMIN", "Internal_User"],  # Mixed case
            "exp": 9999999999,
        }
        mixed_case_token = jwt.encode(payload, "secret", algorithm="HS256")

        # Act
        result = MicrosoftSSOHandler.get_app_roles_from_id_token(mixed_case_token)

        # Assert
        assert result == ["PROXY_ADMIN", "Internal_User"]
        assert "PROXY_ADMIN" in result
        assert "Internal_User" in result
