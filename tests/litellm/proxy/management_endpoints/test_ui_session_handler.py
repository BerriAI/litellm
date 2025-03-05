import datetime
import json
import os
import sys
from datetime import timezone

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import jwt
import pytest
from fastapi.requests import Request
from fastapi.responses import RedirectResponse

from litellm.proxy._types import LitellmUserRoles
from litellm.proxy.management_helpers.ui_session_handler import UISessionHandler


class TestUISessionHandler:

    def test_get_latest_ui_cookie_name(self):
        # Test with multiple cookies
        cookies = {
            "litellm_ui_token_1000": "value1",
            "litellm_ui_token_2000": "value2",
            "other_cookie": "other_value",
        }

        result = UISessionHandler._get_latest_ui_cookie_name(cookies)
        assert result == "litellm_ui_token_2000"

        # Test with no matching cookies
        cookies = {"other_cookie": "value"}
        result = UISessionHandler._get_latest_ui_cookie_name(cookies)
        assert result is None

    def test_get_ui_session_token_from_cookies(self):
        # Create mock request with cookies
        mock_request = MagicMock()
        mock_request.cookies = {
            "litellm_ui_token_1000": "test_token",
            "other_cookie": "other_value",
        }

        result = UISessionHandler._get_ui_session_token_from_cookies(mock_request)
        assert result == "test_token"

        # Test with no matching cookies
        mock_request.cookies = {"other_cookie": "value"}
        result = UISessionHandler._get_ui_session_token_from_cookies(mock_request)
        assert result is None

    @patch("litellm.proxy.proxy_server.master_key", "test_master_key")
    @patch(
        "litellm.proxy.proxy_server.general_settings",
        {"litellm_key_header_name": "X-API-Key"},
    )
    def test_build_authenticated_ui_jwt_token(self):
        # Test token generation
        token = UISessionHandler.build_authenticated_ui_jwt_token(
            user_id="test_user",
            user_role=LitellmUserRoles.PROXY_ADMIN,
            user_email="test@example.com",
            premium_user=True,
            disabled_non_admin_personal_key_creation=False,
            login_method="username_password",
        )

        # Decode and verify token
        decoded = jwt.decode(
            token, "test_master_key", algorithms=["HS256"], audience="litellm-ui"
        )

        assert decoded["user_id"] == "test_user"
        assert decoded["user_email"] == "test@example.com"
        assert decoded["user_role"] == LitellmUserRoles.PROXY_ADMIN
        assert decoded["premium_user"] is True
        assert decoded["login_method"] == "username_password"
        assert decoded["auth_header_name"] == "X-API-Key"
        assert decoded["iss"] == "litellm-proxy"
        assert decoded["aud"] == "litellm-ui"
        assert "exp" in decoded
        assert decoded["disabled_non_admin_personal_key_creation"] is False
        assert decoded["scope"] == ["litellm_proxy_admin"]

    def test_is_ui_session_token(self):
        # Valid UI session token
        token_dict = {
            "iss": "litellm-proxy",
            "aud": "litellm-ui",
            "user_id": "test_user",
        }
        assert UISessionHandler.is_ui_session_token(token_dict) is True

        # Invalid token (wrong issuer)
        token_dict = {
            "iss": "other-issuer",
            "aud": "litellm-ui",
        }
        assert UISessionHandler.is_ui_session_token(token_dict) is False

        # Invalid token (wrong audience)
        token_dict = {
            "iss": "litellm-proxy",
            "aud": "other-audience",
        }
        assert UISessionHandler.is_ui_session_token(token_dict) is False

    def test_generate_authenticated_redirect_response(self):
        redirect_url = "https://example.com/dashboard"
        jwt_token = "test.jwt.token"

        response = UISessionHandler.generate_authenticated_redirect_response(
            redirect_url=redirect_url, jwt_token=jwt_token
        )

        assert isinstance(response, RedirectResponse)
        assert response.status_code == 303
        assert response.headers["location"] == redirect_url

        # Check cookie was set
        cookie_header = response.headers.get("set-cookie", "")
        assert "test.jwt.token" in cookie_header
        assert "Secure" in cookie_header
        assert "HttpOnly" in cookie_header
        assert "SameSite=strict" in cookie_header

    def test_generate_token_name(self):
        # Mock time.time() to return a fixed value
        with patch("time.time", return_value=1234567890):
            token_name = UISessionHandler._generate_token_name()
            assert token_name == "litellm_ui_token_1234567890"

    def test_latest_token_is_used(self):
        """Test that the most recent token is correctly identified and used"""
        # Create mock request with multiple UI tokens of different timestamps
        mock_request = MagicMock()
        mock_request.cookies = {
            "litellm_ui_token_1000": "old_token",
            "litellm_ui_token_2000": "newer_token",
            "litellm_ui_token_1500": "middle_token",
            "other_cookie": "other_value",
        }

        # Get the token from cookies
        result = UISessionHandler._get_ui_session_token_from_cookies(mock_request)

        # Verify the newest token (highest timestamp) is returned
        assert result == "newer_token"

        # Test with timestamps that aren't in numerical order in the dictionary
        mock_request.cookies = {
            "litellm_ui_token_5000": "newest_token",
            "litellm_ui_token_1000": "oldest_token",
            "litellm_ui_token_3000": "middle_token",
            "other_cookie": "other_value",
        }

        result = UISessionHandler._get_ui_session_token_from_cookies(mock_request)
        assert result == "newest_token"

        # Test with non-numeric timestamp parts
        mock_request.cookies = {
            "litellm_ui_token_abc": "invalid_token",
            "litellm_ui_token_2000": "valid_token",
            "other_cookie": "other_value",
        }

        # Should handle the error and still return a token (fallback to string sort)
        result = UISessionHandler._get_ui_session_token_from_cookies(mock_request)
        assert result is not None
