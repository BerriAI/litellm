import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

import pytest
from fastapi import HTTPException

# Import the enterprise route checks
from litellm_enterprise.proxy.auth.route_checks import EnterpriseRouteChecks


@patch("litellm.proxy.proxy_server.premium_user", True)
class TestEnterpriseRouteChecks:

    @patch.object(EnterpriseRouteChecks, "is_management_routes_disabled")
    @patch("litellm.proxy.auth.route_checks.RouteChecks.is_management_route")
    def test_should_call_route_management_disabled(
        self, mock_is_management_route, mock_is_management_disabled
    ):
        """Test that should_call_route raises HTTPException when management routes are disabled and route is a management route"""

        # Mock the methods to return True (route is management route and management is disabled)
        mock_is_management_route.return_value = True
        mock_is_management_disabled.return_value = True

        # Test that calling should_call_route raises HTTPException with 403 status
        with pytest.raises(HTTPException) as exc_info:
            EnterpriseRouteChecks.should_call_route("/config/update")

        # Verify the exception has correct status and message
        assert exc_info.value.status_code == 403
        assert "Management routes are disabled for this instance." in str(
            exc_info.value.detail
        )

    @patch.object(EnterpriseRouteChecks, "is_llm_api_route_disabled")
    @patch.object(EnterpriseRouteChecks, "is_management_routes_disabled")
    @patch("litellm.proxy.auth.route_checks.RouteChecks.is_llm_api_route")
    @patch("litellm.proxy.auth.route_checks.RouteChecks.is_management_route")
    def test_should_call_route_llm_api_disabled(
        self,
        mock_is_management_route,
        mock_is_llm_api_route,
        mock_is_management_disabled,
        mock_is_llm_api_disabled,
    ):
        """Test that should_call_route raises HTTPException when LLM API routes are disabled and route is an LLM API route"""

        # Mock the methods - not a management route but is an LLM API route that's disabled
        mock_is_management_route.return_value = False
        mock_is_llm_api_route.return_value = True
        mock_is_management_disabled.return_value = False
        mock_is_llm_api_disabled.return_value = True

        # Test that calling should_call_route raises HTTPException with 403 status
        with pytest.raises(HTTPException) as exc_info:
            EnterpriseRouteChecks.should_call_route("/v1/chat/completions")

        # Verify the exception has correct status and message
        assert exc_info.value.status_code == 403
        assert "LLM API routes are disabled for this instance." in str(
            exc_info.value.detail
        )

    @patch.object(EnterpriseRouteChecks, "is_llm_api_route_disabled")
    @patch.object(EnterpriseRouteChecks, "is_management_routes_disabled")
    @patch("litellm.proxy.auth.route_checks.RouteChecks.is_llm_api_route")
    @patch("litellm.proxy.auth.route_checks.RouteChecks.is_management_route")
    def test_should_call_route_both_disabled_management_takes_priority(
        self,
        mock_is_management_route,
        mock_is_llm_api_route,
        mock_is_management_disabled,
        mock_is_llm_api_disabled,
    ):
        """Test that management route check takes priority when both are disabled"""

        # Mock the methods - route is both management and LLM API, and both are disabled
        mock_is_management_route.return_value = True
        mock_is_llm_api_route.return_value = True
        mock_is_management_disabled.return_value = True
        mock_is_llm_api_disabled.return_value = True

        # Test that calling should_call_route raises HTTPException with management route message
        with pytest.raises(HTTPException) as exc_info:
            EnterpriseRouteChecks.should_call_route("/config/update")

        # Verify the exception has correct status and management route message (not LLM API message)
        assert exc_info.value.status_code == 403
        assert "Management routes are disabled for this instance." in str(
            exc_info.value.detail
        )

    @patch.object(EnterpriseRouteChecks, "is_llm_api_route_disabled")
    @patch.object(EnterpriseRouteChecks, "is_management_routes_disabled")
    @patch("litellm.proxy.auth.route_checks.RouteChecks.is_llm_api_route")
    @patch("litellm.proxy.auth.route_checks.RouteChecks.is_management_route")
    def test_should_call_route_enabled(
        self,
        mock_is_management_route,
        mock_is_llm_api_route,
        mock_is_management_disabled,
        mock_is_llm_api_disabled,
    ):
        """Test that should_call_route succeeds when routes are enabled"""

        # Test case 1: Management route enabled
        mock_is_management_route.return_value = True
        mock_is_llm_api_route.return_value = False
        mock_is_management_disabled.return_value = False
        mock_is_llm_api_disabled.return_value = False

        # Should not raise exception
        EnterpriseRouteChecks.should_call_route("/config/update")

        # Test case 2: LLM API route enabled
        mock_is_management_route.return_value = False
        mock_is_llm_api_route.return_value = True
        mock_is_management_disabled.return_value = False
        mock_is_llm_api_disabled.return_value = False

        # Should not raise exception
        EnterpriseRouteChecks.should_call_route("/v1/chat/completions")

        # Test case 3: Neither management nor LLM API route
        mock_is_management_route.return_value = False
        mock_is_llm_api_route.return_value = False
        mock_is_management_disabled.return_value = (
            True  # These can be True since route doesn't match
        )
        mock_is_llm_api_disabled.return_value = True

        # Should not raise exception
        EnterpriseRouteChecks.should_call_route("/health")

    @patch.object(EnterpriseRouteChecks, "is_llm_api_route_disabled")
    @patch.object(EnterpriseRouteChecks, "is_management_routes_disabled")
    @patch("litellm.proxy.auth.route_checks.RouteChecks.is_llm_api_route")
    @patch("litellm.proxy.auth.route_checks.RouteChecks.is_management_route")
    def test_should_call_route_management_disabled_llm_enabled(
        self,
        mock_is_management_route,
        mock_is_llm_api_route,
        mock_is_management_disabled,
        mock_is_llm_api_disabled,
    ):
        """Test that LLM API routes work when only management routes are disabled"""

        # Mock the methods - LLM API route but management disabled, LLM API enabled
        mock_is_management_route.return_value = False
        mock_is_llm_api_route.return_value = True
        mock_is_management_disabled.return_value = True
        mock_is_llm_api_disabled.return_value = False

        # Should not raise exception since LLM API routes are enabled
        EnterpriseRouteChecks.should_call_route("/v1/chat/completions")

    @patch.object(EnterpriseRouteChecks, "is_llm_api_route_disabled")
    @patch.object(EnterpriseRouteChecks, "is_management_routes_disabled")
    @patch("litellm.proxy.auth.route_checks.RouteChecks.is_llm_api_route")
    @patch("litellm.proxy.auth.route_checks.RouteChecks.is_management_route")
    def test_should_call_route_llm_disabled_management_enabled(
        self,
        mock_is_management_route,
        mock_is_llm_api_route,
        mock_is_management_disabled,
        mock_is_llm_api_disabled,
    ):
        """Test that management routes work when only LLM API routes are disabled"""

        # Mock the methods - management route but LLM API disabled, management enabled
        mock_is_management_route.return_value = True
        mock_is_llm_api_route.return_value = False
        mock_is_management_disabled.return_value = False
        mock_is_llm_api_disabled.return_value = True

        # Should not raise exception since management routes are enabled
        EnterpriseRouteChecks.should_call_route("/config/update")


@patch("litellm.proxy.proxy_server.premium_user", True)
class TestEnterpriseRouteChecksModelListExemption:
    """Test that /models and /v1/models are exempt from DISABLE_LLM_API_ENDPOINTS"""

    @patch.object(EnterpriseRouteChecks, "is_llm_api_route_disabled")
    @patch.object(EnterpriseRouteChecks, "is_management_routes_disabled")
    @patch("litellm.proxy.auth.route_checks.RouteChecks.is_llm_api_route")
    @patch("litellm.proxy.auth.route_checks.RouteChecks.is_management_route")
    def test_models_route_allowed_when_llm_api_disabled(
        self,
        mock_is_management_route,
        mock_is_llm_api_route,
        mock_is_management_disabled,
        mock_is_llm_api_disabled,
    ):
        """Test that /models is allowed even when LLM API routes are disabled"""
        mock_is_management_route.return_value = False
        mock_is_llm_api_route.return_value = True
        mock_is_management_disabled.return_value = False
        mock_is_llm_api_disabled.return_value = True

        # Should not raise exception for /models
        EnterpriseRouteChecks.should_call_route("/models")

    @patch.object(EnterpriseRouteChecks, "is_llm_api_route_disabled")
    @patch.object(EnterpriseRouteChecks, "is_management_routes_disabled")
    @patch("litellm.proxy.auth.route_checks.RouteChecks.is_llm_api_route")
    @patch("litellm.proxy.auth.route_checks.RouteChecks.is_management_route")
    def test_v1_models_route_allowed_when_llm_api_disabled(
        self,
        mock_is_management_route,
        mock_is_llm_api_route,
        mock_is_management_disabled,
        mock_is_llm_api_disabled,
    ):
        """Test that /v1/models is allowed even when LLM API routes are disabled"""
        mock_is_management_route.return_value = False
        mock_is_llm_api_route.return_value = True
        mock_is_management_disabled.return_value = False
        mock_is_llm_api_disabled.return_value = True

        # Should not raise exception for /v1/models
        EnterpriseRouteChecks.should_call_route("/v1/models")

    @patch.object(EnterpriseRouteChecks, "is_llm_api_route_disabled")
    @patch.object(EnterpriseRouteChecks, "is_management_routes_disabled")
    @patch("litellm.proxy.auth.route_checks.RouteChecks.is_llm_api_route")
    @patch("litellm.proxy.auth.route_checks.RouteChecks.is_management_route")
    def test_chat_completions_still_blocked_when_llm_api_disabled(
        self,
        mock_is_management_route,
        mock_is_llm_api_route,
        mock_is_management_disabled,
        mock_is_llm_api_disabled,
    ):
        """Test that non-exempt LLM routes like /v1/chat/completions are still blocked"""
        mock_is_management_route.return_value = False
        mock_is_llm_api_route.return_value = True
        mock_is_management_disabled.return_value = False
        mock_is_llm_api_disabled.return_value = True

        with pytest.raises(HTTPException) as exc_info:
            EnterpriseRouteChecks.should_call_route("/v1/chat/completions")

        assert exc_info.value.status_code == 403
        assert "LLM API routes are disabled for this instance." in str(
            exc_info.value.detail
        )


class TestEnterpriseRouteChecksErrorMessages:
    """Test that error messages correctly identify which feature requires Enterprise license"""

    @patch("litellm.secret_managers.main.get_secret_bool")
    @patch("litellm.proxy.proxy_server.premium_user", False)
    def test_disable_llm_api_endpoints_error_message(self, mock_get_secret_bool):
        """
        Test that when DISABLE_LLM_API_ENDPOINTS is set without Enterprise license,
        the error message correctly mentions 'LLM API ENDPOINTS'
        """
        with patch.dict(os.environ, {"DISABLE_LLM_API_ENDPOINTS": "true"}):
            with pytest.raises(HTTPException) as exc_info:
                EnterpriseRouteChecks.is_llm_api_route_disabled()

            assert exc_info.value.status_code == 500
            assert "DISABLING LLM API ENDPOINTS is an Enterprise feature" in str(
                exc_info.value.detail
            )

    @patch("litellm.secret_managers.main.get_secret_bool")
    @patch("litellm.proxy.proxy_server.premium_user", False)
    def test_disable_admin_endpoints_error_message(self, mock_get_secret_bool):
        """
        Test that when DISABLE_ADMIN_ENDPOINTS is set without Enterprise license,
        the error message correctly mentions 'ADMIN ENDPOINTS' (not 'LLM API ENDPOINTS')

        This is a regression test for a bug where the error message incorrectly said
        'DISABLING LLM API ENDPOINTS' when the actual issue was DISABLE_ADMIN_ENDPOINTS.
        """
        with patch.dict(os.environ, {"DISABLE_ADMIN_ENDPOINTS": "true"}):
            with pytest.raises(HTTPException) as exc_info:
                EnterpriseRouteChecks.is_management_routes_disabled()

            assert exc_info.value.status_code == 500
            assert "DISABLING ADMIN ENDPOINTS is an Enterprise feature" in str(
                exc_info.value.detail
            )
            # Ensure it does NOT mention LLM API ENDPOINTS (the old buggy message)
            assert "LLM API ENDPOINTS" not in str(exc_info.value.detail)

    @patch("litellm.secret_managers.main.get_secret_bool")
    @patch("litellm.proxy.proxy_server.premium_user", True)
    def test_disable_llm_api_endpoints_with_premium_user(self, mock_get_secret_bool):
        """
        Test that premium users can use DISABLE_LLM_API_ENDPOINTS without error
        """
        mock_get_secret_bool.return_value = True
        with patch.dict(os.environ, {"DISABLE_LLM_API_ENDPOINTS": "true"}):
            # Should not raise exception for premium users
            result = EnterpriseRouteChecks.is_llm_api_route_disabled()
            assert result is True

    @patch("litellm.secret_managers.main.get_secret_bool")
    @patch("litellm.proxy.proxy_server.premium_user", True)
    def test_disable_admin_endpoints_with_premium_user(self, mock_get_secret_bool):
        """
        Test that premium users can use DISABLE_ADMIN_ENDPOINTS without error
        """
        mock_get_secret_bool.return_value = True
        with patch.dict(os.environ, {"DISABLE_ADMIN_ENDPOINTS": "true"}):
            # Should not raise exception for premium users
            result = EnterpriseRouteChecks.is_management_routes_disabled()
            assert result is True
