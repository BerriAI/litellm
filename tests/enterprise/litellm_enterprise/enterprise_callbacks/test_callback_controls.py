import os
import sys
from unittest.mock import patch

# Add the project root to the path so we can import the modules
sys.path.insert(0, os.path.abspath("../.."))

import pytest
from litellm.types.utils import StandardCallbackDynamicParams
from enterprise.litellm_enterprise.enterprise_callbacks.callback_controls import EnterpriseCallbackControls


class TestEnterpriseCallbackControls:
    """Test suite for EnterpriseCallbackControls class."""

    def test_get_disabled_callbacks_with_none_request_headers(self):
        """
        Test that get_disabled_callbacks handles None request headers gracefully.

        This test reproduces the issue where get_proxy_server_request_headers
        could return None, causing a "'NoneType' object has no attribute 'get'" error.
        """
        # Mock the get_proxy_server_request_headers function to return None
        with patch(
            "enterprise.litellm_enterprise.enterprise_callbacks.callback_controls.get_proxy_server_request_headers"
        ) as mock_get_headers:
            mock_get_headers.return_value = None

            # Test data
            litellm_params = {"some": "data"}
            standard_callback_dynamic_params = StandardCallbackDynamicParams()

            # This should not raise an exception
            result = EnterpriseCallbackControls.get_disabled_callbacks(litellm_params, standard_callback_dynamic_params)

            # Should return None when no disabled callbacks are found
            assert result is None

    def test_get_disabled_callbacks_with_empty_request_headers(self):
        """
        Test that get_disabled_callbacks works with empty request headers.
        """
        # Mock the get_proxy_server_request_headers function to return an empty dict
        with patch(
            "enterprise.litellm_enterprise.enterprise_callbacks.callback_controls.get_proxy_server_request_headers"
        ) as mock_get_headers:
            mock_get_headers.return_value = {}

            # Test data
            litellm_params = {"some": "data"}
            standard_callback_dynamic_params = StandardCallbackDynamicParams()

            # This should not raise an exception
            result = EnterpriseCallbackControls.get_disabled_callbacks(litellm_params, standard_callback_dynamic_params)

            # Should return None when no disabled callbacks are found
            assert result is None

    def test_get_disabled_callbacks_with_disabled_header(self):
        """
        Test that get_disabled_callbacks correctly parses disabled callbacks from headers.
        """
        # Mock the get_proxy_server_request_headers function to return headers with disabled callbacks
        with patch(
            "enterprise.litellm_enterprise.enterprise_callbacks.callback_controls.get_proxy_server_request_headers"
        ) as mock_get_headers:
            mock_get_headers.return_value = {"x-litellm-disable-callbacks": "prometheus,slack"}

            # Test data
            litellm_params = {"some": "data"}
            standard_callback_dynamic_params = StandardCallbackDynamicParams()

            # This should not raise an exception
            result = EnterpriseCallbackControls.get_disabled_callbacks(litellm_params, standard_callback_dynamic_params)

            # Should return the list of disabled callbacks
            assert result is not None
            assert isinstance(result, list)
            assert "prometheus" in [cb.lower() for cb in result]
            assert "slack" in [cb.lower() for cb in result]

    def test_is_callback_disabled_dynamically_with_none_headers(self):
        """
        Test that is_callback_disabled_dynamically handles None request headers gracefully.
        """
        # Mock the get_proxy_server_request_headers function to return None
        with patch(
            "enterprise.litellm_enterprise.enterprise_callbacks.callback_controls.get_proxy_server_request_headers"
        ) as mock_get_headers:
            mock_get_headers.return_value = None

            # Mock the premium user check to return True
            with patch(
                "enterprise.litellm_enterprise.enterprise_callbacks.callback_controls.EnterpriseCallbackControls._premium_user_check"
            ) as mock_premium_check:
                mock_premium_check.return_value = True

                # Test data
                callback = "prometheus"
                litellm_params = {"some": "data"}
                standard_callback_dynamic_params = StandardCallbackDynamicParams()

                # This should not raise an exception
                result = EnterpriseCallbackControls.is_callback_disabled_dynamically(
                    callback, litellm_params, standard_callback_dynamic_params
                )

                # Should return False when no headers are present
                assert result is False

    def test_is_callback_disabled_dynamically_with_disabled_callback(self):
        """
        Test that is_callback_disabled_dynamically correctly identifies disabled callbacks.
        """
        # Mock the get_proxy_server_request_headers function to return headers with disabled callbacks
        with patch(
            "enterprise.litellm_enterprise.enterprise_callbacks.callback_controls.get_proxy_server_request_headers"
        ) as mock_get_headers:
            mock_get_headers.return_value = {"x-litellm-disable-callbacks": "prometheus,slack"}

            # Mock the premium user check to return True
            with patch(
                "enterprise.litellm_enterprise.enterprise_callbacks.callback_controls.EnterpriseCallbackControls._premium_user_check"
            ) as mock_premium_check:
                mock_premium_check.return_value = True

                # Test data
                callback = "prometheus"
                litellm_params = {"some": "data"}
                standard_callback_dynamic_params = StandardCallbackDynamicParams()

                # This should not raise an exception
                result = EnterpriseCallbackControls.is_callback_disabled_dynamically(
                    callback, litellm_params, standard_callback_dynamic_params
                )

                # Should return True for a disabled callback
                assert result is True

    def test_is_callback_disabled_dynamically_with_enabled_callback(self):
        """
        Test that is_callback_disabled_dynamically correctly identifies enabled callbacks.
        """
        # Mock the get_proxy_server_request_headers function to return headers with disabled callbacks
        with patch(
            "enterprise.litellm_enterprise.enterprise_callbacks.callback_controls.get_proxy_server_request_headers"
        ) as mock_get_headers:
            mock_get_headers.return_value = {"x-litellm-disable-callbacks": "prometheus,slack"}

            # Mock the premium user check to return True
            with patch(
                "enterprise.litellm_enterprise.enterprise_callbacks.callback_controls.EnterpriseCallbackControls._premium_user_check"
            ) as mock_premium_check:
                mock_premium_check.return_value = True

                # Test data
                callback = "langfuse"
                litellm_params = {"some": "data"}
                standard_callback_dynamic_params = StandardCallbackDynamicParams()

                # This should not raise an exception
                result = EnterpriseCallbackControls.is_callback_disabled_dynamically(
                    callback, litellm_params, standard_callback_dynamic_params
                )

                # Should return False for an enabled callback
                assert result is False


if __name__ == "__main__":
    pytest.main([__file__])
