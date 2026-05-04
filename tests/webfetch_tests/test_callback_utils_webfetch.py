"""Test callback_utils webfetch_interception initialization.

This file tests the 4 lines that register the webfetch_interception callback
in the proxy's callback initialization.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestInitializeCallbacksWebFetch:
    """Test that webfetch_interception callback is initialized correctly."""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        """Set up mocks for heavy dependencies."""
        # Mock proxy_server.prisma_client to avoid DB setup
        self.mock_prisma = MagicMock()
        monkeypatch.setattr(
            "litellm.proxy.proxy_server.prisma_client",
            self.mock_prisma,
            raising=False,
        )

    @patch("litellm.integrations.webfetch_interception.handler.WebFetchInterceptionLogger.initialize_from_proxy_config")
    def test_webfetch_interception_callback_registered(self, mock_init):
        """Test that webfetch_interception callback creates and appends logger."""
        from litellm.proxy.common_utils.callback_utils import initialize_callbacks_on_proxy

        mock_logger = MagicMock()
        mock_init.return_value = mock_logger

        litellm_settings = {
            "webfetch_interception_params": {
                "fetch_provider": "firecrawl",
                "api_key": "fc-test",
            }
        }

        result = initialize_callbacks_on_proxy(
            value=["webfetch_interception"],
            premium_user=False,
            config_file_path="/dev/null",
            litellm_settings=litellm_settings,
        )

        mock_init.assert_called_once_with(
            litellm_settings=litellm_settings,
            callback_specific_params={},
        )
        assert mock_logger in result

    @patch("litellm.integrations.webfetch_interception.handler.WebFetchInterceptionLogger.initialize_from_proxy_config")
    def test_webfetch_interception_with_callback_specific_params(self, mock_init):
        """Test with callback_specific_params containing webfetch_interception config."""
        from litellm.proxy.common_utils.callback_utils import initialize_callbacks_on_proxy

        mock_logger = MagicMock()
        mock_init.return_value = mock_logger

        callback_specific_params = {
            "webfetch_interception": {
                "fetch_provider": "firecrawl",
                "api_key": "fc-specific",
            }
        }

        result = initialize_callbacks_on_proxy(
            value=["webfetch_interception"],
            premium_user=False,
            config_file_path="/dev/null",
            litellm_settings={},
            callback_specific_params=callback_specific_params,
        )

        mock_init.assert_called_once_with(
            litellm_settings={},
            callback_specific_params=callback_specific_params,
        )
        assert mock_logger in result

    @patch("litellm.integrations.webfetch_interception.handler.WebFetchInterceptionLogger.initialize_from_proxy_config")
    def test_webfetch_interception_not_in_list(self, mock_init):
        """Test that other callbacks don't trigger webfetch initialization."""
        from litellm.proxy.common_utils.callback_utils import initialize_callbacks_on_proxy

        mock_logger = MagicMock()
        mock_init.return_value = mock_logger

        result = initialize_callbacks_on_proxy(
            value=["other_callback"],
            premium_user=False,
            config_file_path="/dev/null",
            litellm_settings={},
        )

        assert mock_logger not in result

    def test_webfetch_interception_empty_params(self):
        """Test that empty params still work."""
        from litellm.proxy.common_utils.callback_utils import initialize_callbacks_on_proxy

        with patch(
            "litellm.integrations.webfetch_interception.handler.WebFetchInterceptionLogger.initialize_from_proxy_config"
        ) as mock_init:
            mock_logger = MagicMock()
            mock_init.return_value = mock_logger

            result = initialize_callbacks_on_proxy(
                value=["webfetch_interception"],
                premium_user=False,
                config_file_path="/dev/null",
                litellm_settings={},
            )

            mock_init.assert_called_once()
            assert mock_logger in result
