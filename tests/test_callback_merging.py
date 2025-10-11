"""
Test callback merging behavior when adding callbacks through UI
Tests for issue #12118 - ensuring callbacks from config are preserved when adding through UI
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.abspath("../.."))

from litellm.proxy.proxy_server import app, ProxyConfig


class TestCallbackMerging:
    """Test suite for callback merging behavior"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        # Create a test client
        self.client = TestClient(app)

        # Mock authentication
        self.mock_auth = MagicMock()
        self.mock_auth.user_role = "proxy_admin"
        self.mock_auth.user_id = "test_user"

        yield

        # Cleanup
        self.client = None

    @pytest.mark.asyncio
    async def test_config_update_preserves_existing_callbacks(self):
        """Test that updating callbacks via UI preserves existing callbacks from config"""

        # Mock the proxy config instance
        mock_proxy_config = ProxyConfig()

        # Mock existing config with callbacks
        existing_config = {
            "litellm_settings": {"success_callback": ["langfuse", "datadog"]}
        }

        # Mock methods
        async def mock_get_config():
            return existing_config

        async def mock_save_config(new_config):
            pass

        async def mock_add_deployment(prisma_client, proxy_logging_obj):
            pass

        mock_proxy_config.get_config = mock_get_config
        mock_proxy_config.save_config = mock_save_config
        mock_proxy_config.add_deployment = mock_add_deployment

        with patch("litellm.proxy.proxy_server.proxy_config", mock_proxy_config):
            with patch("litellm.proxy.proxy_server.prisma_client", MagicMock()):
                with patch("litellm.proxy.proxy_server.store_model_in_db", True):
                    with patch(
                        "litellm.proxy.proxy_server.user_api_key_auth",
                        return_value=self.mock_auth,
                    ):
                        # Update config with new callback
                        update_data = {
                            "litellm_settings": {"success_callback": ["prometheus"]}
                        }

                        # Mock save_config to capture the saved config
                        saved_config = None

                        async def capture_save_config(new_config):
                            nonlocal saved_config
                            saved_config = new_config

                        mock_proxy_config.save_config = capture_save_config

                        response = self.client.post(
                            "/config/update",
                            json=update_data,
                            headers={"Authorization": "Bearer test_key"},
                        )

                        assert response.status_code == 200

                        # Verify callbacks were merged, not replaced
                        assert saved_config is not None
                        assert "litellm_settings" in saved_config
                        assert "success_callback" in saved_config["litellm_settings"]

                        # Check that all callbacks are present
                        callbacks = saved_config["litellm_settings"]["success_callback"]
                        assert "langfuse" in callbacks
                        assert "datadog" in callbacks
                        assert "prometheus" in callbacks
                        assert len(callbacks) == 3  # No duplicates

    @pytest.mark.asyncio
    async def test_config_update_handles_empty_existing_callbacks(self):
        """Test that updating callbacks works when no existing callbacks are present"""

        # Mock the proxy config instance
        mock_proxy_config = ProxyConfig()

        # Mock existing config without callbacks
        existing_config = {"litellm_settings": {}}

        # Mock methods
        async def mock_get_config():
            return existing_config

        async def mock_save_config(new_config):
            pass

        async def mock_add_deployment(prisma_client, proxy_logging_obj):
            pass

        mock_proxy_config.get_config = mock_get_config
        mock_proxy_config.save_config = mock_save_config
        mock_proxy_config.add_deployment = mock_add_deployment

        with patch("litellm.proxy.proxy_server.proxy_config", mock_proxy_config):
            with patch("litellm.proxy.proxy_server.prisma_client", MagicMock()):
                with patch("litellm.proxy.proxy_server.store_model_in_db", True):
                    with patch(
                        "litellm.proxy.proxy_server.user_api_key_auth",
                        return_value=self.mock_auth,
                    ):
                        # Update config with new callback
                        update_data = {
                            "litellm_settings": {"success_callback": ["prometheus"]}
                        }

                        # Mock save_config to capture the saved config
                        saved_config = None

                        async def capture_save_config(new_config):
                            nonlocal saved_config
                            saved_config = new_config

                        mock_proxy_config.save_config = capture_save_config

                        response = self.client.post(
                            "/config/update",
                            json=update_data,
                            headers={"Authorization": "Bearer test_key"},
                        )

                        assert response.status_code == 200

                        # Verify callback was added
                        assert saved_config is not None
                        assert "litellm_settings" in saved_config
                        assert "success_callback" in saved_config["litellm_settings"]

                        callbacks = saved_config["litellm_settings"]["success_callback"]
                        assert "prometheus" in callbacks
                        assert len(callbacks) == 1

    @pytest.mark.asyncio
    async def test_config_update_deduplicates_callbacks(self):
        """Test that duplicate callbacks are deduplicated when merging"""

        # Mock the proxy config instance
        mock_proxy_config = ProxyConfig()

        # Mock existing config with callbacks
        existing_config = {
            "litellm_settings": {"success_callback": ["langfuse", "datadog"]}
        }

        # Mock methods
        async def mock_get_config():
            return existing_config

        async def mock_save_config(new_config):
            pass

        async def mock_add_deployment(prisma_client, proxy_logging_obj):
            pass

        mock_proxy_config.get_config = mock_get_config
        mock_proxy_config.save_config = mock_save_config
        mock_proxy_config.add_deployment = mock_add_deployment

        with patch("litellm.proxy.proxy_server.proxy_config", mock_proxy_config):
            with patch("litellm.proxy.proxy_server.prisma_client", MagicMock()):
                with patch("litellm.proxy.proxy_server.store_model_in_db", True):
                    with patch(
                        "litellm.proxy.proxy_server.user_api_key_auth",
                        return_value=self.mock_auth,
                    ):
                        # Update config with duplicate callback
                        update_data = {
                            "litellm_settings": {
                                "success_callback": [
                                    "langfuse",
                                    "prometheus",
                                ]  # langfuse is duplicate
                            }
                        }

                        # Mock save_config to capture the saved config
                        saved_config = None

                        async def capture_save_config(new_config):
                            nonlocal saved_config
                            saved_config = new_config

                        mock_proxy_config.save_config = capture_save_config

                        response = self.client.post(
                            "/config/update",
                            json=update_data,
                            headers={"Authorization": "Bearer test_key"},
                        )

                        assert response.status_code == 200

                        # Verify callbacks were merged and deduplicated
                        assert saved_config is not None
                        callbacks = saved_config["litellm_settings"]["success_callback"]

                        # Check that all unique callbacks are present
                        assert "langfuse" in callbacks
                        assert "datadog" in callbacks
                        assert "prometheus" in callbacks

                        # Check no duplicates
                        assert callbacks.count("langfuse") == 1
                        assert len(callbacks) == 3

    @pytest.mark.asyncio
    async def test_config_update_handles_non_list_callbacks(self):
        """Test that non-list callback values are handled gracefully"""

        # Mock the proxy config instance
        mock_proxy_config = ProxyConfig()

        # Mock existing config with non-list callback value
        existing_config = {
            "litellm_settings": {
                "success_callback": "langfuse"  # String instead of list
            }
        }

        # Mock methods
        async def mock_get_config():
            return existing_config

        async def mock_save_config(new_config):
            pass

        async def mock_add_deployment(prisma_client, proxy_logging_obj):
            pass

        mock_proxy_config.get_config = mock_get_config
        mock_proxy_config.save_config = mock_save_config
        mock_proxy_config.add_deployment = mock_add_deployment

        with patch("litellm.proxy.proxy_server.proxy_config", mock_proxy_config):
            with patch("litellm.proxy.proxy_server.prisma_client", MagicMock()):
                with patch("litellm.proxy.proxy_server.store_model_in_db", True):
                    with patch(
                        "litellm.proxy.proxy_server.user_api_key_auth",
                        return_value=self.mock_auth,
                    ):
                        # Update config with new callback
                        update_data = {
                            "litellm_settings": {"success_callback": ["prometheus"]}
                        }

                        # Mock save_config to capture the saved config
                        saved_config = None

                        async def capture_save_config(new_config):
                            nonlocal saved_config
                            saved_config = new_config

                        mock_proxy_config.save_config = capture_save_config

                        response = self.client.post(
                            "/config/update",
                            json=update_data,
                            headers={"Authorization": "Bearer test_key"},
                        )

                        assert response.status_code == 200

                        # Verify callback was added (existing non-list was ignored)
                        assert saved_config is not None
                        callbacks = saved_config["litellm_settings"]["success_callback"]
                        assert "prometheus" in callbacks
                        assert len(callbacks) == 1  # Only the new callback


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
