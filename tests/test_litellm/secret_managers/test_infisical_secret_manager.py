"""
Unit tests for Infisical Secret Manager integration.

These tests verify the InfisicalSecretManager class works correctly
with mocked HTTP responses.
"""

import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest

# Mock the premium_user check at module level before imports
sys.modules["litellm.proxy.proxy_server"] = MagicMock()
sys.modules["litellm.proxy.proxy_server"].premium_user = True
sys.modules["litellm.proxy.proxy_server"].CommonProxyErrors = MagicMock()


class TestInfisicalSecretManagerUnit:
    """
    Unit tests for Infisical Secret Manager with mocked HTTP calls.
    """

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocks for all tests."""
        # Mock premium_user at the infisical module level
        with patch.dict(
            "litellm.secret_managers.infisical_secret_manager.__dict__",
            {"premium_user": True},
        ):
            yield

    @pytest.fixture
    def infisical_env_vars(self):
        """Set up environment variables for Infisical."""
        env_vars = {
            "INFISICAL_URL": "https://app.infisical.com",
            "INFISICAL_CLIENT_ID": "test-client-id",
            "INFISICAL_CLIENT_SECRET": "test-client-secret",
            "INFISICAL_PROJECT_ID": "test-project-id",
            "INFISICAL_ENVIRONMENT": "dev",
            "INFISICAL_SECRET_PATH": "/",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            yield env_vars

    @pytest.fixture
    def mock_auth_response(self):
        """Mock authentication response from Infisical."""
        return {
            "accessToken": "test-access-token-12345",
            "expiresIn": 7200,
            "tokenType": "Bearer",
        }

    @pytest.fixture
    def mock_secret_response(self):
        """Mock secret read response from Infisical."""
        return {
            "secret": {
                "id": "secret-id-123",
                "secretKey": "TEST_SECRET",
                "secretValue": "test-secret-value",
                "version": 1,
                "type": "shared",
                "environment": "dev",
                "secretPath": "/",
            }
        }

    def test_infisical_manager_initialization(self, infisical_env_vars):
        """
        Test InfisicalSecretManager initialization with environment variables.
        """
        from litellm.secret_managers.infisical_secret_manager import (
            InfisicalSecretManager,
        )

        manager = InfisicalSecretManager()

        assert manager.site_url == "https://app.infisical.com"
        assert manager.client_id == "test-client-id"
        assert manager.client_secret == "test-client-secret"
        assert manager.project_id == "test-project-id"
        assert manager.environment == "dev"
        assert manager.secret_path == "/"

    def test_infisical_manager_initialization_with_params(self, infisical_env_vars):
        """
        Test InfisicalSecretManager initialization with explicit parameters.
        """
        from litellm.secret_managers.infisical_secret_manager import (
            InfisicalSecretManager,
        )

        manager = InfisicalSecretManager(
            site_url="https://custom.infisical.com",
            client_id="custom-client-id",
            client_secret="custom-client-secret",
            project_id="custom-project-id",
            environment="prod",
            secret_path="/custom/path",
        )

        assert manager.site_url == "https://custom.infisical.com"
        assert manager.client_id == "custom-client-id"
        assert manager.client_secret == "custom-client-secret"
        assert manager.project_id == "custom-project-id"
        assert manager.environment == "prod"
        assert manager.secret_path == "/custom/path"

    def test_infisical_manager_missing_credentials(self):
        """
        Test InfisicalSecretManager raises error when credentials are missing.
        """
        # Clear Infisical-related env vars
        env_to_clear = {
            "INFISICAL_CLIENT_ID": "",
            "INFISICAL_CLIENT_SECRET": "",
        }
        with patch.dict(os.environ, env_to_clear, clear=False):
            from litellm.secret_managers.infisical_secret_manager import (
                InfisicalSecretManager,
            )

            with pytest.raises(ValueError, match="Missing Infisical credentials"):
                InfisicalSecretManager(client_id="", client_secret="")

    def test_get_secret_url(self, infisical_env_vars):
        """
        Test URL construction for secret retrieval.
        """
        from litellm.secret_managers.infisical_secret_manager import (
            InfisicalSecretManager,
        )

        manager = InfisicalSecretManager()
        url = manager._get_secret_url("TEST_SECRET")

        expected = (
            "https://app.infisical.com/api/v3/secrets/raw/TEST_SECRET"
            "?workspaceId=test-project-id"
            "&environment=dev"
            "&secretPath=/"
        )
        assert url == expected

    def test_get_secret_url_with_overrides(self, infisical_env_vars):
        """
        Test URL construction with parameter overrides.
        """
        from litellm.secret_managers.infisical_secret_manager import (
            InfisicalSecretManager,
        )

        manager = InfisicalSecretManager()
        url = manager._get_secret_url(
            "TEST_SECRET",
            project_id="override-project",
            environment="prod",
            secret_path="/custom/path",
        )

        expected = (
            "https://app.infisical.com/api/v3/secrets/raw/TEST_SECRET"
            "?workspaceId=override-project"
            "&environment=prod"
            "&secretPath=/custom/path"
        )
        assert url == expected

    @patch("litellm.secret_managers.infisical_secret_manager._get_httpx_client")
    def test_sync_authentication(
        self, mock_get_client, infisical_env_vars, mock_auth_response
    ):
        """
        Test synchronous authentication with Infisical.
        """
        from litellm.secret_managers.infisical_secret_manager import (
            InfisicalSecretManager,
        )

        # Mock the HTTP response
        mock_response = Mock()
        mock_response.json.return_value = mock_auth_response
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_get_client.return_value = mock_client

        manager = InfisicalSecretManager()
        token = manager._authenticate()

        assert token == "test-access-token-12345"
        mock_client.post.assert_called_once()

    @patch("litellm.secret_managers.infisical_secret_manager._get_httpx_client")
    def test_sync_read_secret(
        self,
        mock_get_client,
        infisical_env_vars,
        mock_auth_response,
        mock_secret_response,
    ):
        """
        Test synchronous secret reading from Infisical.
        """
        from litellm.secret_managers.infisical_secret_manager import (
            InfisicalSecretManager,
        )

        # Mock auth response
        auth_response = Mock()
        auth_response.json.return_value = mock_auth_response
        auth_response.raise_for_status = Mock()

        # Mock secret response
        secret_response = Mock()
        secret_response.json.return_value = mock_secret_response
        secret_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.post.return_value = auth_response
        mock_client.get.return_value = secret_response
        mock_get_client.return_value = mock_client

        manager = InfisicalSecretManager()
        secret_value = manager.sync_read_secret("TEST_SECRET")

        assert secret_value == "test-secret-value"

    @patch("litellm.secret_managers.infisical_secret_manager._get_httpx_client")
    def test_sync_read_secret_not_found(
        self,
        mock_get_client,
        infisical_env_vars,
        mock_auth_response,
    ):
        """
        Test synchronous secret reading when secret is not found.
        """
        from litellm.secret_managers.infisical_secret_manager import (
            InfisicalSecretManager,
        )

        # Mock auth response
        auth_response = Mock()
        auth_response.json.return_value = mock_auth_response
        auth_response.raise_for_status = Mock()

        # Mock 404 response
        error_response = Mock()
        error_response.status_code = 404
        http_error = httpx.HTTPStatusError(
            message="Not Found",
            request=Mock(),
            response=error_response,
        )

        mock_client = Mock()
        mock_client.post.return_value = auth_response
        mock_client.get.side_effect = http_error
        mock_get_client.return_value = mock_client

        manager = InfisicalSecretManager()
        secret_value = manager.sync_read_secret("NONEXISTENT_SECRET")

        assert secret_value is None

    @pytest.mark.asyncio
    @patch("litellm.secret_managers.infisical_secret_manager.get_async_httpx_client")
    async def test_async_read_secret(
        self,
        mock_get_async_client,
        infisical_env_vars,
        mock_auth_response,
        mock_secret_response,
    ):
        """
        Test asynchronous secret reading from Infisical.
        """
        from litellm.secret_managers.infisical_secret_manager import (
            InfisicalSecretManager,
        )

        # Mock async client
        mock_client = AsyncMock()

        # Mock auth response
        auth_response = Mock()
        auth_response.json.return_value = mock_auth_response
        auth_response.raise_for_status = Mock()

        # Mock secret response
        secret_response = Mock()
        secret_response.json.return_value = mock_secret_response
        secret_response.raise_for_status = Mock()

        mock_client.post.return_value = auth_response
        mock_client.get.return_value = secret_response
        mock_get_async_client.return_value = mock_client

        manager = InfisicalSecretManager()
        secret_value = await manager.async_read_secret("TEST_SECRET")

        assert secret_value == "test-secret-value"

    @pytest.mark.asyncio
    @patch("litellm.secret_managers.infisical_secret_manager.get_async_httpx_client")
    async def test_async_write_secret_create(
        self,
        mock_get_async_client,
        infisical_env_vars,
        mock_auth_response,
    ):
        """
        Test asynchronous secret creation in Infisical.
        """
        from litellm.secret_managers.infisical_secret_manager import (
            InfisicalSecretManager,
        )

        # Mock async client
        mock_client = AsyncMock()

        # Mock auth response
        auth_response = Mock()
        auth_response.json.return_value = mock_auth_response
        auth_response.raise_for_status = Mock()

        # Mock create response
        create_response = Mock()
        create_response.json.return_value = {
            "secret": {
                "id": "new-secret-id",
                "secretKey": "NEW_SECRET",
                "secretValue": "new-secret-value",
            }
        }
        create_response.raise_for_status = Mock()

        mock_client.post.side_effect = [auth_response, create_response]
        mock_get_async_client.return_value = mock_client

        manager = InfisicalSecretManager()
        result = await manager.async_write_secret(
            secret_name="NEW_SECRET",
            secret_value="new-secret-value",
            description="Test description",
        )

        assert result["status"] == "success"
        assert result["operation"] == "create"
        assert result["secret_name"] == "NEW_SECRET"

    @pytest.mark.asyncio
    @patch("litellm.secret_managers.infisical_secret_manager.get_async_httpx_client")
    async def test_async_delete_secret(
        self,
        mock_get_async_client,
        infisical_env_vars,
        mock_auth_response,
    ):
        """
        Test asynchronous secret deletion from Infisical.
        """
        from litellm.secret_managers.infisical_secret_manager import (
            InfisicalSecretManager,
        )

        # Mock async client
        mock_client = AsyncMock()

        # Mock auth response
        auth_response = Mock()
        auth_response.json.return_value = mock_auth_response
        auth_response.raise_for_status = Mock()

        # Mock delete response
        delete_response = Mock()
        delete_response.json.return_value = {}
        delete_response.text = ""
        delete_response.raise_for_status = Mock()

        mock_client.post.return_value = auth_response
        mock_client.request.return_value = delete_response
        mock_get_async_client.return_value = mock_client

        manager = InfisicalSecretManager()
        result = await manager.async_delete_secret(secret_name="TEST_SECRET")

        assert result["status"] == "success"
        assert "deleted successfully" in result["message"]

    @patch("litellm.secret_managers.infisical_secret_manager._get_httpx_client")
    def test_authentication_caching(
        self, mock_get_client, infisical_env_vars, mock_auth_response
    ):
        """
        Test that authentication tokens are cached.
        """
        from litellm.secret_managers.infisical_secret_manager import (
            InfisicalSecretManager,
        )

        # Mock the HTTP response
        mock_response = Mock()
        mock_response.json.return_value = mock_auth_response
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_get_client.return_value = mock_client

        manager = InfisicalSecretManager()

        # First call should authenticate
        token1 = manager._authenticate()
        assert token1 == "test-access-token-12345"

        # Second call should use cached token
        token2 = manager._authenticate()
        assert token2 == "test-access-token-12345"

        # HTTP post should only be called once (for first auth)
        assert mock_client.post.call_count == 1

    @patch("litellm.secret_managers.infisical_secret_manager._get_httpx_client")
    def test_secret_value_caching(
        self,
        mock_get_client,
        infisical_env_vars,
        mock_auth_response,
        mock_secret_response,
    ):
        """
        Test that secret values are cached.
        """
        from litellm.secret_managers.infisical_secret_manager import (
            InfisicalSecretManager,
        )

        # Mock auth response
        auth_response = Mock()
        auth_response.json.return_value = mock_auth_response
        auth_response.raise_for_status = Mock()

        # Mock secret response
        secret_response = Mock()
        secret_response.json.return_value = mock_secret_response
        secret_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.post.return_value = auth_response
        mock_client.get.return_value = secret_response
        mock_get_client.return_value = mock_client

        manager = InfisicalSecretManager()

        # First read
        value1 = manager.sync_read_secret("TEST_SECRET")
        assert value1 == "test-secret-value"

        # Second read should use cache
        value2 = manager.sync_read_secret("TEST_SECRET")
        assert value2 == "test-secret-value"

        # HTTP get should only be called once
        assert mock_client.get.call_count == 1


class TestInfisicalKeyManagementSystem:
    """
    Test the KeyManagementSystem enum includes Infisical.
    """

    def test_infisical_in_key_management_system(self):
        """
        Test that INFISICAL is a valid KeyManagementSystem value.
        """
        from litellm.types.secret_managers.main import KeyManagementSystem

        assert hasattr(KeyManagementSystem, "INFISICAL")
        assert KeyManagementSystem.INFISICAL.value == "infisical"


class TestSecretManagerHandler:
    """
    Test the secret_manager_handler.py integration with Infisical.
    """

    def test_infisical_handler_integration(self):
        """
        Test that get_secret_from_manager handles Infisical correctly.
        """
        from litellm.secret_managers.secret_manager_handler import get_secret_from_manager
        from litellm.types.secret_managers.main import KeyManagementSystem

        # Create a mock Infisical client
        mock_client = Mock()
        mock_client.sync_read_secret.return_value = "test-secret-value"

        result = get_secret_from_manager(
            client=mock_client,
            key_manager=KeyManagementSystem.INFISICAL.value,
            secret_name="TEST_SECRET",
        )

        assert result == "test-secret-value"
        mock_client.sync_read_secret.assert_called_once_with(secret_name="TEST_SECRET")

    def test_infisical_handler_secret_not_found(self):
        """
        Test that get_secret_from_manager raises error when Infisical secret not found.
        """
        from litellm.secret_managers.secret_manager_handler import get_secret_from_manager
        from litellm.types.secret_managers.main import KeyManagementSystem

        # Create a mock Infisical client that returns None
        mock_client = Mock()
        mock_client.sync_read_secret.return_value = None

        with pytest.raises(ValueError, match="No secret found in Infisical"):
            get_secret_from_manager(
                client=mock_client,
                key_manager=KeyManagementSystem.INFISICAL.value,
                secret_name="NONEXISTENT_SECRET",
            )
