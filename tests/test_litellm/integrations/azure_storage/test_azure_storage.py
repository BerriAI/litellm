import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.integrations.azure_storage.azure_storage import AzureBlobStorageLogger
from litellm.types.utils import StandardLoggingPayload


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up required environment variables for Azure Storage"""
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_NAME", "test-account")
    monkeypatch.setenv("AZURE_STORAGE_FILE_SYSTEM", "test-container")
    monkeypatch.setenv("AZURE_STORAGE_TENANT_ID", "test-tenant-id")
    monkeypatch.setenv("AZURE_STORAGE_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("AZURE_STORAGE_CLIENT_SECRET", "test-client-secret")


@pytest.mark.asyncio
async def test_async_upload_payload_to_azure_blob_storage(mock_env_vars):
    """
    Test that async_upload_payload_to_azure_blob_storage correctly uploads
    a payload to Azure Blob Storage using the 3-step process (create, append, flush).
    """
    with patch(
        "litellm.integrations.azure_storage.azure_storage.get_async_httpx_client"
    ) as mock_get_client, patch(
        "litellm.llms.azure.common_utils.get_azure_ad_token_from_entra_id"
    ) as mock_get_token:
        # Create mock HTTP client
        mock_http_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.raise_for_status = AsyncMock()
        mock_http_client.put.return_value = mock_response
        mock_http_client.patch.return_value = mock_response
        mock_get_client.return_value = mock_http_client

        # Mock Azure AD token provider
        mock_token_provider = MagicMock()
        mock_token_provider.return_value = "mock-azure-ad-token"
        mock_get_token.return_value = mock_token_provider

        # Create logger instance
        logger = AzureBlobStorageLogger()

        # Set a valid token to avoid token refresh during test
        logger.azure_auth_token = "mock-azure-ad-token"
        logger.token_expiry = None  # Set to None so token refresh check passes

        # Create test payload
        test_payload: StandardLoggingPayload = {
            "id": "test-log-id-123",
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        # Call the method under test
        await logger.async_upload_payload_to_azure_blob_storage(test_payload)

        # Verify HTTP client was obtained
        mock_get_client.assert_called_once()

        # Verify the 3-step upload process was called correctly
        # Step 1: Create file
        expected_base_url = (
            "https://test-account.dfs.core.windows.net/test-container/test-log-id-123.json"
        )
        mock_http_client.put.assert_called_once()
        put_call_args = mock_http_client.put.call_args
        assert put_call_args[0][0] == f"{expected_base_url}?resource=file"
        assert put_call_args[1]["headers"]["x-ms-version"] is not None
        assert put_call_args[1]["headers"]["Authorization"] == "Bearer mock-azure-ad-token"

        # Step 2: Append data
        assert mock_http_client.patch.call_count == 2  # Called for append and flush
        append_call = mock_http_client.patch.call_args_list[0]
        assert append_call[0][0] == f"{expected_base_url}?action=append&position=0"
        assert append_call[1]["headers"]["x-ms-version"] is not None
        assert append_call[1]["headers"]["Content-Type"] == "application/json"
        assert append_call[1]["headers"]["Authorization"] == "Bearer mock-azure-ad-token"
        assert "test-log-id-123" in append_call[1]["data"]

        # Step 3: Flush data
        flush_call = mock_http_client.patch.call_args_list[1]
        assert "action=flush" in flush_call[0][0]
        assert flush_call[1]["headers"]["x-ms-version"] is not None
        assert flush_call[1]["headers"]["Authorization"] == "Bearer mock-azure-ad-token"

        # Verify raise_for_status was called on all responses
        assert mock_response.raise_for_status.call_count == 3
