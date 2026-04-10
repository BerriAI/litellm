import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.integrations.azure_storage.azure_storage import AzureBlobStorageLogger
from litellm.llms.base_llm.files.azure_blob_storage_backend import (
    AzureBlobStorageBackend,
)
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


async def _collect_async_chunks(async_iterable):
    chunks = []
    async for chunk in async_iterable:
        chunks.append(chunk)
    return chunks


@pytest.mark.asyncio
async def test_download_file_account_key_flow(mock_env_vars, monkeypatch):
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_KEY", "test-account-key")

    backend = AzureBlobStorageBackend()

    service_client = MagicMock()
    file_system_client = MagicMock()
    file_client = MagicMock()
    download_response = MagicMock()

    backend.get_service_client = AsyncMock(return_value=service_client)  # type: ignore
    service_client.get_file_system_client.return_value = file_system_client
    file_system_client.exists = AsyncMock(return_value=True)
    file_system_client.get_file_client.return_value = file_client
    file_client.download_file = AsyncMock(return_value=download_response)
    download_response.readall = AsyncMock(return_value=b"account-key-content")

    storage_url = "https://test-account.blob.core.windows.net/test-container/path/to/file.txt"
    result = await backend.download_file(storage_url)

    assert result == b"account-key-content"
    service_client.get_file_system_client.assert_called_once_with(
        file_system="test-container"
    )
    file_system_client.get_file_client.assert_called_once_with("path/to/file.txt")


@pytest.mark.asyncio
async def test_download_file_azure_ad_flow(mock_env_vars, monkeypatch):
    monkeypatch.delenv("AZURE_STORAGE_ACCOUNT_KEY", raising=False)

    backend = AzureBlobStorageBackend()
    backend.azure_auth_token = "mock-azure-ad-token"
    backend.set_valid_azure_ad_token = AsyncMock()  # type: ignore

    mock_async_httpx_client = MagicMock()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.content = b"azure-ad-content"
    mock_async_httpx_client.get = AsyncMock(return_value=mock_response)

    with patch(
        "litellm.llms.custom_httpx.http_handler.get_async_httpx_client",
        return_value=mock_async_httpx_client,
    ):
        storage_url = "https://test-account.blob.core.windows.net/test-container/path/to/file.txt"
        result = await backend.download_file(storage_url)

    assert result == b"azure-ad-content"
    backend.set_valid_azure_ad_token.assert_awaited_once()
    mock_async_httpx_client.get.assert_awaited_once()


@pytest.mark.asyncio
async def test_download_file_streaming_account_key_flow(mock_env_vars, monkeypatch):
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_KEY", "test-account-key")

    backend = AzureBlobStorageBackend()

    async def _mock_stream(*args, **kwargs):
        yield b"chunk-1"
        yield b"chunk-2"

    backend._download_file_with_account_key_streaming = _mock_stream  # type: ignore

    storage_url = "https://test-account.blob.core.windows.net/test-container/path/to/file.txt"
    chunks = await _collect_async_chunks(backend.download_file_streaming(storage_url))

    assert chunks == [b"chunk-1", b"chunk-2"]


@pytest.mark.asyncio
async def test_download_file_streaming_azure_ad_flow(mock_env_vars, monkeypatch):
    monkeypatch.delenv("AZURE_STORAGE_ACCOUNT_KEY", raising=False)

    backend = AzureBlobStorageBackend()

    async def _mock_stream(*args, **kwargs):
        yield b"ad-chunk-1"
        yield b"ad-chunk-2"

    backend._download_file_with_azure_ad_streaming = _mock_stream  # type: ignore

    storage_url = "https://test-account.blob.core.windows.net/test-container/path/to/file.txt"
    chunks = await _collect_async_chunks(backend.download_file_streaming(storage_url))

    assert chunks == [b"ad-chunk-1", b"ad-chunk-2"]
