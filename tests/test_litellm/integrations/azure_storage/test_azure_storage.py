import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.integrations.azure_storage.azure_storage import (
    AzureBlobStorageLogger,
    _cached_credential_chain_token_provider,
)
from litellm.types.secret_managers.get_azure_ad_token_provider import AzureCredentialType
from litellm.types.utils import StandardLoggingPayload


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up required environment variables for Azure Storage"""
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_NAME", "test-account")
    monkeypatch.setenv("AZURE_STORAGE_FILE_SYSTEM", "test-container")
    monkeypatch.setenv("AZURE_STORAGE_TENANT_ID", "test-tenant-id")
    monkeypatch.setenv("AZURE_STORAGE_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("AZURE_STORAGE_CLIENT_SECRET", "test-client-secret")
    monkeypatch.delenv("AZURE_STORAGE_ENDPOINT_SUFFIX", raising=False)


@pytest.fixture
def mock_gov_env_vars(mock_env_vars, monkeypatch):
    """Point the logger at an Azure Government storage account"""
    monkeypatch.setenv("AZURE_STORAGE_ENDPOINT_SUFFIX", "core.usgovcloudapi.net")


@pytest.fixture
def workload_identity_env_vars(monkeypatch):
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_NAME", "test-account")
    monkeypatch.setenv("AZURE_STORAGE_FILE_SYSTEM", "test-container")
    for unset in (
        "AZURE_STORAGE_TENANT_ID",
        "AZURE_STORAGE_CLIENT_ID",
        "AZURE_STORAGE_CLIENT_SECRET",
        "AZURE_STORAGE_ACCOUNT_KEY",
        "AZURE_STORAGE_ENDPOINT_SUFFIX",
        "AZURE_CLIENT_SECRET",
        "AZURE_CREDENTIAL",
        "AZURE_SCOPE",
    ):
        monkeypatch.delenv(unset, raising=False)
    monkeypatch.setenv("AZURE_CLIENT_ID", "workload-identity-client-id")
    monkeypatch.setenv("AZURE_TENANT_ID", "workload-identity-tenant-id")
    monkeypatch.setenv("AZURE_FEDERATED_TOKEN_FILE", "/var/run/secrets/azure/tokens/azure-identity-token")


@pytest.mark.asyncio
async def test_async_upload_payload_to_azure_blob_storage(mock_env_vars):
    """
    Test that async_upload_payload_to_azure_blob_storage correctly uploads
    a payload to Azure Blob Storage using the 3-step process (create, append, flush).
    """
    with (
        patch("litellm.integrations.azure_storage.azure_storage.get_async_httpx_client") as mock_get_client,
        patch("litellm.integrations.azure_storage.azure_storage.get_azure_ad_token_from_entra_id") as mock_get_token,
    ):
        # Create mock HTTP client
        mock_http_client = AsyncMock()
        mock_response = MagicMock()
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
        expected_base_url = "https://test-account.dfs.core.windows.net/test-container/test-log-id-123.json"
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


@pytest.mark.asyncio
async def test_async_upload_payload_uses_configured_endpoint_suffix(mock_gov_env_vars):
    """
    AZURE_STORAGE_ENDPOINT_SUFFIX must reach the Entra-ID REST upload path so a
    sovereign-cloud account is addressed instead of the commercial dfs host.
    """
    with patch("litellm.integrations.azure_storage.azure_storage.get_async_httpx_client") as mock_get_client:
        mock_http_client = AsyncMock()
        mock_response = MagicMock()
        mock_http_client.put.return_value = mock_response
        mock_http_client.patch.return_value = mock_response
        mock_get_client.return_value = mock_http_client

        logger = AzureBlobStorageLogger()
        logger.azure_auth_token = "mock-azure-ad-token"
        logger.token_expiry = None

        test_payload: StandardLoggingPayload = {"id": "gov-log-id"}

        await logger.async_upload_payload_to_azure_blob_storage(test_payload)

        expected_base_url = "https://test-account.dfs.core.usgovcloudapi.net/test-container/gov-log-id.json"
        assert mock_http_client.put.call_args[0][0] == f"{expected_base_url}?resource=file"
        assert mock_http_client.patch.call_args_list[0][0][0] == f"{expected_base_url}?action=append&position=0"
        assert mock_http_client.patch.call_args_list[1][0][0].startswith(f"{expected_base_url}?action=flush")


@pytest.mark.asyncio
async def test_service_client_uses_configured_endpoint_suffix(mock_gov_env_vars):
    """
    The account key path builds its own account_url; the Azure SDK derives the blob
    host from it, so the suffix has to be applied here too.
    """
    fake_aio_module = MagicMock()

    with patch.dict(sys.modules, {"azure.storage.filedatalake.aio": fake_aio_module}):
        logger = AzureBlobStorageLogger()
        await logger.get_service_client()

    assert (
        fake_aio_module.DataLakeServiceClient.call_args.kwargs["account_url"]
        == "https://test-account.dfs.core.usgovcloudapi.net"
    )


@pytest.mark.asyncio
async def test_upload_authenticates_through_the_credential_chain_under_workload_identity(
    workload_identity_env_vars,
):
    build_provider = MagicMock(return_value=lambda: "workload-identity-token")
    with patch(  # test-quality-ok: the REST upload path builds its own client, so the header assertions need this seam
        "litellm.integrations.azure_storage.azure_storage.get_async_httpx_client"
    ) as mock_get_client:
        mock_http_client = AsyncMock()
        mock_http_client.put.return_value = MagicMock()
        mock_http_client.patch.return_value = MagicMock()
        mock_get_client.return_value = mock_http_client

        logger = AzureBlobStorageLogger(build_credential_chain_token_provider=build_provider)
        await logger.async_upload_payload_to_azure_blob_storage({"id": "wif-log-id"})

    build_provider.assert_called_once_with()
    assert logger.azure_auth_token == "workload-identity-token"
    sent_headers = [mock_http_client.put.call_args[1]["headers"]] + [
        call[1]["headers"] for call in mock_http_client.patch.call_args_list
    ]
    assert len(sent_headers) == 3
    assert all(headers["Authorization"] == "Bearer workload-identity-token" for headers in sent_headers)


def test_default_chain_provider_is_storage_scoped_and_built_once_per_process():
    _cached_credential_chain_token_provider.cache_clear()
    with (
        patch(  # test-quality-ok: pins the default factory's scope and credential args; the real builder would import azure-identity
            "litellm.integrations.azure_storage.azure_storage.get_azure_ad_token_provider",
            return_value=lambda: "chain-token",
        ) as mock_builder
    ):
        first = _cached_credential_chain_token_provider()
        second = _cached_credential_chain_token_provider()
    _cached_credential_chain_token_provider.cache_clear()

    assert first is second
    assert first() == "chain-token"
    mock_builder.assert_called_once_with(
        azure_scope="https://storage.azure.com/.default",
        azure_credential=AzureCredentialType.DefaultAzureCredential,
    )


@pytest.mark.asyncio
async def test_chain_tokens_are_read_from_the_provider_on_every_refresh(
    workload_identity_env_vars,
):
    provider = MagicMock(side_effect=["chain-token-1", "chain-token-2"])
    logger = AzureBlobStorageLogger(build_credential_chain_token_provider=MagicMock(return_value=provider))
    await logger.set_valid_azure_ad_token()
    first_token = logger.azure_auth_token
    await logger.set_valid_azure_ad_token()

    assert first_token == "chain-token-1"
    assert logger.azure_auth_token == "chain-token-2"
    assert provider.call_count == 2


@pytest.mark.asyncio
async def test_empty_string_service_principal_vars_still_use_the_credential_chain(
    workload_identity_env_vars, monkeypatch
):
    for name in ("AZURE_STORAGE_TENANT_ID", "AZURE_STORAGE_CLIENT_ID", "AZURE_STORAGE_CLIENT_SECRET"):
        monkeypatch.setenv(name, "")

    logger = AzureBlobStorageLogger(
        build_credential_chain_token_provider=MagicMock(return_value=lambda: "workload-identity-token")
    )
    await logger.set_valid_azure_ad_token()

    assert logger.azure_auth_token == "workload-identity-token"


@pytest.mark.asyncio
async def test_client_secret_auth_still_uses_the_storage_scoped_service_principal(mock_env_vars):
    build_provider = MagicMock()
    with (
        patch(  # test-quality-ok: the assertion is the scope handed to the shared entra-id helper, which has no fakeable boundary short of calling Entra
            "litellm.integrations.azure_storage.azure_storage.get_azure_ad_token_from_entra_id",
            return_value=lambda: "client-secret-token",
        ) as mock_entra_id
    ):
        logger = AzureBlobStorageLogger(build_credential_chain_token_provider=build_provider)
        await logger.set_valid_azure_ad_token()

    assert logger.azure_auth_token == "client-secret-token"
    build_provider.assert_not_called()
    assert mock_entra_id.call_args.kwargs == {
        "tenant_id": "test-tenant-id",
        "client_id": "test-client-id",
        "client_secret": "test-client-secret",
        "scope": "https://storage.azure.com/.default",
    }


@pytest.mark.parametrize(
    "missing_var",
    ["AZURE_STORAGE_TENANT_ID", "AZURE_STORAGE_CLIENT_ID", "AZURE_STORAGE_CLIENT_SECRET"],
)
@pytest.mark.asyncio
async def test_partially_configured_service_principal_still_names_the_missing_variable(
    mock_env_vars, monkeypatch, missing_var
):
    monkeypatch.delenv(missing_var)

    build_provider = MagicMock()
    logger = AzureBlobStorageLogger(build_credential_chain_token_provider=build_provider)
    with pytest.raises(ValueError, match=f"Missing required environment variable: {missing_var}"):
        await logger.set_valid_azure_ad_token()

    build_provider.assert_not_called()


@pytest.mark.asyncio
async def test_account_key_auth_never_requests_a_token(workload_identity_env_vars, monkeypatch):
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_KEY", "dGVzdC1rZXk=")

    file_client = MagicMock()
    file_client.create_file = AsyncMock()
    file_client.append_data = AsyncMock()
    file_client.flush_data = AsyncMock()
    directory_client = MagicMock()
    directory_client.exists = AsyncMock(return_value=True)
    directory_client.get_file_client = MagicMock(return_value=file_client)
    file_system_client = MagicMock()
    file_system_client.get_directory_client = MagicMock(return_value=directory_client)
    service_client = MagicMock()
    service_client.get_file_system_client = MagicMock(return_value=file_system_client)
    fake_aio_module = MagicMock()
    fake_aio_module.DataLakeServiceClient = MagicMock(return_value=service_client)

    build_provider = MagicMock()
    with patch.dict(sys.modules, {"azure.storage.filedatalake.aio": fake_aio_module}):
        logger = AzureBlobStorageLogger(build_credential_chain_token_provider=build_provider)
        await logger.async_upload_payload_to_azure_blob_storage({"id": "account-key-log-id"})

    build_provider.assert_not_called()
    assert logger.azure_auth_token is None
    file_client.flush_data.assert_awaited_once()
    assert fake_aio_module.DataLakeServiceClient.call_args.kwargs["credential"] == "dGVzdC1rZXk="


@pytest.mark.asyncio
async def test_service_client_defaults_to_commercial_endpoint(mock_env_vars):
    """Unset AZURE_STORAGE_ENDPOINT_SUFFIX keeps the pre-existing commercial host"""
    fake_aio_module = MagicMock()

    with patch.dict(sys.modules, {"azure.storage.filedatalake.aio": fake_aio_module}):
        logger = AzureBlobStorageLogger()
        await logger.get_service_client()

    assert (
        fake_aio_module.DataLakeServiceClient.call_args.kwargs["account_url"]
        == "https://test-account.dfs.core.windows.net"
    )
