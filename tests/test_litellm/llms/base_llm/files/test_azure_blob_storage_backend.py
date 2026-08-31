import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.llms.base_llm.files.azure_blob_storage_backend import (
    AzureBlobStorageBackend,
)

GOV_SUFFIX = "core.usgovcloudapi.net"


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Azure AD (no account key) configuration for the files backend"""
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_NAME", "test-account")
    monkeypatch.setenv("AZURE_STORAGE_FILE_SYSTEM", "test-container")
    monkeypatch.setenv("AZURE_STORAGE_TENANT_ID", "test-tenant-id")
    monkeypatch.setenv("AZURE_STORAGE_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("AZURE_STORAGE_CLIENT_SECRET", "test-client-secret")
    monkeypatch.delenv("AZURE_STORAGE_ACCOUNT_KEY", raising=False)
    monkeypatch.delenv("AZURE_STORAGE_ENDPOINT_SUFFIX", raising=False)


@pytest.fixture
def mock_gov_env_vars(mock_env_vars, monkeypatch):
    monkeypatch.setenv("AZURE_STORAGE_ENDPOINT_SUFFIX", GOV_SUFFIX)


def _make_backend() -> AzureBlobStorageBackend:
    backend = AzureBlobStorageBackend()
    backend.azure_auth_token = "mock-azure-ad-token"
    backend.token_expiry = None
    return backend


def _mock_upload_client() -> AsyncMock:
    client = AsyncMock()
    response = MagicMock()
    client.put = AsyncMock(return_value=response)
    client.patch = AsyncMock(return_value=response)
    return client


@pytest.mark.parametrize(
    "env_fixture, expected_suffix",
    [("mock_env_vars", "core.windows.net"), ("mock_gov_env_vars", GOV_SUFFIX)],
)
@pytest.mark.asyncio
async def test_upload_file_with_azure_ad_honors_endpoint_suffix(request, env_fixture, expected_suffix):
    """
    The REST upload targets the dfs host and the returned handle is a blob URL, so both
    have to follow AZURE_STORAGE_ENDPOINT_SUFFIX or a sovereign-cloud account is unreachable.
    """
    request.getfixturevalue(env_fixture)
    client = _mock_upload_client()

    with patch(
        "litellm.llms.custom_httpx.http_handler.get_async_httpx_client",
        return_value=client,
    ):
        backend = _make_backend()
        storage_url = await backend.upload_file(
            file_content=b"hello",
            filename="report.json",
            content_type="application/json",
            path_prefix="logs",
            file_naming_strategy="original_filename",
        )

    expected_dfs = f"https://test-account.dfs.{expected_suffix}/test-container/logs/report.json"
    assert client.put.call_args[0][0] == f"{expected_dfs}?resource=file"
    assert client.patch.call_args_list[0][0][0] == f"{expected_dfs}?action=append&position=0"
    assert storage_url == f"https://test-account.blob.{expected_suffix}/test-container/logs/report.json"


@pytest.mark.parametrize(
    "env_fixture, expected_suffix",
    [("mock_env_vars", "core.windows.net"), ("mock_gov_env_vars", GOV_SUFFIX)],
)
@pytest.mark.asyncio
async def test_download_file_honors_endpoint_suffix(request, env_fixture, expected_suffix):
    """
    download_file both validates and splits the stored blob URL on the host, so a
    sovereign-cloud URL must parse and round-trip back to the same host.
    """
    request.getfixturevalue(env_fixture)
    response = MagicMock()
    response.content = b"file-bytes"
    client = AsyncMock()
    client.get = AsyncMock(return_value=response)

    storage_url = f"https://test-account.blob.{expected_suffix}/test-container/logs/report.json"

    with patch(
        "litellm.llms.custom_httpx.http_handler.get_async_httpx_client",
        return_value=client,
    ):
        backend = _make_backend()
        content = await backend.download_file(storage_url)

    assert content == b"file-bytes"
    assert client.get.call_args[0][0] == storage_url


@pytest.mark.asyncio
async def test_download_file_accepts_url_persisted_before_the_suffix_was_set(mock_gov_env_vars):
    """
    storage_url is persisted in the managed files table while the suffix is process config,
    so rows written before the suffix was configured must still resolve. Only the path after
    the container is taken from the stored URL; the host comes from the current config.
    """
    response = MagicMock()
    response.content = b"file-bytes"
    client = AsyncMock()
    client.get = AsyncMock(return_value=response)

    with patch(
        "litellm.llms.custom_httpx.http_handler.get_async_httpx_client",
        return_value=client,
    ):
        backend = _make_backend()
        content = await backend.download_file(
            "https://old-account.blob.core.windows.net/old-container/logs/report.json"
        )

    assert content == b"file-bytes"
    assert (
        client.get.call_args[0][0]
        == f"https://test-account.blob.{GOV_SUFFIX}/test-container/logs/report.json"
    )


@pytest.mark.parametrize(
    "storage_url",
    [
        "https://example-bucket.s3.amazonaws.com/container/report.json",
        "https://example.com/download?u=.blob.core.windows.net/container/report.json",
        "mygovacct.blob.core.windows.net/container/report.json",
    ],
    ids=["other-provider", "blob-host-only-in-query", "no-scheme"],
)
@pytest.mark.asyncio
async def test_download_file_rejects_url_whose_host_is_not_an_azure_blob_host(mock_env_vars, storage_url):
    """
    The host is checked on the parsed hostname, so a blob host appearing anywhere else in the
    string no longer passes. No first-party producer emits these, and rejecting beats issuing a
    request built from a mis-split path.
    """
    client = AsyncMock()

    with patch(
        "litellm.llms.custom_httpx.http_handler.get_async_httpx_client",
        return_value=client,
    ):
        backend = _make_backend()

        with pytest.raises(ValueError, match="Invalid Azure Blob Storage URL"):
            await backend.download_file(storage_url)

    client.get.assert_not_called()


@pytest.mark.asyncio
async def test_download_file_drops_query_string_from_the_stored_url(mock_env_vars):
    """A query string on the stored URL is not part of the blob path and must not reach the request"""
    response = MagicMock()
    response.content = b"file-bytes"
    client = AsyncMock()
    client.get = AsyncMock(return_value=response)

    with patch(
        "litellm.llms.custom_httpx.http_handler.get_async_httpx_client",
        return_value=client,
    ):
        backend = _make_backend()
        await backend.download_file(
            "https://test-account.blob.core.windows.net/test-container/logs/report.json?sig=redacted&se=2026"
        )

    assert (
        client.get.call_args[0][0]
        == "https://test-account.blob.core.windows.net/test-container/logs/report.json"
    )


@pytest.mark.parametrize(
    "env_fixture, expected_suffix",
    [("mock_env_vars", "core.windows.net"), ("mock_gov_env_vars", GOV_SUFFIX)],
)
@pytest.mark.asyncio
async def test_upload_file_with_account_key_honors_endpoint_suffix(request, env_fixture, expected_suffix, monkeypatch):
    """The account key path returns its own blob URL, built independently of the REST path"""
    request.getfixturevalue(env_fixture)
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_KEY", "dGVzdC1rZXk=")

    file_client = MagicMock()
    file_client.create_file = AsyncMock()
    file_client.append_data = AsyncMock()
    file_client.flush_data = AsyncMock()

    directory_client = MagicMock()
    directory_client.exists = AsyncMock(return_value=True)
    directory_client.get_file_client = MagicMock(return_value=file_client)

    file_system_client = MagicMock()
    file_system_client.exists = AsyncMock(return_value=True)
    file_system_client.get_directory_client = MagicMock(return_value=directory_client)

    service_client = MagicMock()
    service_client.get_file_system_client = MagicMock(return_value=file_system_client)

    fake_aio_module = MagicMock()
    fake_aio_module.DataLakeServiceClient = MagicMock(return_value=service_client)

    with patch.dict(sys.modules, {"azure.storage.filedatalake.aio": fake_aio_module}):
        backend = AzureBlobStorageBackend()
        storage_url = await backend.upload_file(
            file_content=b"hello",
            filename="report.json",
            content_type="application/json",
            path_prefix="logs",
            file_naming_strategy="original_filename",
        )

    assert storage_url == f"https://test-account.blob.{expected_suffix}/test-container/logs/report.json"
