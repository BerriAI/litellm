import os
import sys
import pytest
from dotenv import load_dotenv

load_dotenv()
import os
import httpx

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from unittest.mock import patch, MagicMock
import logging
from litellm._logging import verbose_logger
from litellm._uuid import uuid

verbose_logger.setLevel(logging.DEBUG)

# Minimal setup for module-level instantiation
import litellm.proxy.proxy_server
litellm.proxy.proxy_server.premium_user = True

from litellm.secret_managers.hashicorp_secret_manager import HashicorpSecretManager

hashicorp_secret_manager = HashicorpSecretManager()


mock_vault_response = {
    "request_id": "80fafb6a-e96a-4c5b-29fa-ff505ac72201",
    "lease_id": "",
    "renewable": False,
    "lease_duration": 0,
    "data": {
        "data": {"key": "value-mock"},
        "metadata": {
            "created_time": "2025-01-01T22:13:50.93942388Z",
            "custom_metadata": None,
            "deletion_time": "",
            "destroyed": False,
            "version": 1,
        },
    },
    "wrap_info": None,
    "warnings": None,
    "auth": None,
    "mount_type": "kv",
}

# Update the mock_vault_response for write operations
mock_write_response = {
    "request_id": "80fafb6a-e96a-4c5b-29fa-ff505ac72201",
    "lease_id": "",
    "renewable": False,
    "lease_duration": 0,
    "data": {
        "created_time": "2025-01-04T16:58:42.684673531Z",
        "custom_metadata": None,
        "deletion_time": "",
        "destroyed": False,
        "version": 1,
    },
    "wrap_info": None,
    "warnings": None,
    "auth": None,
    "mount_type": "kv",
}


def test_hashicorp_secret_manager_get_secret():
    with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.get") as mock_get:
        # Configure the mock response using MagicMock
        mock_response = MagicMock()
        mock_response.json.return_value = mock_vault_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Test the secret manager
        secret = hashicorp_secret_manager.sync_read_secret("sample-secret-mock")
        assert secret == "value-mock"

        # Verify the request was made with correct parameters
        mock_get.assert_called_once()
        called_url = mock_get.call_args[0][0]
        assert "sample-secret-mock" in called_url

        assert (
            called_url
            == "https://test-cluster-public-vault-0f98180c.e98296b2.z1.hashicorp.cloud:8200/v1/admin/secret/data/sample-secret-mock"
        )
        assert "X-Vault-Token" in mock_get.call_args.kwargs["headers"]


@pytest.mark.asyncio
async def test_hashicorp_secret_manager_write_secret():
    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post"
    ) as mock_post:
        # Configure the mock response
        mock_response = MagicMock()
        mock_response.json.return_value = (
            mock_write_response  # Use the write-specific response
        )
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # Test the secret manager
        secret_name = f"sample-secret-test-{uuid.uuid4()}"
        secret_value = f"value-mock-{uuid.uuid4()}"
        response = await hashicorp_secret_manager.async_write_secret(
            secret_name=secret_name,
            secret_value=secret_value,
        )

        # Verify the response and that the request was made correctly
        assert (
            response == mock_write_response
        )  # Compare against write-specific response
        mock_post.assert_called_once()
        print("CALL ARGS=", mock_post.call_args)
        print("call args[1]=", mock_post.call_args[1])

        # Verify URL
        called_url = mock_post.call_args[1]["url"]
        assert secret_name in called_url
        assert (
            called_url
            == f"{hashicorp_secret_manager.vault_addr}/v1/admin/secret/data/{secret_name}"
        )

        # Verify request body
        json_data = mock_post.call_args[1]["json"]
        assert "data" in json_data
        assert "key" in json_data["data"]
        assert json_data["data"]["key"] == secret_value


@pytest.mark.asyncio
async def test_hashicorp_secret_manager_delete_secret():
    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.delete"
    ) as mock_delete:
        # Configure the mock response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_delete.return_value = mock_response

        # Test the secret manager
        secret_name = f"sample-secret-test-{uuid.uuid4()}"
        response = await hashicorp_secret_manager.async_delete_secret(
            secret_name=secret_name
        )

        # Verify the response
        assert response == {
            "status": "success",
            "message": f"Secret {secret_name} deleted successfully",
        }

        # Verify the request was made correctly
        mock_delete.assert_called_once()

        # Verify URL
        called_url = mock_delete.call_args[1]["url"]
        assert secret_name in called_url
        assert (
            called_url
            == f"{hashicorp_secret_manager.vault_addr}/v1/admin/secret/data/{secret_name}"
        )


def test_hashicorp_secret_manager_tls_cert_auth(monkeypatch):
    monkeypatch.setenv("HCP_VAULT_TOKEN", "test-client-token-12345")
    print("HCP_VAULT_TOKEN=", os.getenv("HCP_VAULT_TOKEN"))
    # Mock both httpx.post and httpx.Client
    with patch("httpx.Client") as mock_client:
        # Configure the mock client and response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "auth": {
                "client_token": "test-client-token-12345",
                "lease_duration": 3600,
                "renewable": True,
            }
        }
        mock_response.raise_for_status.return_value = None
        
        # Configure the mock client's post method
        mock_client_instance = MagicMock()
        mock_client_instance.post.return_value = mock_response
        mock_client.return_value = mock_client_instance

        # Create a new instance with TLS cert config
        test_manager = HashicorpSecretManager()
        test_manager.tls_cert_path = "cert.pem"
        test_manager.tls_key_path = "key.pem"
        test_manager.vault_cert_role = "test-role"
        test_manager.vault_namespace = "test-namespace"
        
        # Test the TLS auth method
        token = test_manager._auth_via_tls_cert()

        # Verify the token
        assert token == "test-client-token-12345"
        
        # Verify Client was created with correct cert tuple
        mock_client.assert_called_once_with(cert=("cert.pem", "key.pem"))
        
        # Verify post was called with correct parameters
        mock_client_instance.post.assert_called_once_with(
            f"{test_manager.vault_addr}/v1/auth/cert/login",
            headers={"X-Vault-Namespace": "test-namespace"},
            json={"name": "test-role"},
        )

        # Verify the token was cached
        assert test_manager.cache.get_cache("hcp_vault_token") == "test-client-token-12345"


def test_hashicorp_secret_manager_approle_auth(monkeypatch):
    """
    Test AppRole authentication makes the expected POST request to the correct URL.
    """
    monkeypatch.setenv("HCP_VAULT_TOKEN", "test-token-12345")
    
    with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "auth": {
                "client_token": "hvs.approle-token-67890",
                "lease_duration": 2764800,
                "renewable": True,
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        test_manager = HashicorpSecretManager()
        test_manager.approle_role_id = "test-role-id-123"
        test_manager.approle_secret_id = "test-secret-id-456"
        test_manager.approle_mount_path = "approle"
        
        token = test_manager._auth_via_approle()

        assert token == "hvs.approle-token-67890"
        
        expected_headers = {}
        if test_manager.vault_namespace:
            expected_headers["X-Vault-Namespace"] = test_manager.vault_namespace
        
        mock_post.assert_called_once_with(
            url=f"{test_manager.vault_addr}/v1/auth/approle/login",
            headers=expected_headers,
            json={
                "role_id": "test-role-id-123",
                "secret_id": "test-secret-id-456",
            },
        )

        assert test_manager.cache.get_cache("hcp_vault_approle_token") == "hvs.approle-token-67890"


def test_hashicorp_custom_mount_and_prefix():
    """Test URL construction with custom mount name and path prefix using get_url method."""
    # Save original values
    original_mount = hashicorp_secret_manager.vault_mount_name
    original_prefix = hashicorp_secret_manager.vault_path_prefix
    original_namespace = hashicorp_secret_manager.vault_namespace
    
    try:
        # Test that existing manager uses default "secret" mount and namespace "admin"
        url = hashicorp_secret_manager.get_url("my-secret")
        assert "/secret/data/" in url
        assert "my-secret" in url
        
        # Test custom mount name
        hashicorp_secret_manager.vault_mount_name = "kv"
        hashicorp_secret_manager.vault_path_prefix = None
        hashicorp_secret_manager.vault_namespace = None
        url = hashicorp_secret_manager.get_url("my-secret")
        assert url == f"{hashicorp_secret_manager.vault_addr}/v1/kv/data/my-secret"
        
        # Test path prefix
        hashicorp_secret_manager.vault_mount_name = "secret"
        hashicorp_secret_manager.vault_path_prefix = "myapp"
        url = hashicorp_secret_manager.get_url("my-secret")
        assert url == f"{hashicorp_secret_manager.vault_addr}/v1/secret/data/myapp/my-secret"
        
        # Test both custom mount and prefix
        hashicorp_secret_manager.vault_mount_name = "kv"
        hashicorp_secret_manager.vault_path_prefix = "production"
        url = hashicorp_secret_manager.get_url("my-secret")
        assert url == f"{hashicorp_secret_manager.vault_addr}/v1/kv/data/production/my-secret"
    finally:
        # Restore original values
        hashicorp_secret_manager.vault_mount_name = original_mount
        hashicorp_secret_manager.vault_path_prefix = original_prefix
        hashicorp_secret_manager.vault_namespace = original_namespace
