import json
import os
import sys
from typing import Optional
from unittest.mock import MagicMock, patch

# Adds the grandparent directory to sys.path to allow importing project modules
sys.path.insert(0, os.path.abspath("../.."))

import pytest

from litellm.secret_managers.get_azure_ad_token_provider import (
    get_azure_ad_token_provider,
    get_azure_credential,
    infer_credential_type_from_environment,
)
from litellm.types.secret_managers.get_azure_ad_token_provider import (
    AzureCredentialType,
)


class TestGetAzureAdTokenProvider:
    @patch.dict(
        os.environ,
        {
            "AZURE_CLIENT_ID": "test-client-id",
            "AZURE_CLIENT_SECRET": "test-client-secret",
            "AZURE_TENANT_ID": "test-tenant-id",
            "AZURE_SCOPE": "https://cognitiveservices.azure.com/.default",
            "AZURE_CREDENTIAL": "ClientSecretCredential",
        },
    )
    @patch("azure.identity.get_bearer_token_provider")
    @patch("azure.identity.ClientSecretCredential")
    def test_get_azure_ad_token_provider_client_secret_credential(
        self, mock_client_secret_credential, mock_get_bearer_token_provider
    ):
        """Test get_azure_ad_token_provider with ClientSecretCredential."""
        # Mock the Azure identity credential instance
        mock_credential_instance = MagicMock()
        mock_client_secret_credential.return_value = mock_credential_instance

        # Mock the bearer token provider
        mock_token_provider = MagicMock(return_value="mock-token")
        mock_get_bearer_token_provider.return_value = mock_token_provider

        # Call the function
        result = get_azure_ad_token_provider()

        # Assertions
        assert callable(result)
        mock_client_secret_credential.assert_called_once_with(
            client_id="test-client-id",
            client_secret="test-client-secret",
            tenant_id="test-tenant-id",
        )
        mock_get_bearer_token_provider.assert_called_once_with(
            mock_credential_instance, "https://cognitiveservices.azure.com/.default"
        )

        # Test that the returned callable works
        token = result()
        assert token == "mock-token"

    @patch.dict(
        os.environ,
        {
            "AZURE_CLIENT_ID": "test-client-id",
            "AZURE_SCOPE": "https://cognitiveservices.azure.com/.default",
            "AZURE_CREDENTIAL": "ManagedIdentityCredential",
        },
    )
    @patch("azure.identity.get_bearer_token_provider")
    @patch("azure.identity.ManagedIdentityCredential")
    def test_get_azure_ad_token_provider_managed_identity_credential(
        self, mock_managed_identity_credential, mock_get_bearer_token_provider
    ):
        """Test get_azure_ad_token_provider with ManagedIdentityCredential."""
        # Mock the Azure identity credential instance
        mock_credential_instance = MagicMock()
        mock_managed_identity_credential.return_value = mock_credential_instance

        # Mock the bearer token provider
        mock_token_provider = MagicMock(return_value="mock-managed-identity-token")
        mock_get_bearer_token_provider.return_value = mock_token_provider

        # Call the function
        result = get_azure_ad_token_provider()

        # Assertions
        assert callable(result)
        mock_managed_identity_credential.assert_called_once_with(
            client_id="test-client-id"
        )
        mock_get_bearer_token_provider.assert_called_once_with(
            mock_credential_instance, "https://cognitiveservices.azure.com/.default"
        )

        # Test that the returned callable works
        token = result()
        assert token == "mock-managed-identity-token"

    @patch.dict(
        os.environ,
        {
            "AZURE_CLIENT_ID": "test-client-id",
            "AZURE_TENANT_ID": "test-tenant-id",
            "AZURE_CERTIFICATE_PATH": "/path/to/cert.pem",
            "AZURE_SCOPE": "https://cognitiveservices.azure.com/.default",
            "AZURE_CREDENTIAL": "CertificateCredential",
        },
    )
    @patch("azure.identity.get_bearer_token_provider")
    @patch("azure.identity.CertificateCredential")
    def test_get_azure_ad_token_provider_certificate_credential(
        self, mock_certificate_credential, mock_get_bearer_token_provider
    ):
        """Test get_azure_ad_token_provider with CertificateCredential."""
        # Mock the Azure identity credential instance
        mock_credential_instance = MagicMock()
        mock_certificate_credential.return_value = mock_credential_instance

        # Mock the bearer token provider
        mock_token_provider = MagicMock(return_value="mock-certificate-token")
        mock_get_bearer_token_provider.return_value = mock_token_provider

        # Call the function
        result = get_azure_ad_token_provider()

        # Assertions
        assert callable(result)
        mock_certificate_credential.assert_called_once_with(
            client_id="test-client-id",
            tenant_id="test-tenant-id",
            certificate_path="/path/to/cert.pem",
        )
        mock_get_bearer_token_provider.assert_called_once_with(
            mock_credential_instance, "https://cognitiveservices.azure.com/.default"
        )

        # Test that the returned callable works
        token = result()
        assert token == "mock-certificate-token"

    @patch.dict(
        os.environ,
        {
            "AZURE_CLIENT_ID": "test-client-id",
            "AZURE_TENANT_ID": "test-tenant-id",
            "AZURE_CERTIFICATE_PATH": "/path/to/cert.pem",
            "AZURE_SCOPE": "https://cognitiveservices.azure.com/.default",
            "AZURE_CREDENTIAL": "CertificateCredential",
            "AZURE_CERTIFICATE_PASSWORD": "pwd4cert.pem",
        },
    )
    @patch("azure.identity.get_bearer_token_provider")
    @patch("azure.identity.CertificateCredential")
    def test_get_azure_ad_token_provider_password_protected_certificate_credential(
        self, mock_certificate_credential, mock_get_bearer_token_provider
    ):
        """Test get_azure_ad_token_provider with password protected certificate in CertificateCredential."""
        # Mock the Azure identity credential instance
        mock_credential_instance = MagicMock()
        mock_certificate_credential.return_value = mock_credential_instance

        # Mock the bearer token provider
        mock_token_provider = MagicMock(return_value="mock-certificate-token")
        mock_get_bearer_token_provider.return_value = mock_token_provider

        # Call the function
        result = get_azure_ad_token_provider()

        # Assertions
        assert callable(result)
        mock_certificate_credential.assert_called_once_with(
            client_id="test-client-id",
            tenant_id="test-tenant-id",
            certificate_path="/path/to/cert.pem",
            password="pwd4cert.pem",
        )
        mock_get_bearer_token_provider.assert_called_once_with(
            mock_credential_instance, "https://cognitiveservices.azure.com/.default"
        )

        # Test that the returned callable works
        token = result()
        assert token == "mock-certificate-token"

    @patch.dict(
        os.environ,
        {
            "AZURE_CREDENTIAL": "DefaultAzureCredential",
        },
    )
    @patch("azure.identity.get_bearer_token_provider")
    @patch("azure.identity.DefaultAzureCredential")
    def test_get_azure_ad_token_provider_default_azure_credential(
        self, mock_certificate_credential, mock_get_bearer_token_provider
    ):
        """Test get_azure_ad_token_provider with DefaultAzureCredential."""
        # Mock the Azure identity credential instance
        mock_credential_instance = MagicMock()
        mock_certificate_credential.return_value = mock_credential_instance

        # Mock the bearer token provider
        mock_token_provider = MagicMock(return_value="mock-certificate-token")
        mock_get_bearer_token_provider.return_value = mock_token_provider

        # Call the function
        result = get_azure_ad_token_provider()

        # Assertions
        assert callable(result)
        mock_certificate_credential.assert_called_once_with()
        mock_get_bearer_token_provider.assert_called_once_with(
            mock_credential_instance, "https://cognitiveservices.azure.com/.default"
        )

        # Test that the returned callable works
        token = result()
        assert token == "mock-certificate-token"

    @patch.dict(os.environ, {}, clear=True)  # Clear all environment variables
    @patch("azure.identity.get_bearer_token_provider")
    @patch("azure.identity.DefaultAzureCredential")
    def test_get_azure_ad_token_provider_defaults_to_default_azure_credential(
        self, mock_default_azure_credential, mock_get_bearer_token_provider
    ):
        """Test get_azure_ad_token_provider defaults to DefaultAzureCredential when no credentials are present."""
        # Mock the Azure identity credential instance
        mock_credential_instance = MagicMock()
        mock_default_azure_credential.return_value = mock_credential_instance

        # Mock the bearer token provider
        mock_token_provider = MagicMock(return_value="mock-default-token")
        mock_get_bearer_token_provider.return_value = mock_token_provider

        # Call the function
        result = get_azure_ad_token_provider()

        # Assertions
        assert callable(result)
        mock_default_azure_credential.assert_called_once_with()
        mock_get_bearer_token_provider.assert_called_once_with(
            mock_credential_instance, "https://cognitiveservices.azure.com/.default"
        )

        # Test that the returned callable works
        token = result()
        assert token == "mock-default-token"

    @patch.dict(
        os.environ,
        {
            "AZURE_CLIENT_ID": "test-client-id",
            "AZURE_TENANT_ID": "test-tenant-id",
            "AZURE_FEDERATED_TOKEN_FILE": "/var/run/secrets/azure/tokens/azure-identity-token",
            "AZURE_SCOPE": "https://cognitiveservices.azure.com/.default",
            "AZURE_CREDENTIAL": "WorkloadIdentityCredential",
        },
    )
    @patch("azure.identity.get_bearer_token_provider")
    @patch("azure.identity.WorkloadIdentityCredential")
    def test_get_azure_ad_token_provider_workload_identity_credential(
        self, mock_workload_identity_credential, mock_get_bearer_token_provider
    ):
        """Test get_azure_ad_token_provider with WorkloadIdentityCredential (AKS federation)."""
        mock_credential_instance = MagicMock()
        mock_workload_identity_credential.return_value = mock_credential_instance

        mock_token_provider = MagicMock(return_value="mock-workload-identity-token")
        mock_get_bearer_token_provider.return_value = mock_token_provider

        result = get_azure_ad_token_provider()

        assert callable(result)
        mock_workload_identity_credential.assert_called_once_with(
            client_id="test-client-id",
            tenant_id="test-tenant-id",
            token_file_path="/var/run/secrets/azure/tokens/azure-identity-token",
        )
        mock_get_bearer_token_provider.assert_called_once_with(
            mock_credential_instance, "https://cognitiveservices.azure.com/.default"
        )

        token = result()
        assert token == "mock-workload-identity-token"


class TestInferCredentialTypeFromEnvironment:
    @patch.dict(
        os.environ,
        {
            "AZURE_CLIENT_ID": "test-client-id",
            "AZURE_TENANT_ID": "test-tenant-id",
            "AZURE_FEDERATED_TOKEN_FILE": "/var/run/secrets/azure/tokens/azure-identity-token",
        },
        clear=True,
    )
    def test_infers_workload_identity_from_federated_token_file(self):
        """A federated token file plus client/tenant id (and no client secret) implies
        workload identity, not managed identity."""
        assert (
            infer_credential_type_from_environment()
            == AzureCredentialType.WorkloadIdentityCredential
        )

    @patch.dict(
        os.environ,
        {
            "AZURE_CLIENT_ID": "test-client-id",
            "AZURE_CLIENT_SECRET": "test-client-secret",
            "AZURE_TENANT_ID": "test-tenant-id",
            "AZURE_FEDERATED_TOKEN_FILE": "/var/run/secrets/azure/tokens/azure-identity-token",
        },
        clear=True,
    )
    def test_client_secret_takes_precedence_over_workload_identity(self):
        """When a client secret is present, keep using the service-principal flow."""
        assert (
            infer_credential_type_from_environment()
            == AzureCredentialType.ClientSecretCredential
        )

    @patch.dict(
        os.environ,
        {"AZURE_CLIENT_ID": "test-client-id"},
        clear=True,
    )
    def test_client_id_only_infers_managed_identity(self):
        assert (
            infer_credential_type_from_environment()
            == AzureCredentialType.ManagedIdentityCredential
        )


class TestGetAzureCredential:
    @patch.dict(
        os.environ,
        {
            "AZURE_CLIENT_ID": "test-client-id",
            "AZURE_TENANT_ID": "test-tenant-id",
            "AZURE_FEDERATED_TOKEN_FILE": "/var/run/secrets/azure/tokens/azure-identity-token",
        },
        clear=True,
    )
    @patch("azure.identity.WorkloadIdentityCredential")
    def test_returns_workload_identity_credential_object(
        self, mock_workload_identity_credential
    ):
        """get_azure_credential returns the credential object (used by the Key Vault
        loader), inferring workload identity from the environment."""
        mock_credential_instance = MagicMock()
        mock_workload_identity_credential.return_value = mock_credential_instance

        credential = get_azure_credential()

        assert credential is mock_credential_instance
        mock_workload_identity_credential.assert_called_once_with(
            client_id="test-client-id",
            tenant_id="test-tenant-id",
            token_file_path="/var/run/secrets/azure/tokens/azure-identity-token",
        )

    @patch.dict(os.environ, {}, clear=True)
    @patch("azure.identity.DefaultAzureCredential")
    def test_explicit_argument_overrides_environment(
        self, mock_default_azure_credential
    ):
        mock_credential_instance = MagicMock()
        mock_default_azure_credential.return_value = mock_credential_instance

        credential = get_azure_credential(
            azure_credential=AzureCredentialType.DefaultAzureCredential
        )

        assert credential is mock_credential_instance
        mock_default_azure_credential.assert_called_once_with()
