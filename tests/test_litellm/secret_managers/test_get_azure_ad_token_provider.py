import json
import os
from typing import Optional
from unittest.mock import MagicMock, patch

# Adds the grandparent directory to sys.path to allow importing project modules

import pytest
from azure.core.exceptions import ClientAuthenticationError

from litellm.secret_managers.get_azure_ad_token_provider import (
    get_azure_ad_token_provider,
    infer_credential_type_from_environment,
)
from litellm.types.secret_managers.get_azure_ad_token_provider import (
    AzureCredentialType,
)


class TestDeploymentIdentityCredential:
    @staticmethod
    def _chain_for(credential_type):
        with patch("azure.identity.get_bearer_token_provider", return_value=lambda: "token") as bearer:
            get_azure_ad_token_provider(
                azure_scope="https://storage.azure.com/.default",
                azure_credential=credential_type,
            )
        bearer.assert_called_once()
        with bearer.call_args.args[0] as chain:
            return {type(link).__name__ for link in chain.credentials}

    @staticmethod
    def _managed_identity_client_ids(credential_type):
        with patch("azure.identity.get_bearer_token_provider", return_value=lambda: "token") as bearer:
            get_azure_ad_token_provider(
                azure_scope="https://storage.azure.com/.default",
                azure_credential=credential_type,
            )
        with bearer.call_args.args[0] as chain:
            return [
                (link._credential._settings or {}).get("client_id")
                for link in chain.credentials
                if type(link).__name__ == "ManagedIdentityCredential"
            ]

    @patch.dict(
        os.environ,
        {
            "AZURE_CLIENT_ID": "workload-identity-client-id",
            "AZURE_TENANT_ID": "workload-identity-tenant-id",
            "AZURE_FEDERATED_TOKEN_FILE": "/var/run/secrets/azure/tokens/azure-identity-token",
        },
        clear=True,
    )
    def test_deployment_identity_reaches_workload_and_managed_identity_only(self):
        assert self._chain_for(AzureCredentialType.DeploymentIdentityCredential) == {
            "WorkloadIdentityCredential",
            "ManagedIdentityCredential",
        }

    @patch.dict(
        os.environ,
        {
            "AZURE_CLIENT_ID": "workload-identity-client-id",
            "AZURE_TENANT_ID": "workload-identity-tenant-id",
            "AZURE_FEDERATED_TOKEN_FILE": "/var/run/secrets/azure/tokens/azure-identity-token",
            "AZURE_TOKEN_CREDENTIALS": "dev",
        },
        clear=True,
    )
    def test_deployment_identity_survives_a_developer_only_token_credentials_setting(self):
        """AZURE_TOKEN_CREDENTIALS=dev asks the SDK for developer credentials only, which is every
        credential this chain drops, so the deployment's own identity has to win over it"""
        assert self._chain_for(AzureCredentialType.DeploymentIdentityCredential) == {
            "WorkloadIdentityCredential",
            "ManagedIdentityCredential",
        }

    @patch.dict(
        os.environ,
        {
            "AZURE_CLIENT_ID": "azure-openai-client-id",
            "AZURE_CLIENT_SECRET": "azure-openai-client-secret",
            "AZURE_TENANT_ID": "azure-openai-tenant-id",
        },
        clear=True,
    )
    def test_default_azure_credential_keeps_its_full_chain(self):
        """Azure OpenAI callers pass DefaultAzureCredential and must be unaffected by the
        narrowing that the storage callback asks for"""
        full_chain = self._chain_for(AzureCredentialType.DefaultAzureCredential)

        assert "EnvironmentCredential" in full_chain
        assert "AzureCliCredential" in full_chain
        assert "EnvironmentCredential" not in self._chain_for(
            AzureCredentialType.DeploymentIdentityCredential
        )

    @patch.dict(
        os.environ,
        {
            "AZURE_CLIENT_ID": "azure-openai-client-id",
            "AZURE_CLIENT_SECRET": "azure-openai-client-secret",
            "AZURE_TENANT_ID": "azure-openai-tenant-id",
        },
        clear=True,
    )
    def test_deployment_identity_refuses_to_mint_a_token_for_a_configured_service_principal(self):
        """A host carrying only an Azure OpenAI client secret must get no token at all, and the
        refusal must name the identities that were actually tried"""
        provider = get_azure_ad_token_provider(
            azure_scope="https://storage.azure.com/.default",
            azure_credential=AzureCredentialType.DeploymentIdentityCredential,
        )

        with pytest.raises(ClientAuthenticationError) as refusal:
            provider()

        assert "ManagedIdentityCredential" in str(refusal.value)
        assert "EnvironmentCredential" not in str(refusal.value)
        assert "AzureCliCredential" not in str(refusal.value)
        assert "azure-openai-client-secret" not in str(refusal.value)

    @patch.dict(
        os.environ,
        {
            "AZURE_CLIENT_ID": "azure-openai-client-id",
            "AZURE_CLIENT_SECRET": "azure-openai-client-secret",
            "AZURE_TENANT_ID": "azure-openai-tenant-id",
        },
        clear=True,
    )
    def test_deployment_identity_still_reaches_a_system_assigned_managed_identity(self):
        """AZURE_CLIENT_ID names one identity for the whole proxy, and pointing it at Azure OpenAI
        must not hide the system assigned identity the host runs as"""
        client_ids = self._managed_identity_client_ids(AzureCredentialType.DeploymentIdentityCredential)

        assert "azure-openai-client-id" in client_ids
        assert None in client_ids

    @patch.dict(
        os.environ,
        {
            "AZURE_CLIENT_ID": "user-assigned-identity-client-id",
            "AZURE_TOKEN_CREDENTIALS": "dev",
        },
        clear=True,
    )
    def test_deployment_identity_keeps_the_user_assigned_identity_under_a_dev_only_setting(self):
        """AZURE_TOKEN_CREDENTIALS=dev asks the SDK for developer credentials only, and the
        identity a host actually runs as has to survive that"""
        assert "user-assigned-identity-client-id" in self._managed_identity_client_ids(
            AzureCredentialType.DeploymentIdentityCredential
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

    @patch.dict(
        os.environ,
        {
            "AZURE_CLIENT_ID": "test-client-id",
            "AZURE_TENANT_ID": "test-tenant-id",
            "AZURE_FEDERATED_TOKEN_FILE": "/var/run/secrets/azure/tokens/azure-identity-token",
            "AZURE_AUTHORITY_HOST": "https://login.microsoftonline.com/",
        },
        clear=True,
    )
    @patch("azure.identity.get_bearer_token_provider")
    @patch("azure.identity.ManagedIdentityCredential")
    @patch("azure.identity.DefaultAzureCredential")
    def test_get_azure_ad_token_provider_prefers_workload_identity_over_managed_identity(
        self,
        mock_default_azure_credential,
        mock_managed_identity_credential,
        mock_get_bearer_token_provider,
    ):
        """The AKS workload identity webhook injects AZURE_CLIENT_ID, AZURE_TENANT_ID, and
        AZURE_FEDERATED_TOKEN_FILE, and never a client secret. Reading the bare client id as a
        managed identity sends the pod to IMDS, which has no identity attached to it, so every
        token request fails and the federated token is never exchanged. Only
        DefaultAzureCredential's chain reaches WorkloadIdentityCredential."""
        mock_credential_instance = MagicMock()
        mock_default_azure_credential.return_value = mock_credential_instance
        mock_get_bearer_token_provider.return_value = MagicMock(
            return_value="mock-workload-identity-token"
        )

        result = get_azure_ad_token_provider()

        assert (
            infer_credential_type_from_environment()
            == AzureCredentialType.DefaultAzureCredential
        )
        mock_managed_identity_credential.assert_not_called()
        mock_default_azure_credential.assert_called_once_with()
        assert result() == "mock-workload-identity-token"

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
