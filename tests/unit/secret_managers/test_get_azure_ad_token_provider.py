import itertools
import json
import os
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from queue import SimpleQueue
from typing import Final, Optional
from unittest.mock import ANY, MagicMock, patch

# Adds the grandparent directory to sys.path to allow importing project modules

import pytest
from azure.core.credentials import AccessToken, AccessTokenInfo, TokenRequestOptions
from azure.core.exceptions import ClientAuthenticationError

from litellm.secret_managers.get_azure_ad_token_provider import (
    get_azure_ad_token_provider,
    infer_credential_type_from_environment,
    single_flight_bearer_token_provider,
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
            ANY, "https://cognitiveservices.azure.com/.default"
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
            ANY, "https://cognitiveservices.azure.com/.default"
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
            ANY, "https://cognitiveservices.azure.com/.default"
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
            ANY, "https://cognitiveservices.azure.com/.default"
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
            ANY, "https://cognitiveservices.azure.com/.default"
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
            ANY, "https://cognitiveservices.azure.com/.default"
        )

        # Test that the returned callable works
        token = result()
        assert token == "mock-default-token"


_CONCURRENT_CALLERS: Final = 4
_SCOPE: Final = "api://audit/.default"
_YEAR_2100: Final = 4_102_444_800
_LONG_EXPIRED: Final = 1
_JOIN_WINDOW_SECONDS: Final = 0.5
_FAST_FAILING_CALLERS: Final = 16


@dataclass(frozen=True, slots=True)
class _HeldCredential:
    expires_on: int = _YEAR_2100
    failing_attempts: int = 0
    attempts: Iterator[int] = field(default_factory=itertools.count)
    requests: SimpleQueue[tuple[str, ...]] = field(default_factory=SimpleQueue)
    everyone_asked: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)

    def get_token_info(self, *scopes: str, options: TokenRequestOptions | None = None) -> AccessTokenInfo:
        self.requests.put(scopes)
        if self.requests.qsize() >= _CONCURRENT_CALLERS:
            self.everyone_asked.set()
        self.release.wait(5)
        attempt: Final = next(self.attempts)
        if attempt < self.failing_attempts:
            raise ClientAuthenticationError("Entra is down")
        return AccessTokenInfo(f"token-{attempt}", self.expires_on)


def _call_together(provider: Callable[[], str], credential: _HeldCredential) -> tuple[Future[str], ...]:
    start: Final = threading.Barrier(_CONCURRENT_CALLERS)

    def call() -> str:
        start.wait(5)
        return provider()

    with ThreadPoolExecutor(max_workers=_CONCURRENT_CALLERS) as pool:
        calls: Final = tuple(pool.submit(call) for _ in range(_CONCURRENT_CALLERS))
        assert credential.requests.get(timeout=5) == (_SCOPE,)
        credential.everyone_asked.wait(0.5)
        credential.release.set()
    return calls


def test_concurrent_cold_callers_share_one_token_request() -> None:
    credential: Final = _HeldCredential()
    provider: Final = single_flight_bearer_token_provider(credential, _SCOPE)

    calls: Final = _call_together(provider, credential)

    assert tuple(call.result() for call in calls) == ("token-0",) * _CONCURRENT_CALLERS
    assert credential.requests.qsize() == 0


def test_concurrent_callers_after_the_token_expired_share_one_token_request() -> None:
    credential: Final = _HeldCredential(expires_on=_LONG_EXPIRED)
    provider: Final = single_flight_bearer_token_provider(credential, _SCOPE)
    credential.release.set()
    provider()
    credential.requests.get_nowait()
    credential.release.clear()

    calls: Final = _call_together(provider, credential)

    assert tuple(call.result() for call in calls) == ("token-1",) * _CONCURRENT_CALLERS
    assert credential.requests.qsize() == 0


def test_a_failed_cold_token_request_fails_only_the_caller_that_made_it() -> None:
    credential: Final = _HeldCredential(failing_attempts=1)
    provider: Final = single_flight_bearer_token_provider(credential, _SCOPE)

    calls: Final = _call_together(provider, credential)

    failed: Final = sorted(isinstance(call.exception(), ClientAuthenticationError) for call in calls)
    assert failed == [False] * (_CONCURRENT_CALLERS - 1) + [True]


@dataclass(frozen=True, slots=True)
class _ScriptedCredential:
    expiries: Iterator[int | None] = field(default_factory=lambda: itertools.repeat(None))
    attempts: Iterator[int] = field(default_factory=itertools.count)
    requests: SimpleQueue[None] = field(default_factory=SimpleQueue)
    answers: threading.Semaphore = field(default_factory=lambda: threading.Semaphore(0))

    def get_token_info(self, *_: str, options: TokenRequestOptions | None = None) -> AccessTokenInfo:
        self.requests.put(None)
        self.answers.acquire(timeout=5)
        attempt: Final = next(self.attempts)
        expires_on: Final = next(self.expiries)
        if expires_on is None:
            raise ClientAuthenticationError("Entra is down")
        return AccessTokenInfo(f"token-{attempt}", expires_on)


def _answer_together(
    provider: Callable[[], str], credential: _ScriptedCredential, token_requests: int
) -> tuple[Future[str], ...]:
    start: Final = threading.Barrier(_CONCURRENT_CALLERS)

    def call() -> str:
        start.wait(5)
        return provider()

    with ThreadPoolExecutor(max_workers=_CONCURRENT_CALLERS) as pool:
        calls: Final = tuple(pool.submit(call) for _ in range(_CONCURRENT_CALLERS))
        for _ in range(token_requests):
            credential.requests.get(timeout=5)
            time.sleep(_JOIN_WINDOW_SECONDS)
            credential.answers.release()
    return calls


def test_an_entra_outage_retries_only_the_first_failed_token_request() -> None:
    credential: Final = _ScriptedCredential()
    provider: Final = single_flight_bearer_token_provider(credential, _SCOPE)

    first_burst: Final = _answer_together(provider, credential, token_requests=2)
    later_burst: Final = _answer_together(provider, credential, token_requests=1)

    assert all(isinstance(call.exception(), ClientAuthenticationError) for call in first_burst + later_burst)
    assert credential.requests.qsize() == 0


def test_waiting_callers_retry_a_failed_token_request_again_once_entra_recovers() -> None:
    credential: Final = _ScriptedCredential(iter((None, None, _LONG_EXPIRED, None, _LONG_EXPIRED)))
    provider: Final = single_flight_bearer_token_provider(credential, _SCOPE)
    _answer_together(provider, credential, token_requests=2)
    credential.answers.release()
    assert provider() == "token-2"
    credential.requests.get_nowait()

    recovered: Final = _answer_together(provider, credential, token_requests=2)

    assert sorted(call.exception() is None for call in recovered) == [False] + [True] * (_CONCURRENT_CALLERS - 1)
    assert {call.result() for call in recovered if call.exception() is None} == {"token-4"}


def test_a_failed_refresh_fails_the_request_that_made_it_even_while_a_valid_token_is_held() -> None:
    credential: Final = _ScriptedCredential(iter((int(time.time()) + 120, None)))
    provider: Final = single_flight_bearer_token_provider(credential, _SCOPE)
    credential.answers.release(2)

    assert provider() == "token-0"
    with pytest.raises(ClientAuthenticationError):
        provider()


def test_callers_waiting_on_a_failed_token_request_share_one_retry_when_entra_fails_fast() -> None:
    credential: Final = _ScriptedCredential()
    provider: Final = single_flight_bearer_token_provider(credential, _SCOPE)
    start: Final = threading.Barrier(_FAST_FAILING_CALLERS)

    def call() -> str:
        start.wait(5)
        return provider()

    with ThreadPoolExecutor(max_workers=_FAST_FAILING_CALLERS) as pool:
        calls: Final = tuple(pool.submit(call) for _ in range(_FAST_FAILING_CALLERS))
        credential.requests.get(timeout=5)
        time.sleep(_JOIN_WINDOW_SECONDS)
        credential.answers.release(_FAST_FAILING_CALLERS)

    assert all(isinstance(call.exception(), ClientAuthenticationError) for call in calls)
    assert credential.requests.qsize() == 1


@dataclass(frozen=True, slots=True)
class _ExpiringSoonTokens:
    first_lifetime_seconds: int = 120
    requests: SimpleQueue[None] = field(default_factory=SimpleQueue)
    refresh_started: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)

    def next_token(self) -> tuple[str, int]:
        self.requests.put(None)
        if self.requests.qsize() == 1:
            return "expiring-token", int(time.time()) + self.first_lifetime_seconds
        self.refresh_started.set()
        self.release.wait(5)
        return "refreshed-token", _YEAR_2100


@dataclass(frozen=True, slots=True)
class _ExpiringSoonTokenInfoCredential:
    tokens: _ExpiringSoonTokens = field(default_factory=_ExpiringSoonTokens)

    def get_token_info(self, *_: str, options: TokenRequestOptions | None = None) -> AccessTokenInfo:
        return AccessTokenInfo(*self.tokens.next_token())


@dataclass(frozen=True, slots=True)
class _ExpiringSoonTokenOnlyCredential:
    tokens: _ExpiringSoonTokens = field(default_factory=_ExpiringSoonTokens)

    def get_token(
        self,
        *_: str,
        claims: str | None = None,
        tenant_id: str | None = None,
        enable_cae: bool = False,
        **kwargs: object,
    ) -> AccessToken:
        return AccessToken(*self.tokens.next_token())


@pytest.mark.parametrize("credential_type", (_ExpiringSoonTokenInfoCredential, _ExpiringSoonTokenOnlyCredential))
def test_callers_during_a_refresh_keep_the_valid_token_while_one_of_them_refreshes(
    credential_type: type[_ExpiringSoonTokenInfoCredential | _ExpiringSoonTokenOnlyCredential],
) -> None:
    credential: Final = credential_type()
    tokens: Final = credential.tokens
    provider: Final = single_flight_bearer_token_provider(credential, _SCOPE)
    assert provider() == "expiring-token"

    with ThreadPoolExecutor(max_workers=_CONCURRENT_CALLERS) as pool:
        refresh: Final = pool.submit(provider)
        assert tokens.refresh_started.wait(5)
        others: Final = tuple(pool.submit(provider) for _ in range(_CONCURRENT_CALLERS - 1))
        try:
            assert tuple(other.result(timeout=2) for other in others) == ("expiring-token",) * (_CONCURRENT_CALLERS - 1)
        finally:
            tokens.release.set()

    assert (refresh.result(), provider()) == ("refreshed-token", "refreshed-token")
    assert tokens.requests.qsize() == 2


def test_callers_during_a_refresh_wait_for_the_new_token_when_the_valid_one_is_about_to_expire() -> None:
    tokens: Final = _ExpiringSoonTokens(first_lifetime_seconds=30)
    provider: Final = single_flight_bearer_token_provider(_ExpiringSoonTokenInfoCredential(tokens), _SCOPE)
    assert provider() == "expiring-token"

    with ThreadPoolExecutor(max_workers=_CONCURRENT_CALLERS) as pool:
        refresh: Final = pool.submit(provider)
        assert tokens.refresh_started.wait(5)
        others: Final = tuple(pool.submit(provider) for _ in range(_CONCURRENT_CALLERS - 1))
        try:
            assert not wait(others, timeout=_JOIN_WINDOW_SECONDS).done
        finally:
            tokens.release.set()

    assert tuple(call.result() for call in (refresh, *others)) == ("refreshed-token",) * _CONCURRENT_CALLERS
    assert tokens.requests.qsize() == 2
