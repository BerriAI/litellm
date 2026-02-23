"""
Credential providers for proxy authentication.

This module provides a provider-agnostic interface for obtaining OAuth2/JWT tokens.
It follows the same TokenCredential protocol used by Azure SDK.
"""

import time
from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable


@dataclass
class AccessToken:
    """
    Represents an OAuth2 access token with expiration.

    This matches the structure used by azure.core.credentials.AccessToken.

    Attributes:
        token: The access token string (typically a JWT).
        expires_on: Unix timestamp when the token expires.
    """

    token: str
    expires_on: int


@runtime_checkable
class TokenCredential(Protocol):
    """
    Protocol for credential providers.

    This matches the azure.core.credentials.TokenCredential interface,
    allowing any Azure SDK credential to be used directly.

    Any class implementing get_token(scope) -> AccessToken can be used.
    """

    def get_token(self, scope: str) -> AccessToken:
        """
        Get an access token for the specified scope.

        Args:
            scope: The OAuth2 scope to request (e.g., "api://my-app/.default")

        Returns:
            AccessToken with the token string and expiration timestamp.
        """
        ...


class AzureADCredential:
    """
    Wrapper for Azure Identity credentials.

    This wraps any azure-identity credential (DefaultAzureCredential,
    ClientSecretCredential, ManagedIdentityCredential, etc.) and converts
    the token to our AccessToken format.

    If no credential is provided, it will use DefaultAzureCredential
    which tries multiple authentication methods automatically.

    Example:
        # Use default credential chain (env vars, managed identity, CLI, etc.)
        cred = AzureADCredential()

        # Or provide a specific credential
        from azure.identity import ClientSecretCredential
        azure_cred = ClientSecretCredential(tenant_id, client_id, client_secret)
        cred = AzureADCredential(credential=azure_cred)
    """

    def __init__(self, credential: Optional[Any] = None):
        """
        Initialize with an optional Azure credential.

        Args:
            credential: An azure-identity credential object. If None,
                       DefaultAzureCredential will be used on first token request.
        """
        self._credential: Any = credential
        self._initialized = credential is not None

    def get_token(self, scope: str) -> AccessToken:
        """
        Get an access token from Azure AD.

        Args:
            scope: The OAuth2 scope (e.g., "api://my-app/.default")

        Returns:
            AccessToken with the JWT and expiration.

        Raises:
            ImportError: If azure-identity is not installed.
        """
        if not self._initialized:
            try:
                from azure.identity import DefaultAzureCredential

                self._credential = DefaultAzureCredential()
                self._initialized = True
            except ImportError:
                raise ImportError(
                    "azure-identity is required for AzureADCredential. "
                    "Install it with: pip install azure-identity"
                )

        result = self._credential.get_token(scope)
        return AccessToken(token=result.token, expires_on=result.expires_on)


class GenericOAuth2Credential:
    """
    Generic OAuth2 client credentials flow.

    This works with any OAuth2 provider (Okta, Auth0, Keycloak, etc.)
    that supports the client_credentials grant type.

    Example:
        cred = GenericOAuth2Credential(
            client_id="my-client-id",
            client_secret="my-client-secret",
            token_url="https://my-idp.com/oauth2/token"
        )
    """

    def __init__(self, client_id: str, client_secret: str, token_url: str):
        """
        Initialize OAuth2 client credentials.

        Args:
            client_id: OAuth2 client ID
            client_secret: OAuth2 client secret
            token_url: Token endpoint URL (e.g., "https://idp.com/oauth2/token")
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url
        self._cached_token: Optional[AccessToken] = None

    def get_token(self, scope: str) -> AccessToken:
        """
        Get an access token using OAuth2 client credentials flow.

        Tokens are cached and reused until they expire (with 60s buffer).

        Args:
            scope: The OAuth2 scope to request

        Returns:
            AccessToken with the token and expiration.
        """
        # Return cached token if still valid (with 60s buffer)
        if self._cached_token and self._cached_token.expires_on > time.time() + 60:
            return self._cached_token

        import httpx

        response = httpx.post(
            self.token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "scope": scope,
            },
        )
        response.raise_for_status()
        data = response.json()

        self._cached_token = AccessToken(
            token=data["access_token"],
            expires_on=int(time.time()) + data.get("expires_in", 3600),
        )
        return self._cached_token


class ProxyAuthHandler:
    """
    Manages OAuth2/JWT token lifecycle for proxy authentication.

    This handler:
    - Obtains tokens from the configured credential provider
    - Caches tokens to avoid unnecessary requests
    - Automatically refreshes tokens before they expire (60s buffer)
    - Generates Authorization headers for HTTP requests

    Set this as litellm.proxy_auth to automatically inject auth headers
    into all requests to your LiteLLM Proxy.

    Example:
        import litellm
        from litellm.proxy_auth import AzureADCredential, ProxyAuthHandler

        litellm.proxy_auth = ProxyAuthHandler(
            credential=AzureADCredential(),
            scope="api://my-litellm-proxy/.default"
        )
        litellm.api_base = "https://my-proxy.example.com"

        # Auth headers are now automatically injected
        response = litellm.completion(model="gpt-4", messages=[...])
    """

    def __init__(self, credential: TokenCredential, scope: str):
        """
        Initialize the proxy auth handler.

        Args:
            credential: A TokenCredential implementation (AzureADCredential,
                       GenericOAuth2Credential, or any custom implementation)
            scope: The OAuth2 scope to request tokens for
        """
        self.credential = credential
        self.scope = scope
        self._cached_token: Optional[AccessToken] = None

    def get_token(self) -> AccessToken:
        """
        Get a valid access token, refreshing if necessary.

        Returns:
            AccessToken that is valid for at least 60 more seconds.
        """
        # Refresh if no token or token expires within 60 seconds
        if not self._cached_token or self._cached_token.expires_on <= time.time() + 60:
            self._cached_token = self.credential.get_token(self.scope)
        return self._cached_token

    def get_auth_headers(self) -> dict:
        """
        Get HTTP headers for authentication.

        Returns:
            Dict with Authorization header containing Bearer token.
        """
        token = self.get_token()
        return {"Authorization": f"Bearer {token.token}"}
