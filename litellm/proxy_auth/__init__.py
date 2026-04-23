"""
Proxy Authentication module for LiteLLM SDK.

This module provides OAuth2/JWT token management for authenticating
with LiteLLM Proxy or any OAuth2-protected endpoint.

Usage:
    from litellm.proxy_auth import AzureADCredential, ProxyAuthHandler

    litellm.proxy_auth = ProxyAuthHandler(
        credential=AzureADCredential(),
        scope="api://my-proxy/.default"
    )
"""

from .credentials import (
    AccessToken,
    TokenCredential,
    AzureADCredential,
    GenericOAuth2Credential,
    ProxyAuthHandler,
)

__all__ = [
    "AccessToken",
    "TokenCredential",
    "AzureADCredential",
    "GenericOAuth2Credential",
    "ProxyAuthHandler",
]
