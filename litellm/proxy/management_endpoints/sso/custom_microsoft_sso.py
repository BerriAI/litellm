"""
Custom Microsoft SSO class that allows overriding default Microsoft endpoints.

This module provides a subclass of fastapi_sso's MicrosoftSSO that allows
custom authorization, token, and userinfo endpoints to be specified via environment
variables.

Environment Variables:
- MICROSOFT_AUTHORIZATION_ENDPOINT: Custom authorization endpoint URL
- MICROSOFT_TOKEN_ENDPOINT: Custom token endpoint URL  
- MICROSOFT_USERINFO_ENDPOINT: Custom userinfo endpoint URL

If these are not set, the default Microsoft endpoints are used.
"""

import os
from typing import List, Optional, Union

import pydantic
from fastapi_sso.sso.base import DiscoveryDocument
from fastapi_sso.sso.microsoft import MicrosoftSSO

from litellm._logging import verbose_proxy_logger


class CustomMicrosoftSSO(MicrosoftSSO):
    """
    Microsoft SSO subclass that allows overriding default endpoints via environment variables.

    Supports:
    - MICROSOFT_AUTHORIZATION_ENDPOINT
    - MICROSOFT_TOKEN_ENDPOINT
    - MICROSOFT_USERINFO_ENDPOINT
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: Optional[Union[pydantic.AnyHttpUrl, str]] = None,
        allow_insecure_http: bool = False,
        scope: Optional[List[str]] = None,
        tenant: Optional[str] = None,
    ):
        super().__init__(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            allow_insecure_http=allow_insecure_http,
            scope=scope,
            tenant=tenant,
        )

    async def get_discovery_document(self) -> DiscoveryDocument:
        """
        Override to support custom endpoints via environment variables.
        Falls back to default Microsoft endpoints if not set.
        """
        custom_authorization_endpoint = os.getenv(
            "MICROSOFT_AUTHORIZATION_ENDPOINT", None
        )
        custom_token_endpoint = os.getenv("MICROSOFT_TOKEN_ENDPOINT", None)
        custom_userinfo_endpoint = os.getenv("MICROSOFT_USERINFO_ENDPOINT", None)

        # Use custom endpoints if set, otherwise use defaults
        authorization_endpoint = (
            custom_authorization_endpoint
            or f"https://login.microsoftonline.com/{self.tenant}/oauth2/v2.0/authorize"
        )
        token_endpoint = (
            custom_token_endpoint
            or f"https://login.microsoftonline.com/{self.tenant}/oauth2/v2.0/token"
        )
        userinfo_endpoint = (
            custom_userinfo_endpoint or f"https://graph.microsoft.com/{self.version}/me"
        )

        if custom_authorization_endpoint or custom_token_endpoint or custom_userinfo_endpoint:
            verbose_proxy_logger.debug(
                f"Using custom Microsoft SSO endpoints - "
                f"authorization: {authorization_endpoint}, "
                f"token: {token_endpoint}, "
                f"userinfo: {userinfo_endpoint}"
            )

        return DiscoveryDocument(
            authorization_endpoint=authorization_endpoint,
            token_endpoint=token_endpoint,
            userinfo_endpoint=userinfo_endpoint,
        )

