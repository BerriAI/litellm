import os
from typing import Dict, Optional, Tuple
from urllib.parse import urlencode

import httpx
from authlib.oauth2.auth import ClientAuth

from litellm._logging import verbose_proxy_logger
from litellm.identity import build_user_api_key_auth_from_oauth2_response
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth


class Oauth2Handler:
    """
    Handles OAuth2 token validation.
    """

    @staticmethod
    def _is_introspection_endpoint(
        token_info_endpoint: str,
        oauth_client_id: Optional[str],
        oauth_client_secret: Optional[str],
    ) -> bool:
        """
        Determine if this is an introspection endpoint (requires POST) or token info endpoint (uses GET).

        Args:
            token_info_endpoint: The OAuth2 endpoint URL
            oauth_client_id: OAuth2 client ID
            oauth_client_secret: OAuth2 client secret

        Returns:
            bool: True if this is an introspection endpoint
        """
        return (
            "introspect" in token_info_endpoint.lower()
            and oauth_client_id is not None
            and oauth_client_secret is not None
        )

    @staticmethod
    def _prepare_introspection_request(
        token: str,
        oauth_client_id: Optional[str],
        oauth_client_secret: Optional[str],
    ) -> Tuple[Dict[str, str], str]:
        """
        Prepare headers and form body for OAuth2 introspection (RFC 7662).

        Confidential clients (id + secret) authenticate via HTTP Basic;
        public clients (id only) carry ``client_id`` in the body. Both are
        delegated to Authlib's RFC 6749 client-auth encoders.

        Returns:
            Tuple of (headers, form-encoded body string).
        """
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        body = urlencode({"token": token})

        if not oauth_client_id:
            return headers, body

        auth_method = "client_secret_basic" if oauth_client_secret else "none"
        _, headers, body = ClientAuth(
            oauth_client_id, oauth_client_secret, auth_method
        ).prepare("POST", "", headers, body)
        return headers, body

    @staticmethod
    def _prepare_token_info_request(token: str) -> Dict[str, str]:
        """
        Prepare headers for generic token info endpoint.

        Args:
            token: The OAuth2 token to validate

        Returns:
            Dict of headers for the token info request
        """
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    @staticmethod
    async def check_oauth2_token(token: str) -> UserAPIKeyAuth:
        """
        Makes a request to the token introspection endpoint to validate the OAuth2 token.

        This function implements OAuth2 token introspection according to RFC 7662.
        It supports both generic token info endpoints (GET) and OAuth2 introspection endpoints (POST).

        Args:
            token (str): The OAuth2 token to validate.

        Returns:
            UserAPIKeyAuth: If the token is valid, containing user information.

        Raises:
            ValueError: If the token is invalid, the request fails, or the token info endpoint is not set.
        """
        from litellm.proxy.proxy_server import premium_user

        if premium_user is not True:
            raise ValueError(
                "Oauth2 token validation is only available for premium users"
                + CommonProxyErrors.not_premium_user.value
            )

        verbose_proxy_logger.debug(
            "Oauth2 token validation for token=[set=%s]", token is not None
        )

        # Get the token info endpoint from environment variable
        token_info_endpoint = os.getenv("OAUTH_TOKEN_INFO_ENDPOINT")
        user_id_field_name = os.environ.get("OAUTH_USER_ID_FIELD_NAME", "sub")
        user_role_field_name = os.environ.get("OAUTH_USER_ROLE_FIELD_NAME", "role")
        user_team_id_field_name = os.environ.get(
            "OAUTH_USER_TEAM_ID_FIELD_NAME", "team_id"
        )

        # OAuth2 client credentials for introspection endpoint authentication
        oauth_client_id = os.environ.get("OAUTH_CLIENT_ID")
        oauth_client_secret = os.environ.get("OAUTH_CLIENT_SECRET")

        if not token_info_endpoint:
            raise ValueError(
                "OAUTH_TOKEN_INFO_ENDPOINT environment variable is not set"
            )

        client = get_async_httpx_client(llm_provider=httpxSpecialProvider.Oauth2Check)

        # Determine if this is an introspection endpoint (requires POST) or token info endpoint (uses GET)
        is_introspection_endpoint = Oauth2Handler._is_introspection_endpoint(
            token_info_endpoint=token_info_endpoint,
            oauth_client_id=oauth_client_id,
            oauth_client_secret=oauth_client_secret,
        )

        try:
            if is_introspection_endpoint:
                # OAuth2 Token Introspection (RFC 7662) - requires POST with form data
                verbose_proxy_logger.debug("Using OAuth2 introspection endpoint (POST)")

                headers, introspection_body = (
                    Oauth2Handler._prepare_introspection_request(
                        token=token,
                        oauth_client_id=oauth_client_id,
                        oauth_client_secret=oauth_client_secret,
                    )
                )

                response = await client.post(
                    token_info_endpoint,
                    headers=headers,
                    content=introspection_body,
                )
            else:
                # Generic token info endpoint - uses GET with Bearer token
                verbose_proxy_logger.debug("Using generic token info endpoint (GET)")
                headers = Oauth2Handler._prepare_token_info_request(token=token)
                response = await client.get(token_info_endpoint, headers=headers)

            # if it's a bad token we expect it to raise an HTTPStatusError
            response.raise_for_status()

            # If we get here, the request was successful
            data = response.json()

            verbose_proxy_logger.debug(
                "Oauth2 token validation for token=%s, response from endpoint=%s",
                token,
                data,
            )

            # For introspection endpoints, check if token is active
            if is_introspection_endpoint and not data.get("active", True):
                raise ValueError("Token is not active")

            return build_user_api_key_auth_from_oauth2_response(
                token=token,
                response_data=data,
                user_id_field_name=user_id_field_name,
                user_role_field_name=user_role_field_name,
                user_team_id_field_name=user_team_id_field_name,
            )
        except httpx.HTTPStatusError as e:
            # This will catch any 4xx or 5xx errors
            raise ValueError(f"Oauth 2.0 Token validation failed: {e}")
        except Exception as e:
            # This will catch any other errors (like network issues)
            raise ValueError(f"An error occurred during token validation: {e}")
