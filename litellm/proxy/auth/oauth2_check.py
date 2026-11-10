import base64
import os
from typing import Final

import httpx
from pydantic import ConfigDict, TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.core.answers import Introspection
from litellm.proxy.auth.core.identity import (
    InactiveToken,
    IntrospectionNotConfigured,
    OAuth2Identity,
)
from litellm.proxy.auth.core.inputs import OAuth2IntrospectionSettings
from litellm.proxy.auth.selected_core import AUTH_CORE

_TOKEN_INFO_RESPONSE: Final = TypeAdapter(dict[str, object], config=ConfigDict(strict=True, hide_input_in_errors=True))


class OAuth2TokenValidationError(ValueError):
    """The token-info or introspection endpoint did not vouch for the token: it rejected it, reported it
    inactive, or could not be reached."""


class Oauth2Handler:
    """
    Handles OAuth2 token validation.
    """

    @staticmethod
    def _is_introspection_endpoint(
        token_info_endpoint: str,
        oauth_client_id: str | None,
        oauth_client_secret: str | None,
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
        oauth_client_id: str | None,
        oauth_client_secret: str | None,
    ) -> tuple[dict[str, str], dict[str, str]]:
        """
        Prepare headers and data for OAuth2 introspection endpoint (RFC 7662).

        Args:
            token: The OAuth2 token to validate
            oauth_client_id: OAuth2 client ID
            oauth_client_secret: OAuth2 client secret

        Returns:
            Tuple of (headers, data) for the introspection request
        """
        headers: Final = {"Content-Type": "application/x-www-form-urlencoded"}
        data: Final = {"token": token}

        # Add client authentication if credentials are provided
        if oauth_client_id and oauth_client_secret:
            # Use HTTP Basic authentication for client credentials
            credentials: Final = base64.b64encode(f"{oauth_client_id}:{oauth_client_secret}".encode()).decode()
            headers["Authorization"] = f"Basic {credentials}"
        elif oauth_client_id:
            # For public clients, include client_id in the request body
            data["client_id"] = oauth_client_id

        return headers, data

    @staticmethod
    def _prepare_token_info_request(token: str) -> dict[str, str]:
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
        """Validate ``token`` against the configured token-info or RFC 7662 introspection endpoint.

        Raises ``ValueError`` when the endpoint is not configured, and ``OAuth2TokenValidationError``
        when the endpoint rejects the token, reports it inactive, or cannot be reached."""
        verbose_proxy_logger.debug("Oauth2 token validation for token=[set=%s]", bool(token))
        endpoint: Final = _TokenInfoEndpoint.from_env()
        result: Final = await AUTH_CORE.oauth2().introspect(endpoint, token, endpoint.settings())
        match result:
            case OAuth2Identity(user_id=user_id, role=role, team_id=team_id):
                return UserAPIKeyAuth.model_validate(
                    {"api_key": token, "team_id": team_id, "user_id": user_id, "user_role": role}
                )
            case IntrospectionNotConfigured():
                raise ValueError("OAUTH_TOKEN_INFO_ENDPOINT environment variable is not set")
            case InactiveToken():
                raise OAuth2TokenValidationError("An error occurred during token validation: Token is not active")


class _TokenInfoEndpoint:
    def __init__(
        self,
        url: str | None,
        client_id: str | None,
        client_secret: str | None,
        field_names: tuple[str, str, str],
    ) -> None:
        self._url: Final = url
        self._client_id: Final = client_id
        self._client_secret: Final = client_secret
        self._field_names: Final = field_names

    @classmethod
    def from_env(cls) -> "_TokenInfoEndpoint":
        return cls(
            url=os.getenv("OAUTH_TOKEN_INFO_ENDPOINT"),
            client_id=os.environ.get("OAUTH_CLIENT_ID"),
            client_secret=os.environ.get("OAUTH_CLIENT_SECRET"),
            field_names=(
                os.environ.get("OAUTH_USER_ID_FIELD_NAME", "sub"),
                os.environ.get("OAUTH_USER_ROLE_FIELD_NAME", "role"),
                os.environ.get("OAUTH_USER_TEAM_ID_FIELD_NAME", "team_id"),
            ),
        )

    def _is_introspection(self) -> bool:
        return self._url is not None and Oauth2Handler._is_introspection_endpoint(  # pyright: ignore[reportPrivateUsage]  # same module
            token_info_endpoint=self._url,
            oauth_client_id=self._client_id,
            oauth_client_secret=self._client_secret,
        )

    def settings(self) -> OAuth2IntrospectionSettings:
        return OAuth2IntrospectionSettings(
            endpoint_configured=bool(self._url), requires_active=self._is_introspection()
        )

    async def _response(self, url: str, token: str) -> httpx.Response:
        client: Final = get_async_httpx_client(llm_provider=httpxSpecialProvider.Oauth2Check)
        if self._is_introspection():
            verbose_proxy_logger.debug("Using OAuth2 introspection endpoint (POST)")
            headers, data = Oauth2Handler._prepare_introspection_request(  # pyright: ignore[reportPrivateUsage]  # same module
                token=token, oauth_client_id=self._client_id, oauth_client_secret=self._client_secret
            )
            return await client.post(url, headers=headers, data=data)
        verbose_proxy_logger.debug("Using generic token info endpoint (GET)")
        return await client.get(url, headers=Oauth2Handler._prepare_token_info_request(token=token))  # pyright: ignore[reportPrivateUsage]  # same module

    async def introspect(self, token: str) -> Introspection:
        url: Final = self._url or ""
        try:
            response: Final = await self._response(url, token)
            response.raise_for_status()
            data: Final = _TOKEN_INFO_RESPONSE.validate_python(response.json())
        except httpx.HTTPStatusError as e:
            raise OAuth2TokenValidationError(f"Oauth 2.0 Token validation failed: {e}")
        except Exception as e:
            raise OAuth2TokenValidationError(f"An error occurred during token validation: {e}")
        verbose_proxy_logger.debug(
            "Oauth2 token validation for token=%s, response from endpoint=%s",
            token,
            data,
        )
        user_id_field, role_field, team_id_field = self._field_names
        return Introspection(
            active=bool(data.get("active", True)),
            user_id=data.get(user_id_field),
            role=data.get(role_field),
            team_id=data.get(team_id_field),
        )
