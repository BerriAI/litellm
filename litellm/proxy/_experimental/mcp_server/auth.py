from typing import Optional

from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.types import Scope

from litellm._logging import verbose_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth


class UserAPIKeyAuthMCP:
    """
    Class to handle Authentication for MCP requests

    Utilizes the main `user_api_key_auth` function to validate the request
    """

    LITELLM_API_KEY_HEADER_NAME_PRIMARY = "x-litellm-api-key"
    LITELLM_API_KEY_HEADER_NAME_SECONDARY = "Authorization"

    @staticmethod
    async def user_api_key_auth_mcp(scope: Scope) -> UserAPIKeyAuth:
        """
        Validate and extract headers from the ASGI scope for MCP requests.

        Args:
            scope: ASGI scope containing request information

        Returns:
            UserAPIKeyAuth containing validated authentication information

        Raises:
            HTTPException: If headers are invalid or missing required headers
        """
        headers = UserAPIKeyAuthMCP._safe_get_headers_from_scope(scope)
        litellm_api_key = (
            UserAPIKeyAuthMCP.get_litellm_api_key_from_headers(headers) or ""
        )

        validated_user_api_key_auth = await user_api_key_auth(
            api_key=litellm_api_key, request=Request(scope=scope)
        )

        return validated_user_api_key_auth

    @staticmethod
    def get_litellm_api_key_from_headers(headers: Headers) -> Optional[str]:
        """
        Get the Litellm API key from the headers using case-insensitive lookup

        1. Check if `x-litellm-api-key` is in the headers
        2. If not, check if `Authorization` is in the headers

        Args:
            headers: Starlette Headers object that handles case insensitivity
        """
        # Headers object handles case insensitivity automatically
        api_key = headers.get(UserAPIKeyAuthMCP.LITELLM_API_KEY_HEADER_NAME_PRIMARY)
        if api_key:
            return api_key

        auth_header = headers.get(
            UserAPIKeyAuthMCP.LITELLM_API_KEY_HEADER_NAME_SECONDARY
        )
        if auth_header:
            return auth_header

        return None

    @staticmethod
    def _safe_get_headers_from_scope(scope: Scope) -> Headers:
        """
        Safely extract headers from ASGI scope using Starlette's Headers class
        which handles case insensitivity and proper header parsing.

        ASGI headers are in format: List[List[bytes, bytes]]
        We need to convert them to the format Headers expects.
        """
        try:
            # ASGI headers are list of [name: bytes, value: bytes] pairs
            raw_headers = scope.get("headers", [])
            # Convert bytes to strings and create dict for Headers constructor
            headers_dict = {
                name.decode("latin-1"): value.decode("latin-1")
                for name, value in raw_headers
            }
            return Headers(headers_dict)
        except Exception as e:
            verbose_logger.exception(f"Error getting headers from scope: {e}")
            # Return empty Headers object with empty dict
            return Headers({})
