from typing import List, Optional

from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.types import Scope

from litellm._logging import verbose_logger
from litellm.proxy._types import LiteLLM_TeamTableCachedObj, UserAPIKeyAuth
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
        # Debug HTTPS connection if verbose logging is enabled
        UserAPIKeyAuthMCP._debug_https_connection(scope)

        headers = UserAPIKeyAuthMCP._safe_get_headers_from_scope(scope)
        litellm_api_key = (
            UserAPIKeyAuthMCP.get_litellm_api_key_from_headers(headers) or ""
        )

        # Create a proper Request object with mock body method to avoid ASGI receive channel issues
        request = Request(scope=scope)

        # Mock the body method to return empty dict as JSON bytes
        # This prevents "Receive channel has not been made available" error
        async def mock_body():
            return b"{}"  # Empty JSON object as bytes

        request.body = mock_body  # type: ignore

        validated_user_api_key_auth = await user_api_key_auth(
            api_key=litellm_api_key, request=request
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
        # Debug: Log all available headers for HTTPS debugging
        all_headers = dict(headers.items()) if headers else {}
        verbose_logger.debug(f"MCP Auth: Available headers: {list(all_headers.keys())}")

        # Headers object handles case insensitivity automatically
        api_key = headers.get(UserAPIKeyAuthMCP.LITELLM_API_KEY_HEADER_NAME_PRIMARY)
        if api_key:
            verbose_logger.debug(
                f"MCP Auth: Found API key via {UserAPIKeyAuthMCP.LITELLM_API_KEY_HEADER_NAME_PRIMARY}"
            )
            return api_key

        auth_header = headers.get(
            UserAPIKeyAuthMCP.LITELLM_API_KEY_HEADER_NAME_SECONDARY
        )
        if auth_header:
            verbose_logger.debug(
                f"MCP Auth: Found API key via {UserAPIKeyAuthMCP.LITELLM_API_KEY_HEADER_NAME_SECONDARY}"
            )
            return auth_header

        # Fallback: Check for common proxy headers that might contain the auth info
        proxy_headers = [
            "x-forwarded-authorization",
            "x-original-authorization",
            "x-real-authorization",
            "x-forwarded-api-key",
            "x-original-api-key",
        ]

        for proxy_header in proxy_headers:
            proxy_value = headers.get(proxy_header)
            if proxy_value:
                verbose_logger.debug(
                    f"MCP Auth: Found API key via proxy header {proxy_header}"
                )
                return proxy_value

        verbose_logger.debug("MCP Auth: No API key found in any headers")
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

            # Debug logging for HTTPS header issues
            verbose_logger.debug(
                f"MCP Auth: Raw ASGI headers count: {len(raw_headers)}"
            )
            verbose_logger.debug(
                f"MCP Auth: ASGI scope type: {scope.get('type', 'unknown')}"
            )
            verbose_logger.debug(
                f"MCP Auth: ASGI scheme: {scope.get('scheme', 'unknown')}"
            )

            # Convert bytes to strings and create dict for Headers constructor
            headers_dict = {}
            for name, value in raw_headers:
                try:
                    name_str = name.decode("latin-1")
                    value_str = value.decode("latin-1")
                    headers_dict[name_str] = value_str
                    verbose_logger.debug(
                        f"MCP Auth: Header {name_str}: {value_str[:50]}{'...' if len(value_str) > 50 else ''}"
                    )
                except Exception as decode_error:
                    verbose_logger.warning(
                        f"MCP Auth: Failed to decode header {name}: {decode_error}"
                    )

            verbose_logger.debug(f"MCP Auth: Total headers parsed: {len(headers_dict)}")
            return Headers(headers_dict)
        except Exception as e:
            verbose_logger.exception(f"Error getting headers from scope: {e}")
            # Return empty Headers object with empty dict
            return Headers({})

    @staticmethod
    async def get_allowed_mcp_servers(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ) -> List[str]:
        """
        Apply least privilege
        """
        from typing import List

        allowed_mcp_servers: List[str] = []
        allowed_mcp_servers_for_key = (
            await UserAPIKeyAuthMCP._get_allowed_mcp_servers_for_key(user_api_key_auth)
        )
        allowed_mcp_servers_for_team = (
            await UserAPIKeyAuthMCP._get_allowed_mcp_servers_for_team(user_api_key_auth)
        )

        #########################################################
        # If team has mcp_servers, then key must have a subset of the team's mcp_servers
        #########################################################
        if len(allowed_mcp_servers_for_team) > 0:
            for _mcp_server in allowed_mcp_servers_for_key:
                if _mcp_server in allowed_mcp_servers_for_team:
                    allowed_mcp_servers.append(_mcp_server)
        else:
            allowed_mcp_servers = allowed_mcp_servers_for_key

        return list(set(allowed_mcp_servers))

    @staticmethod
    async def _get_allowed_mcp_servers_for_key(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ) -> List[str]:
        from litellm.proxy.proxy_server import prisma_client

        if user_api_key_auth is None:
            return []

        if user_api_key_auth.object_permission_id is None:
            return []

        if prisma_client is None:
            verbose_logger.debug("prisma_client is None")
            return []

        key_object_permission = (
            await prisma_client.db.litellm_objectpermissiontable.find_unique(
                where={"object_permission_id": user_api_key_auth.object_permission_id},
            )
        )
        if key_object_permission is None:
            return []

        return key_object_permission.mcp_servers or []

    @staticmethod
    async def _get_allowed_mcp_servers_for_team(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ) -> List[str]:
        """
        The `object_permission` for a team is not stored on the user_api_key_auth object

        first we check if the team has a object_permission_id attached
            - if it does then we look up the object_permission for the team
        """
        from litellm.proxy.auth.auth_checks import get_team_object
        from litellm.proxy.proxy_server import (
            prisma_client,
            proxy_logging_obj,
            user_api_key_cache,
        )

        if user_api_key_auth is None:
            return []

        if user_api_key_auth.team_id is None:
            return []

        team_obj: LiteLLM_TeamTableCachedObj = await get_team_object(
            team_id=user_api_key_auth.team_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=None,
            proxy_logging_obj=proxy_logging_obj,
            check_cache_only=True,
        )
        if prisma_client is None:
            verbose_logger.debug("prisma_client is None")
            return []

        object_permissions = team_obj.object_permission
        if object_permissions is None:
            return []

        return object_permissions.mcp_servers or []

    @staticmethod
    def _debug_https_connection(scope: Scope) -> None:
        """
        Debug HTTPS connection and proxy configuration issues
        """
        scheme = scope.get("scheme", "unknown")
        server = scope.get("server", ("unknown", "unknown"))
        client = scope.get("client", ("unknown", "unknown"))

        verbose_logger.debug(f"MCP Auth HTTPS Debug:")
        verbose_logger.debug(f"  - Scheme: {scheme}")
        verbose_logger.debug(f"  - Server: {server[0]}:{server[1]}")
        verbose_logger.debug(f"  - Client: {client[0]}:{client[1]}")

        # Check for common proxy indicators
        raw_headers = scope.get("headers", [])
        proxy_indicators = [
            b"x-forwarded-for",
            b"x-forwarded-proto",
            b"x-forwarded-host",
            b"x-real-ip",
            b"cf-connecting-ip",  # Cloudflare
            b"x-cluster-client-ip",  # GKE
            b"x-forwarded-prefix",
            b"forwarded",
        ]

        found_proxy_headers = []
        for name, value in raw_headers:
            if name.lower() in proxy_indicators:
                found_proxy_headers.append(
                    (name.decode("latin-1"), value.decode("latin-1"))
                )

        if found_proxy_headers:
            verbose_logger.debug(f"  - Proxy headers detected: {found_proxy_headers}")
        else:
            verbose_logger.debug("  - No proxy headers detected")

        # Check if this looks like a direct HTTPS connection vs proxied
        if scheme == "https" and not found_proxy_headers:
            verbose_logger.debug("  - Direct HTTPS connection detected")
        elif scheme == "http" and found_proxy_headers:
            verbose_logger.debug(
                "  - Proxied connection detected (HTTPS terminated at proxy)"
            )
        elif scheme == "https" and found_proxy_headers:
            verbose_logger.debug(
                "  - HTTPS connection through proxy (headers should be preserved)"
            )
        else:
            verbose_logger.debug(
                f"  - Unknown connection type (scheme={scheme}, proxy_headers={bool(found_proxy_headers)})"
            )
