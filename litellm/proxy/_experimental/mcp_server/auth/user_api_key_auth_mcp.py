from typing import Dict, List, Optional, Set, Tuple

from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.types import Scope

from litellm._logging import verbose_logger
from litellm.proxy._types import LiteLLM_TeamTable, SpecialHeaders, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth


class MCPRequestHandler:
    """
    Class to handle MCP request processing, including:
    1. Authentication via LiteLLM API keys
    2. MCP server configuration and routing
    3. Header extraction and validation

    Utilizes the main `user_api_key_auth` function to validate authentication
    """

    LITELLM_API_KEY_HEADER_NAME_PRIMARY = SpecialHeaders.custom_litellm_api_key.value
    LITELLM_API_KEY_HEADER_NAME_SECONDARY = SpecialHeaders.openai_authorization.value

    # This is the header to use if you want LiteLLM to use this header for authenticating to the MCP server
    LITELLM_MCP_AUTH_HEADER_NAME = SpecialHeaders.mcp_auth.value

    LITELLM_MCP_SERVERS_HEADER_NAME = SpecialHeaders.mcp_servers.value

    LITELLM_MCP_ACCESS_GROUPS_HEADER_NAME = SpecialHeaders.mcp_access_groups.value

    # MCP Protocol Version header
    MCP_PROTOCOL_VERSION_HEADER_NAME = "MCP-Protocol-Version"

    @staticmethod
    async def process_mcp_request(
        scope: Scope,
    ) -> Tuple[
        UserAPIKeyAuth,
        Optional[str],
        Optional[List[str]],
        Optional[Dict[str, Dict[str, str]]],
        Optional[Dict[str, str]],
        Optional[Dict[str, str]],
    ]:
        """
        Process and validate MCP request headers from the ASGI scope.
        This includes:
        1. Extracting and validating authentication headers
        2. Processing MCP server configuration
        3. Handling MCP-specific headers
        4. Handling oauth2 headers
        5. Raw headers - allows forwarding specific headers to the MCP server, specified by the admin.

        Args:
            scope: ASGI scope containing request information

        Returns:
            UserAPIKeyAuth containing validated authentication information
            mcp_auth_header: Optional[str] MCP auth header to be passed to the MCP server (deprecated)
            mcp_servers: Optional[List[str]] List of MCP servers and access groups to use
            mcp_server_auth_headers: Optional[Dict[str, str]] Server-specific auth headers in format {server_alias: auth_value}
            oauth2_headers: Optional[Dict[str, str]] OAuth2 headers
            raw_headers: Optional[Dict[str, str]] Raw headers to be forwarded to the MCP server
        Raises:
            HTTPException: If headers are invalid or missing required headers
        """
        headers = MCPRequestHandler._safe_get_headers_from_scope(scope)
        litellm_api_key = (
            MCPRequestHandler.get_litellm_api_key_from_headers(headers) or ""
        )

        # Get the old mcp_auth_header for backward compatibility
        mcp_auth_header = MCPRequestHandler._get_mcp_auth_header_from_headers(headers)

        # Get the new server-specific auth headers
        mcp_server_auth_headers = (
            MCPRequestHandler._get_mcp_server_auth_headers_from_headers(headers)
        )

        # Get the oauth2 headers
        oauth2_headers = MCPRequestHandler._get_oauth2_headers_from_headers(headers)

        # Parse MCP servers from header
        mcp_servers_header = headers.get(
            MCPRequestHandler.LITELLM_MCP_SERVERS_HEADER_NAME
        )
        verbose_logger.debug(f"Raw MCP servers header: {mcp_servers_header}")
        mcp_servers = None
        if mcp_servers_header is not None:
            try:
                mcp_servers = [
                    s.strip() for s in mcp_servers_header.split(",") if s.strip()
                ]
                verbose_logger.debug(f"Parsed MCP servers: {mcp_servers}")
            except Exception as e:
                verbose_logger.debug(f"Error parsing mcp_servers header: {e}")
                mcp_servers = None
            if mcp_servers_header == "" or (
                mcp_servers is not None and len(mcp_servers) == 0
            ):
                mcp_servers = []
        # Create a proper Request object with mock body method to avoid ASGI receive channel issues
        request = Request(scope=scope)

        async def mock_body():
            return b"{}"

        request.body = mock_body  # type: ignore
        if ".well-known" in str(request.url):  # public routes
            validated_user_api_key_auth = UserAPIKeyAuth()
        # elif litellm_api_key == "":
        #     from fastapi import HTTPException

        #     raise HTTPException(
        #         status_code=401,
        #         detail="LiteLLM API key is missing. Please add it or use OAuth authentication.",
        #         headers={
        #             "WWW-Authenticate": f'Bearer resource_metadata=f"{request.base_url}/.well-known/oauth-protected-resource"',
        #         },
        #     )
        else:
            validated_user_api_key_auth = await user_api_key_auth(
                api_key=litellm_api_key, request=request
            )

        return (
            validated_user_api_key_auth,
            mcp_auth_header,
            mcp_servers,
            mcp_server_auth_headers,
            oauth2_headers,
            dict(headers),
        )

    @staticmethod
    def _get_mcp_auth_header_from_headers(headers: Headers) -> Optional[str]:
        """
        Get the header passed to LiteLLM to pass to downstream MCP servers

        By default litellm will check for the header `x-mcp-auth` by setting one of the following:
            1. `LITELLM_MCP_CLIENT_SIDE_AUTH_HEADER_NAME` as an environment variable
            2. `mcp_client_side_auth_header_name` in the general settings on the config.yaml file

        Support this auth: https://docs.litellm.ai/docs/mcp#using-your-mcp-with-client-side-credentials

        If you want to use a different header name, you can set the `LITELLM_MCP_CLIENT_SIDE_AUTH_HEADER_NAME` in the secret manager or `mcp_client_side_auth_header_name` in the general settings.

        DEPRECATED: This method is deprecated in favor of server-specific auth headers using the format x-mcp-{{server_alias}}-{{header_name}} instead.
        """
        mcp_client_side_auth_header_name: str = (
            MCPRequestHandler._get_mcp_client_side_auth_header_name()
        )
        auth_header = headers.get(mcp_client_side_auth_header_name)
        if auth_header:
            verbose_logger.warning(
                f"The '{mcp_client_side_auth_header_name}' header is deprecated. "
                f"Please use server-specific auth headers in the format 'x-mcp-{{server_alias}}-{{header_name}}' instead."
            )
        return auth_header

    @staticmethod
    def _get_mcp_server_auth_headers_from_headers(
        headers: Headers,
    ) -> Dict[str, Dict[str, str]]:
        """
        Parse server-specific MCP auth headers from the request headers.

        Looks for headers in the format: x-mcp-{server_alias}-{header_name}
        Examples:
        - x-mcp-github-authorization: Bearer token123
        - x-mcp-zapier-x-api-key: api_key_456
        - x-mcp-deepwiki-authorization: Basic base64_encoded_creds

        Returns:
            Dict[str, Dict[str, str]]: Mapping of server alias to header dict
        """
        server_auth_headers: Dict[str, Dict[str, str]] = {}
        prefix = "x-mcp-"

        for header_name, header_value in headers.items():
            if header_name.lower().startswith(prefix):
                # Skip the access groups header as it's not a server auth header
                if (
                    header_name.lower()
                    == MCPRequestHandler.LITELLM_MCP_ACCESS_GROUPS_HEADER_NAME.lower()
                    or header_name.lower()
                    == MCPRequestHandler.LITELLM_MCP_SERVERS_HEADER_NAME.lower()
                ):
                    continue

                # Extract server_alias and header_name from x-mcp-{server_alias}-{header_name}
                remaining = header_name[len(prefix) :].lower()
                if "-" in remaining:
                    # Split on the first dash to separate server_alias from header_name
                    parts = remaining.split("-", 1)
                    if len(parts) == 2:
                        server_alias, auth_header_name = parts

                        # Convert common header names to proper case
                        if auth_header_name == "authorization":
                            auth_header_name = "Authorization"

                        # Initialize server dict if not exists
                        if server_alias not in server_auth_headers:
                            server_auth_headers[server_alias] = {}

                        server_auth_headers[server_alias][
                            auth_header_name
                        ] = header_value
                        verbose_logger.debug(
                            f"Found server auth header: {server_alias} -> {auth_header_name}: {header_value[:10]}..."
                        )

        return server_auth_headers

    @staticmethod
    def _get_oauth2_headers_from_headers(headers: Headers) -> Dict[str, str]:
        """
        Get the oauth2 headers from the request headers.
        """
        oauth2_headers = {}
        for header_name, header_value in headers.items():
            if header_name.lower().startswith("authorization"):
                oauth2_headers["Authorization"] = header_value
        return oauth2_headers

    @staticmethod
    def _get_mcp_client_side_auth_header_name() -> str:
        """
        Get the header name used to pass the MCP auth header to the MCP server

        By default litellm will check for the header `x-mcp-auth` by setting one of the following:
            1. `LITELLM_MCP_CLIENT_SIDE_AUTH_HEADER_NAME` as an environment variable
            2. `mcp_client_side_auth_header_name` in the general settings on the config.yaml file
        """
        from litellm.proxy.proxy_server import general_settings
        from litellm.secret_managers.main import get_secret_str

        MCP_CLIENT_SIDE_AUTH_HEADER_NAME: str = (
            MCPRequestHandler.LITELLM_MCP_AUTH_HEADER_NAME
        )
        if get_secret_str("LITELLM_MCP_CLIENT_SIDE_AUTH_HEADER_NAME") is not None:
            MCP_CLIENT_SIDE_AUTH_HEADER_NAME = (
                get_secret_str("LITELLM_MCP_CLIENT_SIDE_AUTH_HEADER_NAME")
                or MCP_CLIENT_SIDE_AUTH_HEADER_NAME
            )
        elif general_settings.get("mcp_client_side_auth_header_name") is not None:
            MCP_CLIENT_SIDE_AUTH_HEADER_NAME = (
                general_settings.get("mcp_client_side_auth_header_name")
                or MCP_CLIENT_SIDE_AUTH_HEADER_NAME
            )
        return MCP_CLIENT_SIDE_AUTH_HEADER_NAME

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
        api_key = headers.get(MCPRequestHandler.LITELLM_API_KEY_HEADER_NAME_PRIMARY)
        if api_key:
            return api_key

        auth_header = headers.get(
            MCPRequestHandler.LITELLM_API_KEY_HEADER_NAME_SECONDARY
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
        except (UnicodeDecodeError, AttributeError, TypeError) as e:
            verbose_logger.exception(f"Error getting headers from scope: {e}")
            # Return empty Headers object with empty dict
            return Headers({})

    @staticmethod
    async def get_allowed_mcp_servers(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ) -> List[str]:
        """
        Get list of allowed MCP servers for the given user/key based on permissions

        Returns:
            List[str]: List of allowed MCP servers by server id
        """
        from typing import List

        try:
            allowed_mcp_servers: List[str] = []
            allowed_mcp_servers_for_key = (
                await MCPRequestHandler._get_allowed_mcp_servers_for_key(
                    user_api_key_auth
                )
            )
            allowed_mcp_servers_for_team = (
                await MCPRequestHandler._get_allowed_mcp_servers_for_team(
                    user_api_key_auth
                )
            )

            #########################################################
            # If team has mcp_servers, handle inheritance and intersection logic
            #########################################################
            if len(allowed_mcp_servers_for_team) > 0:
                if len(allowed_mcp_servers_for_key) > 0:
                    # Key has its own MCP permissions - use intersection with team permissions
                    for _mcp_server in allowed_mcp_servers_for_key:
                        if _mcp_server in allowed_mcp_servers_for_team:
                            allowed_mcp_servers.append(_mcp_server)
                else:
                    # Key has no MCP permissions - inherit from team
                    allowed_mcp_servers = allowed_mcp_servers_for_team
            else:
                allowed_mcp_servers = allowed_mcp_servers_for_key

            return list(set(allowed_mcp_servers))
        except Exception as e:
            verbose_logger.warning(f"Failed to get allowed MCP servers: {str(e)}")
            return []

    @staticmethod
    async def _get_key_object_permission(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ):
        """Helper to get key object_permission from cache or DB."""
        from litellm.proxy.auth.auth_checks import get_object_permission
        from litellm.proxy.proxy_server import (
            prisma_client,
            proxy_logging_obj,
            user_api_key_cache,
        )

        if not user_api_key_auth:
            return None

        # Already loaded
        if user_api_key_auth.object_permission:
            return user_api_key_auth.object_permission

        # Need to fetch from DB
        if user_api_key_auth.object_permission_id and prisma_client:
            return await get_object_permission(
                object_permission_id=user_api_key_auth.object_permission_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=user_api_key_auth.parent_otel_span,
                proxy_logging_obj=proxy_logging_obj,
            )

        return None

    @staticmethod
    async def _get_team_object_permission(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ):
        """Helper to get team object_permission from cache or DB."""
        from litellm.proxy.auth.auth_checks import (
            get_object_permission,
            get_team_object,
        )
        from litellm.proxy.proxy_server import (
            prisma_client,
            proxy_logging_obj,
            user_api_key_cache,
        )

        if not user_api_key_auth or not user_api_key_auth.team_id or not prisma_client:
            return None

        # First get the team object (which may have object_permission already loaded)
        team_obj: Optional[LiteLLM_TeamTable] = await get_team_object(
            team_id=user_api_key_auth.team_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=user_api_key_auth.parent_otel_span,
            proxy_logging_obj=proxy_logging_obj,
        )

        if not team_obj:
            return None

        # Already loaded
        if team_obj.object_permission:
            return team_obj.object_permission

        # Need to fetch from DB using object_permission_id
        if team_obj.object_permission_id:
            return await get_object_permission(
                object_permission_id=team_obj.object_permission_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=user_api_key_auth.parent_otel_span,
                proxy_logging_obj=proxy_logging_obj,
            )

        return None

    @staticmethod
    async def get_allowed_tools_for_server(
        server_id: str,
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ) -> Optional[List[str]]:
        """
        Get list of allowed tool names for a specific server based on key/team permissions.
        Follows same inheritance logic as get_allowed_mcp_servers.

        Args:
            server_id: Server ID to check permissions for
            user_api_key_auth: User auth

        Returns:
            List[str] if restrictions exist, None if no restrictions (allow all)
        """
        if not user_api_key_auth:
            return None

        try:
            # Get key and team object permissions
            key_obj_perm = await MCPRequestHandler._get_key_object_permission(
                user_api_key_auth
            )
            team_obj_perm = await MCPRequestHandler._get_team_object_permission(
                user_api_key_auth
            )

            # Extract tool permissions for this server
            key_tools = (
                key_obj_perm.mcp_tool_permissions.get(server_id)
                if key_obj_perm and key_obj_perm.mcp_tool_permissions
                else None
            )
            team_tools = (
                team_obj_perm.mcp_tool_permissions.get(server_id)
                if team_obj_perm and team_obj_perm.mcp_tool_permissions
                else None
            )

            # Apply same inheritance logic as get_allowed_mcp_servers
            if team_tools:
                if key_tools:
                    # Both have restrictions → intersection
                    return list(set(team_tools) & set(key_tools))
                else:
                    # Only team has restrictions → inherit from team
                    return team_tools
            else:
                # No team restrictions → use key restrictions
                return key_tools

        except Exception as e:
            verbose_logger.warning(f"Failed to get allowed tools for server: {str(e)}")
            return None

    @staticmethod
    async def is_tool_allowed_for_server(
        tool_name: str,
        server_id: str,
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ) -> bool:
        """
        Check if a specific tool is allowed for a server based on key/team permissions.

        Args:
            tool_name: Name of the tool to check
            server_id: Server ID
            user_api_key_auth: User auth

        Returns:
            True if allowed, False if blocked
        """
        allowed_tools = await MCPRequestHandler.get_allowed_tools_for_server(
            server_id=server_id,
            user_api_key_auth=user_api_key_auth,
        )

        # None means no restrictions (allow all)
        if allowed_tools is None:
            return True

        # Empty list means no tools allowed
        if not allowed_tools:
            return False

        # Check if tool is in allowed list
        return tool_name in allowed_tools

    @staticmethod
    def is_tool_allowed(
        allowed_mcp_servers: List[str],
        server_name: str,
    ) -> bool:
        """
        Check if the tool is allowed for the given user/key based on permissions
        """
        if len(allowed_mcp_servers) == 0:
            return True
        elif server_name in allowed_mcp_servers:
            return True
        return False

    @staticmethod
    async def _get_allowed_mcp_servers_for_key(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ) -> List[str]:
        from litellm.proxy.auth.auth_checks import get_object_permission
        from litellm.proxy.proxy_server import (
            prisma_client,
            proxy_logging_obj,
            user_api_key_cache,
        )

        if user_api_key_auth is None:
            return []

        if user_api_key_auth.object_permission_id is None:
            return []

        if prisma_client is None:
            verbose_logger.debug("prisma_client is None")
            return []

        try:
            key_object_permission = await get_object_permission(
                object_permission_id=user_api_key_auth.object_permission_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=user_api_key_auth.parent_otel_span,
                proxy_logging_obj=proxy_logging_obj,
            )
            if key_object_permission is None:
                return []

            # Get direct MCP servers
            direct_mcp_servers = key_object_permission.mcp_servers or []

            # Get MCP servers from access groups
            access_group_servers = (
                await MCPRequestHandler._get_mcp_servers_from_access_groups(
                    key_object_permission.mcp_access_groups or []
                )
            )

            # Combine both lists
            all_servers = direct_mcp_servers + access_group_servers
            return list(set(all_servers))
        except Exception as e:
            verbose_logger.warning(
                f"Failed to get allowed MCP servers for key: {str(e)}"
            )
            return []

    @staticmethod
    async def _get_allowed_mcp_servers_for_team(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ) -> List[str]:
        """
        Get allowed MCP servers for a team.

        Uses the helper _get_team_object_permission which:
        1. First checks if object_permission is already loaded on the team
        2. If not, fetches from DB using object_permission_id if it exists
        """
        if user_api_key_auth is None:
            return []

        if user_api_key_auth.team_id is None:
            return []

        try:
            # Use the helper method that properly handles fetching from DB if needed
            object_permissions = await MCPRequestHandler._get_team_object_permission(
                user_api_key_auth
            )

            if object_permissions is None:
                return []

            # Get direct MCP servers
            direct_mcp_servers = object_permissions.mcp_servers or []

            # Get MCP servers from access groups
            access_group_servers = (
                await MCPRequestHandler._get_mcp_servers_from_access_groups(
                    object_permissions.mcp_access_groups or []
                )
            )

            # Combine both lists
            all_servers = direct_mcp_servers + access_group_servers
            return list(set(all_servers))
        except Exception as e:
            verbose_logger.warning(
                f"Failed to get allowed MCP servers for team: {str(e)}"
            )
            return []

    @staticmethod
    def _get_config_server_ids_for_access_groups(
        config_mcp_servers, access_groups: List[str]
    ) -> Set[str]:
        """
        Helper to get server_ids from config-loaded servers that match any of the given access groups.
        """
        server_ids: Set[str] = set()
        for server_id, server in config_mcp_servers.items():
            if server.access_groups:
                if any(group in server.access_groups for group in access_groups):
                    server_ids.add(server_id)
        return server_ids

    @staticmethod
    async def _get_db_server_ids_for_access_groups(
        prisma_client, access_groups: List[str]
    ) -> Set[str]:
        """
        Helper to get server_ids from DB servers that match any of the given access groups.
        """
        server_ids: Set[str] = set()
        if access_groups and prisma_client is not None:
            try:
                mcp_servers = await prisma_client.db.litellm_mcpservertable.find_many(
                    where={"mcp_access_groups": {"hasSome": access_groups}}
                )
                for server in mcp_servers:
                    server_ids.add(server.server_id)
            except Exception as e:
                verbose_logger.debug(
                    f"Error getting MCP servers from access groups: {e}"
                )
        return server_ids

    @staticmethod
    async def _get_mcp_servers_from_access_groups(
        access_groups: List[str],
    ) -> List[str]:
        """
        Resolve MCP access groups to server IDs by querying BOTH the MCP server table (DB) AND config-loaded servers
        """
        from litellm.proxy.proxy_server import prisma_client

        try:
            # Import here to avoid circular import
            from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
                global_mcp_server_manager,
            )

            # Use the new helper for config-loaded servers
            server_ids = MCPRequestHandler._get_config_server_ids_for_access_groups(
                global_mcp_server_manager.config_mcp_servers, access_groups
            )

            # Use the new helper for DB servers
            db_server_ids = (
                await MCPRequestHandler._get_db_server_ids_for_access_groups(
                    prisma_client, access_groups
                )
            )
            server_ids.update(db_server_ids)

            return list(server_ids)
        except Exception as e:
            verbose_logger.warning(
                f"Failed to get MCP servers from access groups: {str(e)}"
            )
            return []

    @staticmethod
    async def get_mcp_access_groups(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ) -> List[str]:
        """
        Get list of MCP access groups for the given user/key based on permissions
        """
        from typing import List

        access_groups: List[str] = []
        access_groups_for_key = await MCPRequestHandler._get_mcp_access_groups_for_key(
            user_api_key_auth
        )
        access_groups_for_team = (
            await MCPRequestHandler._get_mcp_access_groups_for_team(user_api_key_auth)
        )

        #########################################################
        # If team has access groups, then key must have a subset of the team's access groups
        #########################################################
        if len(access_groups_for_team) > 0:
            for access_group in access_groups_for_key:
                if access_group in access_groups_for_team:
                    access_groups.append(access_group)
        else:
            access_groups = access_groups_for_key

        return list(set(access_groups))

    @staticmethod
    async def _get_mcp_access_groups_for_key(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ) -> List[str]:
        from litellm.proxy.auth.auth_checks import get_object_permission
        from litellm.proxy.proxy_server import (
            prisma_client,
            proxy_logging_obj,
            user_api_key_cache,
        )

        if user_api_key_auth is None:
            return []

        if user_api_key_auth.object_permission_id is None:
            return []

        if prisma_client is None:
            verbose_logger.debug("prisma_client is None")
            return []

        try:
            key_object_permission = await get_object_permission(
                object_permission_id=user_api_key_auth.object_permission_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=user_api_key_auth.parent_otel_span,
                proxy_logging_obj=proxy_logging_obj,
            )
            if key_object_permission is None:
                return []

            return key_object_permission.mcp_access_groups or []
        except Exception as e:
            verbose_logger.warning(f"Failed to get MCP access groups for key: {str(e)}")
            return []

    @staticmethod
    async def _get_mcp_access_groups_for_team(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ) -> List[str]:
        """
        Get MCP access groups for the team
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

        if prisma_client is None:
            verbose_logger.debug("prisma_client is None")
            return []

        try:
            team_obj: Optional[LiteLLM_TeamTable] = await get_team_object(
                team_id=user_api_key_auth.team_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=user_api_key_auth.parent_otel_span,
                proxy_logging_obj=proxy_logging_obj,
            )
            if team_obj is None:
                verbose_logger.debug("team_obj is None")
                return []

            object_permissions = team_obj.object_permission
            if object_permissions is None:
                return []

            return object_permissions.mcp_access_groups or []
        except Exception as e:
            verbose_logger.warning(
                f"Failed to get MCP access groups for team: {str(e)}"
            )
            return []

    @staticmethod
    def get_mcp_access_groups_from_headers(headers: Headers) -> Optional[List[str]]:
        """
        Extract and parse the x-mcp-access-groups header as a list of strings.
        """
        mcp_access_groups_header = headers.get(
            MCPRequestHandler.LITELLM_MCP_ACCESS_GROUPS_HEADER_NAME
        )
        if mcp_access_groups_header is not None:
            try:
                return [
                    s.strip() for s in mcp_access_groups_header.split(",") if s.strip()
                ]
            except Exception:
                return None
        return None

    @staticmethod
    def get_mcp_access_groups_from_scope(scope: Scope) -> Optional[List[str]]:
        """
        Extract and parse the x-mcp-access-groups header from an ASGI scope.
        """
        headers = MCPRequestHandler._safe_get_headers_from_scope(scope)
        return MCPRequestHandler.get_mcp_access_groups_from_headers(headers)
