"""
MCP Client Manager

This class is responsible for managing MCP clients with support for both SSE and HTTP streamable transports.

This is a Proxy
"""

import asyncio
import datetime
import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast
from urllib.parse import urlparse

from fastapi import HTTPException
from httpx import HTTPStatusError
from mcp.types import CallToolRequestParams as MCPCallToolRequestParams
from mcp.types import CallToolResult
from mcp.types import Tool as MCPTool

import litellm
from litellm._logging import verbose_logger
from litellm.exceptions import BlockedPiiEntityError, GuardrailRaisedException
from litellm.experimental_mcp_client.client import MCPClient
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
    MCPRequestHandler,
)
from litellm.proxy._experimental.mcp_server.utils import (
    add_server_prefix_to_tool_name,
    get_server_name_prefix_tool_mcp,
    get_server_prefix,
    is_tool_name_prefixed,
    normalize_server_name,
    validate_mcp_server_name,
)
from litellm.proxy._types import (
    LiteLLM_MCPServerTable,
    MCPAuthType,
    MCPTransport,
    MCPTransportType,
    UserAPIKeyAuth,
)
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper
from litellm.proxy.utils import ProxyLogging
from litellm.types.llms.custom_http import httpxSpecialProvider
from litellm.types.mcp import MCPAuth, MCPStdioConfig
from litellm.types.mcp_server.mcp_server_manager import (
    MCPInfo,
    MCPOAuthMetadata,
    MCPServer,
)


def _deserialize_json_dict(data: Any) -> Optional[Dict[str, str]]:
    """
    Deserialize optional JSON mappings stored in the database.

    Accepts values kept as JSON strings or materialized dictionaries and
    returns None when the input is empty or cannot be decoded.
    """
    if not data:
        return None

    if isinstance(data, str):
        try:
            return json.loads(data)
        except (json.JSONDecodeError, TypeError):
            # If it's not valid JSON, return as-is (shouldn't happen but safety)
            return None
    else:
        # Already a dictionary
        return data


class MCPServerManager:
    def __init__(self):
        self.registry: Dict[str, MCPServer] = {}
        self.config_mcp_servers: Dict[str, MCPServer] = {}
        """
        eg.
        [
            "server-1": {
                "name": "zapier_mcp_server",
                "url": "https://actions.zapier.com/mcp/sk-ak-2ew3bofIeQIkNoeKIdXrF1Hhhp/sse"
                "transport": "sse",
                "auth_type": "api_key"
            },
            "uuid-2": {
                "name": "google_drive_mcp_server",
                "url": "https://actions.zapier.com/mcp/sk-ak-2ew3bofIeQIkNoeKIdXrF1Hhhp/sse"
            }
        ]
        """

        self.tool_name_to_mcp_server_name_mapping: Dict[str, str] = {}
        """
        {
            "gmail_send_email": "zapier_mcp_server",
        }
        """

    def get_registry(self) -> Dict[str, MCPServer]:
        """
        Get the registered MCP Servers from the registry and union with the config MCP Servers
        """
        return self.config_mcp_servers | self.registry

    async def load_servers_from_config(
        self,
        mcp_servers_config: Dict[str, Any],
        mcp_aliases: Optional[Dict[str, str]] = None,
    ):
        """
        Load the MCP Servers from the config

        Args:
            mcp_servers_config: Dictionary of MCP server configurations
            mcp_aliases: Optional dictionary mapping aliases to server names from litellm_settings
        """
        verbose_logger.debug("Loading MCP Servers from config-----")

        # Track which aliases have been used to ensure only first occurrence is used
        used_aliases = set()

        for server_name, server_config in mcp_servers_config.items():
            validate_mcp_server_name(server_name)
            _mcp_info: Dict[str, Any] = server_config.get("mcp_info", None) or {}
            # Preserve all custom fields from config while setting defaults for core fields
            mcp_info: MCPInfo = _mcp_info.copy()
            # Set default values for core fields if not present
            if "server_name" not in mcp_info:
                mcp_info["server_name"] = server_name
            if "description" not in mcp_info and server_config.get("description"):
                mcp_info["description"] = server_config.get("description")

            # Use alias for name if present, else server_name
            alias = server_config.get("alias", None)

            # Apply mcp_aliases mapping if provided
            if mcp_aliases and alias is None:
                # Check if this server_name has an alias in mcp_aliases
                for alias_name, target_server_name in mcp_aliases.items():
                    if (
                        target_server_name == server_name
                        and alias_name not in used_aliases
                    ):
                        alias = alias_name
                        used_aliases.add(alias_name)
                        verbose_logger.debug(
                            f"Mapped alias '{alias_name}' to server '{server_name}'"
                        )
                        break

            # Create a temporary server object to use with get_server_prefix utility
            temp_server = type(
                "TempServer",
                (),
                {"alias": alias, "server_name": server_name, "server_id": None},
            )()
            name_for_prefix = get_server_prefix(temp_server)

            # Use alias for name if present, else server_name
            alias = server_config.get("alias", None)

            # Apply mcp_aliases mapping if provided
            if mcp_aliases and alias is None:
                # Check if this server_name has an alias in mcp_aliases
                for alias_name, target_server_name in mcp_aliases.items():
                    if (
                        target_server_name == server_name
                        and alias_name not in used_aliases
                    ):
                        alias = alias_name
                        used_aliases.add(alias_name)
                        verbose_logger.debug(
                            f"Mapped alias '{alias_name}' to server '{server_name}'"
                        )
                        break

            # Create a temporary server object to use with get_server_prefix utility
            temp_server = type(
                "TempServer",
                (),
                {"alias": alias, "server_name": server_name, "server_id": None},
            )()
            name_for_prefix = get_server_prefix(temp_server)

            server_url = server_config.get("url", None) or ""
            # Generate stable server ID based on parameters
            server_id = self._generate_stable_server_id(
                server_name=server_name,
                url=server_url,
                transport=server_config.get("transport", MCPTransport.http),
                auth_type=server_config.get("auth_type", None),
                alias=alias,
            )

            auth_type = server_config.get("auth_type", None)
            if server_url and auth_type is not None and auth_type == MCPAuth.oauth2:
                mcp_oauth_metadata = await self._descovery_metadata(
                    server_url=server_url,
                )
            else:
                mcp_oauth_metadata = None

            resolved_scopes = server_config.get("scopes") or (
                mcp_oauth_metadata.scopes if mcp_oauth_metadata else None
            )
            resolved_authorization_url = server_config.get("authorization_url") or (
                mcp_oauth_metadata.authorization_url if mcp_oauth_metadata else None
            )
            resolved_token_url = server_config.get("token_url") or (
                mcp_oauth_metadata.token_url if mcp_oauth_metadata else None
            )
            resolved_registration_url = server_config.get("registration_url") or (
                mcp_oauth_metadata.registration_url if mcp_oauth_metadata else None
            )

            new_server = MCPServer(
                server_id=server_id,
                name=name_for_prefix,
                alias=alias,
                server_name=server_name,
                spec_path=server_config.get("spec_path", None),
                url=server_url,
                command=server_config.get("command", None) or "",
                args=server_config.get("args", None) or [],
                env=server_config.get("env", None) or {},
                # oauth specific fields
                client_id=server_config.get("client_id", None),
                client_secret=server_config.get("client_secret", None),
                scopes=resolved_scopes,
                authorization_url=resolved_authorization_url,
                token_url=resolved_token_url,
                registration_url=resolved_registration_url,
                # TODO: utility fn the default values
                transport=server_config.get("transport", MCPTransport.http),
                auth_type=auth_type,
                authentication_token=server_config.get(
                    "authentication_token", server_config.get("auth_value", None)
                ),
                mcp_info=mcp_info,
                extra_headers=server_config.get("extra_headers", None),
                allowed_tools=server_config.get("allowed_tools", None),
                disallowed_tools=server_config.get("disallowed_tools", None),
                allowed_params=server_config.get("allowed_params", None),
                access_groups=server_config.get("access_groups", None),
                static_headers=server_config.get("static_headers", None),
            )
            self.config_mcp_servers[server_id] = new_server

            # Check if this is an OpenAPI-based server
            spec_path = server_config.get("spec_path", None)
            if spec_path:
                verbose_logger.info(
                    f"Loading OpenAPI spec from {spec_path} for server {server_name}"
                )
                self._register_openapi_tools(
                    spec_path=spec_path,
                    server=new_server,
                    base_url=server_config.get("url", ""),
                )

        verbose_logger.debug(
            f"Loaded MCP Servers: {json.dumps(self.config_mcp_servers, indent=4, default=str)}"
        )

        self.initialize_tool_name_to_mcp_server_name_mapping()

    def _register_openapi_tools(self, spec_path: str, server: MCPServer, base_url: str):
        """
        Register tools from an OpenAPI specification for a given server.

        This creates "virtual" MCP tools from OpenAPI endpoints that are:
        1. Registered in the global tool registry with server prefix
        2. Mapped to the server for routing
        3. Executed via the local tool handler

        Args:
            spec_path: Path to the OpenAPI specification file
            server: The MCPServer instance to register tools for
            base_url: Base URL for API calls
        """
        from litellm.proxy._experimental.mcp_server.openapi_to_mcp_generator import (
            build_input_schema,
            create_tool_function,
        )
        from litellm.proxy._experimental.mcp_server.openapi_to_mcp_generator import (
            get_base_url as get_openapi_base_url,
        )
        from litellm.proxy._experimental.mcp_server.openapi_to_mcp_generator import (
            load_openapi_spec,
        )
        from litellm.proxy._experimental.mcp_server.tool_registry import (
            global_mcp_tool_registry,
        )

        try:
            # Load OpenAPI spec
            spec = load_openapi_spec(spec_path)

            # Use base_url from config if provided, otherwise extract from spec
            if not base_url:
                base_url = get_openapi_base_url(spec)

            verbose_logger.info(
                f"Registering OpenAPI tools for server {server.name} with base URL: {base_url}"
            )

            # Get server prefix for tool naming
            server_prefix = get_server_prefix(server)

            # Build headers from server configuration
            headers = {}

            # Add authentication headers if configured
            if server.authentication_token:
                from litellm.types.mcp import MCPAuth

                if server.auth_type == MCPAuth.bearer_token:
                    headers["Authorization"] = f"Bearer {server.authentication_token}"
                elif server.auth_type == MCPAuth.api_key:
                    headers["Authorization"] = f"ApiKey {server.authentication_token}"
                elif server.auth_type == MCPAuth.basic:
                    headers["Authorization"] = f"Basic {server.authentication_token}"

            # Add any extra headers from server config
            # Note: extra_headers is a List[str] of header names to forward, not a dict
            # For OpenAPI tools, we'll just use the authentication headers
            # If extra_headers were needed, they would be processed separately

            verbose_logger.debug(
                f"Using headers for OpenAPI tools (excluding sensitive values): "
                f"{list(headers.keys())}"
            )

            # Extract and register tools from OpenAPI paths
            paths = spec.get("paths", {})
            registered_count = 0

            verbose_logger.debug(f"Processing {len(paths)} paths from OpenAPI spec")

            for path, path_item in paths.items():
                for method in ["get", "post", "put", "delete", "patch"]:
                    if method not in path_item:
                        continue

                    operation = path_item[method]

                    # Generate tool name (without prefix initially)
                    operation_id = operation.get(
                        "operationId", f"{method}_{path.replace('/', '_')}"
                    )
                    base_tool_name = operation_id.replace(" ", "_").lower()

                    # Add server prefix to tool name
                    prefixed_tool_name = add_server_prefix_to_tool_name(
                        base_tool_name, server_prefix
                    )

                    # Get description
                    description = operation.get(
                        "summary",
                        operation.get("description", f"{method.upper()} {path}"),
                    )

                    # Build input schema using imported function
                    input_schema = build_input_schema(operation)

                    # Create tool function with headers using imported function
                    tool_func = create_tool_function(
                        path, method, operation, base_url, headers=headers
                    )
                    tool_func.__name__ = prefixed_tool_name
                    tool_func.__doc__ = description

                    # Register tool with prefixed name in global registry
                    global_mcp_tool_registry.register_tool(
                        name=prefixed_tool_name,
                        description=description,
                        input_schema=input_schema,
                        handler=tool_func,
                    )

                    # Update tool name to server name mapping (for both prefixed and base names)
                    self.tool_name_to_mcp_server_name_mapping[base_tool_name] = (
                        server_prefix
                    )
                    self.tool_name_to_mcp_server_name_mapping[prefixed_tool_name] = (
                        server_prefix
                    )

                    registered_count += 1
                    verbose_logger.debug(
                        f"Registered OpenAPI tool: {prefixed_tool_name} for server {server.name}"
                    )

            verbose_logger.info(
                f"Successfully registered {registered_count} OpenAPI tools for server {server.name}"
            )

        except Exception as e:
            verbose_logger.error(
                f"Failed to register OpenAPI tools for server {server.name}: {str(e)}"
            )
            raise e

    def remove_server(self, mcp_server: LiteLLM_MCPServerTable):
        """
        Remove a server from the registry
        """
        if mcp_server.server_name in self.get_registry():
            del self.registry[mcp_server.server_name]
            verbose_logger.debug(f"Removed MCP Server: {mcp_server.server_name}")
        elif mcp_server.server_id in self.get_registry():
            del self.registry[mcp_server.server_id]
            verbose_logger.debug(f"Removed MCP Server: {mcp_server.server_id}")
        else:
            verbose_logger.warning(
                f"Server ID {mcp_server.server_id} not found in registry"
            )

    def add_update_server(self, mcp_server: LiteLLM_MCPServerTable):
        try:
            if mcp_server.server_id not in self.get_registry():
                _mcp_info: MCPInfo = mcp_server.mcp_info or {}
                # Use helper to deserialize dictionary
                # Safely access env field which may not exist on Prisma model objects
                env_dict = _deserialize_json_dict(getattr(mcp_server, "env", None))
                static_headers_dict = _deserialize_json_dict(
                    getattr(mcp_server, "static_headers", None)
                )
                credentials_dict = _deserialize_json_dict(
                    getattr(mcp_server, "credentials", None)
                )

                encrypted_auth_value: Optional[str] = None
                if credentials_dict:
                    encrypted_auth_value = credentials_dict.get("auth_value")

                auth_value: Optional[str] = None
                if encrypted_auth_value:
                    auth_value = decrypt_value_helper(
                        value=encrypted_auth_value,
                        key="auth_value",
                    )
                # Use alias for name if present, else server_name
                name_for_prefix = (
                    mcp_server.alias or mcp_server.server_name or mcp_server.server_id
                )
                # Preserve all custom fields from database while setting defaults for core fields
                mcp_info: MCPInfo = _mcp_info.copy()
                # Set default values for core fields if not present
                if "server_name" not in mcp_info:
                    mcp_info["server_name"] = (
                        mcp_server.server_name or mcp_server.server_id
                    )
                if "description" not in mcp_info and mcp_server.description:
                    mcp_info["description"] = mcp_server.description

                new_server = MCPServer(
                    server_id=mcp_server.server_id,
                    name=name_for_prefix,
                    alias=getattr(mcp_server, "alias", None),
                    server_name=getattr(mcp_server, "server_name", None),
                    url=mcp_server.url,
                    transport=cast(MCPTransportType, mcp_server.transport),
                    auth_type=cast(MCPAuthType, mcp_server.auth_type),
                    authentication_token=auth_value,
                    mcp_info=mcp_info,
                    extra_headers=getattr(mcp_server, "extra_headers", None),
                    static_headers=static_headers_dict,
                    # oauth specific fields
                    client_id=getattr(mcp_server, "client_id", None),
                    client_secret=getattr(mcp_server, "client_secret", None),
                    scopes=getattr(mcp_server, "scopes", None),
                    authorization_url=getattr(mcp_server, "authorization_url", None),
                    token_url=getattr(mcp_server, "token_url", None),
                    registration_url=getattr(mcp_server, "registration_url", None),
                    # Stdio-specific fields
                    command=getattr(mcp_server, "command", None),
                    args=getattr(mcp_server, "args", None) or [],
                    env=env_dict,
                    access_groups=getattr(mcp_server, "mcp_access_groups", None),
                    allowed_tools=getattr(mcp_server, "allowed_tools", None),
                    disallowed_tools=getattr(mcp_server, "disallowed_tools", None),
                )
                self.registry[mcp_server.server_id] = new_server
                verbose_logger.debug(f"Added MCP Server: {name_for_prefix}")

        except Exception as e:
            verbose_logger.debug(f"Failed to add MCP server: {str(e)}")
            raise e

    def get_all_mcp_server_ids(self) -> Set[str]:
        """
        Get all MCP server IDs
        """
        all_servers = list(self.get_registry().values())
        return {server.server_id for server in all_servers}

    async def get_allowed_mcp_servers(
        self, user_api_key_auth: Optional[UserAPIKeyAuth] = None
    ) -> List[str]:
        """
        Get the allowed MCP Servers for the user
        """
        from litellm.proxy.management_endpoints.common_utils import _user_has_admin_view

        # If admin, get all servers
        if user_api_key_auth and _user_has_admin_view(user_api_key_auth):
            return list(self.get_registry().keys())

        try:
            allowed_mcp_servers = await MCPRequestHandler.get_allowed_mcp_servers(
                user_api_key_auth
            )
            verbose_logger.debug(
                f"Allowed MCP Servers for user api key auth: {allowed_mcp_servers}"
            )
            if len(allowed_mcp_servers) == 0:
                verbose_logger.debug(
                    "No allowed MCP Servers found for user api key auth."
                )
            return allowed_mcp_servers
        except Exception as e:
            verbose_logger.warning(f"Failed to get allowed MCP servers: {str(e)}.")
            return []

    async def get_tools_for_server(self, server_id: str) -> List[MCPTool]:
        """
        Get the tools for a given server
        """
        try:
            server = self.get_mcp_server_by_id(server_id)
            if server is None:
                verbose_logger.warning(f"MCP Server {server_id} not found")
                return []
            return await self._get_tools_from_server(server)
        except Exception as e:
            verbose_logger.warning(
                f"Failed to get tools from server {server_id}: {str(e)}"
            )
            return []

    async def list_tools(
        self,
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
        mcp_auth_header: Optional[str] = None,
        mcp_server_auth_headers: Optional[Dict[str, Union[str, Dict[str, str]]]] = None,
    ) -> List[MCPTool]:
        """
        List all tools available across all MCP Servers.

        Args:
            user_api_key_auth: User authentication
            mcp_auth_header: MCP auth header (deprecated)
            mcp_server_auth_headers: Optional dict of server-specific auth headers {server_alias: auth_value}
            mcp_protocol_version: Optional MCP protocol version from request header

        Returns:
            List[MCPTool]: Combined list of tools from all servers
        """
        allowed_mcp_servers = await self.get_allowed_mcp_servers(user_api_key_auth)

        list_tools_result: List[MCPTool] = []
        verbose_logger.debug("SERVER MANAGER LISTING TOOLS")

        for server_id in allowed_mcp_servers:
            server = self.get_mcp_server_by_id(server_id)
            if server is None:
                verbose_logger.warning(f"MCP Server {server_id} not found")
                continue

            # Get server-specific auth header if available
            server_auth_header = None
            if mcp_server_auth_headers and server.alias:
                server_auth_header = mcp_server_auth_headers.get(server.alias)
            elif mcp_server_auth_headers and server.server_name:
                server_auth_header = mcp_server_auth_headers.get(server.server_name)

            # Fall back to deprecated mcp_auth_header if no server-specific header found
            if server_auth_header is None:
                server_auth_header = mcp_auth_header

            try:
                tools = await self._get_tools_from_server(
                    server=server,
                    mcp_auth_header=server_auth_header,
                )
                list_tools_result.extend(tools)
                verbose_logger.info(
                    f"Successfully fetched {len(tools)} tools from server {server.name}"
                )
            except Exception as e:
                verbose_logger.warning(
                    f"Failed to list tools from server {server.name}: {str(e)}. Continuing with other servers."
                )
                # Continue with other servers instead of failing completely

        verbose_logger.info(
            f"Successfully fetched {len(list_tools_result)} tools total from all servers"
        )
        return list_tools_result

    #########################################################
    # Methods that call the upstream MCP servers
    #########################################################
    def _create_mcp_client(
        self,
        server: MCPServer,
        mcp_auth_header: Optional[Union[str, Dict[str, str]]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> MCPClient:
        """
        Create an MCPClient instance for the given server.

        Args:
            server (MCPServer): The server configuration
            mcp_auth_header: MCP auth header to be passed to the MCP server. This is optional and will be used if provided.

        Returns:
            MCPClient: Configured MCP client instance
        """
        transport = server.transport or MCPTransport.sse

        # Handle stdio transport
        if transport == MCPTransport.stdio:
            # For stdio, we need to get the stdio config from the server
            stdio_config: Optional[MCPStdioConfig] = None
            if server.command and server.args is not None:
                stdio_config = MCPStdioConfig(
                    command=server.command, args=server.args, env=server.env or {}
                )

            return MCPClient(
                server_url="",  # Not used for stdio
                transport_type=transport,
                auth_type=server.auth_type,
                auth_value=mcp_auth_header or server.authentication_token,
                timeout=60.0,
                stdio_config=stdio_config,
                extra_headers=extra_headers,
            )
        else:
            # For HTTP/SSE transports
            server_url = server.url or ""
            return MCPClient(
                server_url=server_url,
                transport_type=transport,
                auth_type=server.auth_type,
                auth_value=mcp_auth_header or server.authentication_token,
                timeout=60.0,
                extra_headers=extra_headers,
            )

    async def _get_tools_from_server(
        self,
        server: MCPServer,
        mcp_auth_header: Optional[Union[str, Dict[str, str]]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        add_prefix: bool = True,
    ) -> List[MCPTool]:
        """
        Helper method to get tools from a single MCP server with prefixed names.

        Args:
            server (MCPServer): The server to query tools from
            mcp_auth_header: Optional auth header for MCP server

        Returns:
            List[MCPTool]: List of tools available on the server with prefixed names
        """
        from litellm.proxy._experimental.mcp_server.tool_registry import (
            global_mcp_tool_registry,
        )

        verbose_logger.debug(f"Connecting to url: {server.url}")
        verbose_logger.info(f"_get_tools_from_server for {server.name}...")

        client = None

        try:
            if server.static_headers:
                if extra_headers is None:
                    extra_headers = {}
                extra_headers.update(server.static_headers)

            client = self._create_mcp_client(
                server=server,
                mcp_auth_header=mcp_auth_header,
                extra_headers=extra_headers,
            )

            ## HANDLE OPENAPI TOOLS
            if server.spec_path:
                _tools = global_mcp_tool_registry.list_tools(tool_prefix=server.name)
                tools = global_mcp_tool_registry.convert_tools_to_mcp_sdk_tool_type(
                    _tools
                )
            else:
                tools = await self._fetch_tools_with_timeout(client, server.name)

            prefixed_or_original_tools = self._create_prefixed_tools(
                tools, server, add_prefix=add_prefix
            )

            return prefixed_or_original_tools

        except Exception as e:
            verbose_logger.warning(
                f"Failed to get tools from server {server.name}: {str(e)}"
            )
            return []

    async def _descovery_metadata(
        self,
        server_url: str,
    ) -> Optional[MCPOAuthMetadata]:
        """Discover OAuth metadata by following RFC 9728 (protected resource metadata discovery)."""

        try:
            client = get_async_httpx_client(llm_provider=httpxSpecialProvider.MCP)
            response = await client.get(server_url)
            response.raise_for_status()
            verbose_logger.warning(
                "MCP OAuth discovery unexpectedly succeeded for %s; server did not challenge",
                server_url,
            )
            raise RuntimeError("OAuth discovery must not succeed without a challenge")
        except HTTPStatusError as exc:
            verbose_logger.debug(
                "MCP OAuth discovery for %s received status error: %s",
                server_url,
                exc,
            )

            header_value: Optional[str] = None
            if exc.response is not None:
                header_value = exc.response.headers.get(
                    "WWW-Authenticate"
                ) or exc.response.headers.get("www-authenticate")

            resource_metadata_url, scopes = self._parse_www_authenticate_header(
                header_value
            )

            authorization_servers: List[str] = []
            resource_scopes: Optional[List[str]] = None
            if resource_metadata_url:
                (
                    authorization_servers,
                    resource_scopes,
                ) = await self._fetch_oauth_metadata_from_resource(
                    resource_metadata_url
                )
            else:
                (
                    authorization_servers,
                    resource_scopes,
                ) = await self._attempt_well_known_discovery(server_url)

            metadata = None
            if not authorization_servers:
                try:
                    parsed_url = urlparse(server_url)
                    if parsed_url.scheme and parsed_url.netloc:
                        authorization_servers = [
                            f"{parsed_url.scheme}://{parsed_url.netloc}"
                        ]
                except Exception:
                    authorization_servers = []

            if authorization_servers:
                metadata = await self._fetch_authorization_server_metadata(
                    authorization_servers
                )

            preferred_scopes = scopes or resource_scopes
            if metadata is None and preferred_scopes:
                metadata = MCPOAuthMetadata(scopes=preferred_scopes)
            elif metadata is not None and preferred_scopes:
                metadata.scopes = preferred_scopes

            return metadata
        except Exception as exc:  # pragma: no cover - network/transient issues
            verbose_logger.debug(
                "MCP OAuth discovery failed for %s: %s", server_url, exc
            )
            return None

    def _parse_www_authenticate_header(
        self, header_value: Optional[str]
    ) -> Tuple[Optional[str], Optional[List[str]]]:
        if not header_value:
            return None, None

        _, _, params_section = header_value.partition(" ")
        params_section = params_section or header_value

        param_pattern = re.compile(r"([a-zA-Z0-9_]+)\s*=\s*\"?([^\",]+)\"?")
        params: Dict[str, str] = {
            match.group(1).lower(): match.group(2).strip()
            for match in param_pattern.finditer(params_section)
        }

        resource_metadata_url = params.get("resource_metadata")

        scope_value = params.get("scope")
        scopes_list = [s for s in (scope_value.split() if scope_value else []) if s]
        scopes = scopes_list or None

        return resource_metadata_url, scopes

    async def _fetch_oauth_metadata_from_resource(
        self, resource_metadata_url: str
    ) -> Tuple[List[str], Optional[List[str]]]:
        if not resource_metadata_url:
            return [], None

        try:
            client = get_async_httpx_client(
                llm_provider=httpxSpecialProvider.MCP,
                params={"timeout": 10.0},
            )
            response = await client.get(resource_metadata_url)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # pragma: no cover - network issues
            verbose_logger.debug(
                "Failed to fetch MCP OAuth metadata from %s: %s",
                resource_metadata_url,
                exc,
            )
            return [], None

        raw_servers = data.get("authorization_servers")
        if isinstance(raw_servers, list):
            authorization_servers = [
                entry
                for entry in raw_servers
                if isinstance(entry, str) and entry.strip() != ""
            ]
        else:
            authorization_servers = []

        scopes = self._extract_scopes(
            data.get("scopes_supported") or data.get("scopes")
        )

        return authorization_servers, scopes

    async def _attempt_well_known_discovery(
        self, server_url: str
    ) -> Tuple[List[str], Optional[List[str]]]:
        try:
            parsed = urlparse(server_url)
        except Exception:
            return [], None

        if not parsed.scheme or not parsed.netloc:
            return [], None

        base = f"{parsed.scheme}://{parsed.netloc}"
        path = parsed.path or ""
        path = path.strip("/")

        candidate_urls: List[str] = []
        if path:
            candidate_urls.append(f"{base}/.well-known/oauth-protected-resource/{path}")
        candidate_urls.append(f"{base}/.well-known/oauth-protected-resource")

        for url in candidate_urls:
            (
                authorization_servers,
                scopes,
            ) = await self._fetch_oauth_metadata_from_resource(url)
            if authorization_servers:
                return authorization_servers, scopes

        return [], None

    async def _fetch_authorization_server_metadata(
        self, authorization_servers: List[str]
    ) -> Optional[MCPOAuthMetadata]:
        for issuer in authorization_servers:
            metadata = await self._fetch_single_authorization_server_metadata(issuer)
            if metadata is not None:
                return metadata
        return None

    async def _fetch_single_authorization_server_metadata(
        self, issuer_url: str
    ) -> Optional[MCPOAuthMetadata]:
        try:
            parsed = urlparse(issuer_url)
        except Exception:
            return None

        if not parsed.scheme or not parsed.netloc:
            return None

        base = f"{parsed.scheme}://{parsed.netloc}"
        path = (parsed.path or "").strip("/")

        candidate_urls: List[str] = []
        if path:
            candidate_urls.append(
                f"{base}/.well-known/oauth-authorization-server/{path}"
            )
            candidate_urls.append(f"{base}/.well-known/openid-configuration/{path}")
        candidate_urls.append(f"{base}/.well-known/oauth-authorization-server")
        candidate_urls.append(f"{base}/.well-known/openid-configuration")
        candidate_urls.append(issuer_url.rstrip("/"))

        for url in candidate_urls:
            try:
                client = get_async_httpx_client(
                    llm_provider=httpxSpecialProvider.MCP,
                    params={"timeout": 10.0},
                )
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
            except Exception as exc:  # pragma: no cover - network issues
                verbose_logger.debug(
                    "Failed to fetch authorization metadata from %s: %s",
                    url,
                    exc,
                )
                continue

            scopes = self._extract_scopes(data.get("scopes_supported"))
            metadata = MCPOAuthMetadata(
                scopes=scopes,
                authorization_url=data.get("authorization_endpoint"),
                token_url=data.get("token_endpoint"),
                registration_url=data.get("registration_endpoint"),
            )

            if any(
                [
                    metadata.scopes,
                    metadata.authorization_url,
                    metadata.token_url,
                    metadata.registration_url,
                ]
            ):
                return metadata

        return None

    def _extract_scopes(self, scopes_value: Any) -> Optional[List[str]]:
        if isinstance(scopes_value, str):
            scopes = [s.strip() for s in scopes_value.split() if s.strip()]
            return scopes or None
        if isinstance(scopes_value, list):
            scopes = [s for s in scopes_value if isinstance(s, str) and s.strip()]
            return scopes or None
        return None

    async def _fetch_tools_with_timeout(
        self, client: MCPClient, server_name: str
    ) -> List[MCPTool]:
        """
        Fetch tools from MCP client with timeout and error handling.

        Args:
            client: MCP client instance
            server_name: Name of the server for logging

        Returns:
            List of tools from the server
        """

        async def _list_tools_task():
            try:
                tools = await client.list_tools()
                verbose_logger.debug(f"Tools from {server_name}: {tools}")
                return tools
            except asyncio.CancelledError:
                verbose_logger.warning(f"Client operation cancelled for {server_name}")
                return []
            except Exception as e:
                verbose_logger.warning(
                    f"Client operation failed for {server_name}: {str(e)}"
                )
                return []

        try:
            return await asyncio.wait_for(_list_tools_task(), timeout=30.0)
        except asyncio.TimeoutError:
            verbose_logger.warning(f"Timeout while listing tools from {server_name}")
            return []
        except asyncio.CancelledError:
            verbose_logger.warning(
                f"Task cancelled while listing tools from {server_name}"
            )
            return []
        except ConnectionError as e:
            verbose_logger.warning(
                f"Connection error while listing tools from {server_name}: {str(e)}"
            )
            return []
        except Exception as e:
            verbose_logger.warning(f"Error listing tools from {server_name}: {str(e)}")
            return []

    def _create_prefixed_tools(
        self, tools: List[MCPTool], server: MCPServer, add_prefix: bool = True
    ) -> List[MCPTool]:
        """
        Create prefixed tools and update tool mapping.

        Args:
            tools: List of original tools from server
            server: Server instance

        Returns:
            List of tools with prefixed names
        """
        prefixed_tools = []
        prefix = get_server_prefix(server)

        for tool in tools:
            prefixed_name = add_server_prefix_to_tool_name(tool.name, prefix)

            name_to_use = prefixed_name if add_prefix else tool.name

            tool_obj = MCPTool(
                name=name_to_use,
                description=tool.description,
                inputSchema=tool.inputSchema,
            )
            prefixed_tools.append(tool_obj)

            # Update tool to server mapping for resolution (support both forms)
            self.tool_name_to_mcp_server_name_mapping[tool.name] = prefix
            self.tool_name_to_mcp_server_name_mapping[prefixed_name] = prefix

        verbose_logger.info(
            f"Successfully fetched {len(prefixed_tools)} tools from server {server.name}"
        )
        return prefixed_tools

    def check_allowed_or_banned_tools(self, tool_name: str, server: MCPServer) -> bool:
        """
        Check if the tool is allowed or banned for the given server
        """
        if server.allowed_tools:
            return (
                tool_name in server.allowed_tools
                or f"{server.name}-{tool_name}" in server.allowed_tools
            )
        if server.disallowed_tools:
            return (
                tool_name not in server.disallowed_tools
                and f"{server.name}-{tool_name}" not in server.disallowed_tools
            )
        return True

    def validate_allowed_params(
        self, tool_name: str, arguments: Dict[str, Any], server: MCPServer
    ) -> None:
        """
        Filter arguments to only include allowed parameters for the given tool.

        Args:
            tool_name: Name of the tool (with or without prefix)
            arguments: Dictionary of arguments to filter
            server: MCPServer configuration

        Returns:
            Filtered dictionary containing only allowed parameters

        Raises:
            HTTPException: If allowed_params is configured for this tool but arguments contain disallowed params
        """
        from litellm.proxy._experimental.mcp_server.utils import (
            get_server_name_prefix_tool_mcp,
        )

        # If no allowed_params configured, return all arguments
        if not server.allowed_params:
            return

        # Get the unprefixed tool name to match against config
        unprefixed_tool_name, _ = get_server_name_prefix_tool_mcp(tool_name)

        # Check both prefixed and unprefixed tool names
        allowed_params_list = server.allowed_params.get(
            tool_name
        ) or server.allowed_params.get(unprefixed_tool_name)

        # If this tool doesn't have allowed_params specified, allow all params
        if allowed_params_list is None:
            return None

        # Filter arguments to only include allowed parameters
        disallowed_params = [
            param for param in arguments.keys() if param not in allowed_params_list
        ]

        if disallowed_params:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": f"Parameters {disallowed_params} are not allowed for tool {tool_name}. "
                    f"Allowed parameters: {allowed_params_list}. "
                    f"Contact proxy admin to allow these parameters."
                },
            )

    async def check_tool_permission_for_key_team(
        self,
        tool_name: str,
        server: MCPServer,
        user_api_key_auth: Optional[UserAPIKeyAuth],
    ) -> None:
        """
        Check if a tool is allowed based on key/team object_permission.mcp_tool_permissions.
        Uses MCPRequestHandler.is_tool_allowed_for_server for consistent inheritance logic.
        Raises HTTPException if tool is not allowed.

        Args:
            tool_name: Name of the tool to check
            server: MCPServer object
            user_api_key_auth: User authentication

        Raises:
            HTTPException: If tool is not allowed for this key/team
        """
        from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
            MCPRequestHandler,
        )

        if not user_api_key_auth:
            return

        # Check if tool is allowed
        is_allowed = await MCPRequestHandler.is_tool_allowed_for_server(
            tool_name=tool_name,
            server_id=server.server_id,
            user_api_key_auth=user_api_key_auth,
        )

        if not is_allowed:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": f"Tool '{tool_name}' is not allowed for your key/team on server '{server.name}'. Contact proxy admin for access."
                },
            )

    async def _call_openapi_tool_handler(
        self,
        server: MCPServer,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> CallToolResult:
        """
        Call an OpenAPI tool handler directly.

        For OpenAPI servers, instead of using MCP protocol, we call the tool handler
        that was registered during OpenAPI spec parsing. This handler makes direct
        HTTP requests to the API.

        Args:
            tool_name: The full tool name (with prefix) to call
            arguments: Tool arguments to pass to the handler

        Returns:
            CallToolResult with the response from the API
        """
        from mcp.types import TextContent

        from litellm.proxy._experimental.mcp_server.tool_registry import (
            global_mcp_tool_registry,
        )

        # Get the tool from the registry
        tool = global_mcp_tool_registry.get_tool(f"{server.name}-{tool_name}")
        if tool is None:
            # Tool not found in registry
            error_msg = f"OpenAPI tool {tool_name} not found in registry"
            verbose_logger.error(error_msg)
            return CallToolResult(
                content=[TextContent(type="text", text=error_msg)],
                isError=True,
            )

        try:
            # Call the tool handler with the arguments
            # The handler is an async function that makes the HTTP request
            handler_result = await tool.handler(**arguments)

            # Convert the handler result (string response) to CallToolResult format
            result = CallToolResult(
                content=[TextContent(type="text", text=str(handler_result))],
                isError=False,
            )

            return result

        except Exception as e:
            error_msg = f"Error calling OpenAPI tool {tool_name}: {str(e)}"
            verbose_logger.error(error_msg)
            return CallToolResult(
                content=[TextContent(type="text", text=error_msg)],
                isError=True,
            )

    async def pre_call_tool_check(
        self,
        name: str,
        arguments: Dict[str, Any],
        server_name: str,
        user_api_key_auth: Optional[UserAPIKeyAuth],
        proxy_logging_obj: ProxyLogging,
        server: MCPServer,
    ):
        ## check if the tool is allowed or banned for the given server
        if not self.check_allowed_or_banned_tools(name, server):
            raise HTTPException(
                status_code=403,
                detail={
                    "error": f"Tool {name} is not allowed for server {server.name}. Contact proxy admin to allow this tool."
                },
            )

        ## check tool-level permissions from object_permission
        await self.check_tool_permission_for_key_team(
            tool_name=name,
            server=server,
            user_api_key_auth=user_api_key_auth,
        )

        ## filter parameters based on allowed_params configuration
        self.validate_allowed_params(
            tool_name=name,
            arguments=arguments,
            server=server,
        )

        pre_hook_kwargs = {
            "name": name,
            "arguments": arguments,
            "server_name": server_name,
            "user_api_key_auth": user_api_key_auth,
            "user_api_key_user_id": (
                getattr(user_api_key_auth, "user_id", None)
                if user_api_key_auth
                else None
            ),
            "user_api_key_team_id": (
                getattr(user_api_key_auth, "team_id", None)
                if user_api_key_auth
                else None
            ),
            "user_api_key_end_user_id": (
                getattr(user_api_key_auth, "end_user_id", None)
                if user_api_key_auth
                else None
            ),
            "user_api_key_hash": (
                getattr(user_api_key_auth, "api_key_hash", None)
                if user_api_key_auth
                else None
            ),
        }

        # Create MCP request object for processing
        mcp_request_obj = proxy_logging_obj._create_mcp_request_object_from_kwargs(
            pre_hook_kwargs
        )

        # Convert to LLM format for existing guardrail compatibility
        synthetic_llm_data = proxy_logging_obj._convert_mcp_to_llm_format(
            mcp_request_obj, pre_hook_kwargs
        )

        try:
            # Use standard pre_call_hook with call_type="mcp_call"
            modified_data = await proxy_logging_obj.pre_call_hook(
                user_api_key_dict=user_api_key_auth,  # type: ignore
                data=synthetic_llm_data,
                call_type="mcp_call",  # type: ignore
            )
            if modified_data:
                # Convert response back to MCP format and apply modifications
                modified_kwargs = (
                    proxy_logging_obj._convert_mcp_hook_response_to_kwargs(
                        modified_data, pre_hook_kwargs
                    )
                )
                if modified_kwargs.get("arguments") != arguments:
                    arguments = modified_kwargs["arguments"]

        except (
            BlockedPiiEntityError,
            GuardrailRaisedException,
            HTTPException,
        ) as e:
            # Re-raise guardrail exceptions to properly fail the MCP call
            verbose_logger.error(f"Guardrail blocked MCP tool call pre call: {str(e)}")
            raise e

    def _create_during_hook_task(
        self,
        name: str,
        arguments: Dict[str, Any],
        server_name_from_prefix: Optional[str],
        user_api_key_auth: Optional[UserAPIKeyAuth],
        proxy_logging_obj: ProxyLogging,
        start_time: datetime.datetime,
    ):
        """Create and return a during hook task for MCP tool calls."""
        from litellm.types.llms.base import HiddenParams
        from litellm.types.mcp import MCPDuringCallRequestObject

        request_obj = MCPDuringCallRequestObject(
            tool_name=name,
            arguments=arguments,
            server_name=server_name_from_prefix,
            start_time=start_time.timestamp() if start_time else None,
            hidden_params=HiddenParams(),
        )

        during_hook_kwargs = {
            "name": name,
            "arguments": arguments,
            "server_name": server_name_from_prefix,
            "user_api_key_auth": user_api_key_auth,
        }

        synthetic_llm_data = proxy_logging_obj._convert_mcp_to_llm_format(
            request_obj, during_hook_kwargs
        )

        return asyncio.create_task(
            proxy_logging_obj.during_call_hook(
                user_api_key_dict=user_api_key_auth,
                data=synthetic_llm_data,
                call_type="mcp_call",  # type: ignore
            )
        )

    async def _call_regular_mcp_tool(
        self,
        mcp_server: MCPServer,
        original_tool_name: str,
        arguments: Dict[str, Any],
        tasks: List,
        mcp_auth_header: Optional[str],
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]],
        oauth2_headers: Optional[Dict[str, str]],
        raw_headers: Optional[Dict[str, str]],
        proxy_logging_obj: Optional[ProxyLogging],
    ) -> CallToolResult:
        """
        Call a regular MCP tool using the MCP client.

        Args:
            mcp_server: The MCP server configuration
            original_tool_name: The original tool name (without prefix)
            arguments: Tool arguments
            tasks: List of async tasks to append to (for during hooks)
            mcp_auth_header: MCP auth header (deprecated)
            mcp_server_auth_headers: Optional dict of server-specific auth headers
            oauth2_headers: Optional OAuth2 headers
            raw_headers: Optional raw headers from the request
            proxy_logging_obj: Optional ProxyLogging object for hook integration

        Returns:
            CallToolResult from the MCP server

        Raises:
            BlockedPiiEntityError: If PII is blocked by guardrails
            GuardrailRaisedException: If guardrails block the call
            HTTPException: If an HTTP error occurs
        """
        # Get server-specific auth header if available (case-insensitive)
        # FIX: Added case-insensitive matching to handle auth header keys that may not match
        # the exact case of server alias/name (e.g., '1litellmagcgateway' vs '1LiteLLMAGCGateway')
        server_auth_header: Optional[Union[Dict[str, str], str]] = None
        if mcp_server_auth_headers:
            # Normalize keys for case-insensitive lookup
            normalized_headers = {
                k.lower(): v for k, v in mcp_server_auth_headers.items()
            }

            if mcp_server.alias:
                server_auth_header = normalized_headers.get(mcp_server.alias.lower())
            if server_auth_header is None and mcp_server.server_name:
                server_auth_header = normalized_headers.get(
                    mcp_server.server_name.lower()
                )

        # Fall back to deprecated mcp_auth_header if no server-specific header found
        if server_auth_header is None:
            server_auth_header = mcp_auth_header

        # oauth2 headers
        extra_headers: Optional[Dict[str, str]] = None
        if mcp_server.auth_type == MCPAuth.oauth2:
            extra_headers = oauth2_headers

        if mcp_server.extra_headers and raw_headers:
            if extra_headers is None:
                extra_headers = {}
            for header in mcp_server.extra_headers:
                if header in raw_headers:
                    extra_headers[header] = raw_headers[header]

        if mcp_server.static_headers:
            if extra_headers is None:
                extra_headers = {}
            extra_headers.update(mcp_server.static_headers)

        client = self._create_mcp_client(
            server=mcp_server,
            mcp_auth_header=server_auth_header,
            extra_headers=extra_headers,
        )

        call_tool_params = MCPCallToolRequestParams(
            name=original_tool_name,
            arguments=arguments,
        )

        async def _call_tool_via_client(client, params):
            return await client.call_tool(params)

        tasks.append(
            asyncio.create_task(_call_tool_via_client(client, call_tool_params))
        )

        try:
            mcp_responses = await asyncio.gather(*tasks)
        except (
            BlockedPiiEntityError,
            GuardrailRaisedException,
            HTTPException,
        ) as e:
            # Re-raise guardrail exceptions to properly fail the MCP call
            verbose_logger.error(
                f"Guardrail blocked MCP tool call during result check: {str(e)}"
            )
            raise e

        # If proxy_logging_obj is None, the tool call result is at index 0
        # If proxy_logging_obj is not None, the tool call result is at index 1 (after the during hook task)
        result_index = 1 if proxy_logging_obj else 0
        result = mcp_responses[result_index]

        return cast(CallToolResult, result)

    async def call_tool(
        self,
        server_name: str,
        name: str,
        arguments: Dict[str, Any],
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
        mcp_auth_header: Optional[str] = None,
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]] = None,
        proxy_logging_obj: Optional[ProxyLogging] = None,
        oauth2_headers: Optional[Dict[str, str]] = None,
        raw_headers: Optional[Dict[str, str]] = None,
    ) -> CallToolResult:
        """
        Call a tool with the given name and arguments

        Args:
            server_name: Server name
            name: Tool name
            arguments: Tool arguments
            user_api_key_auth: User authentication
            mcp_auth_header: MCP auth header (deprecated)
            mcp_server_auth_headers: Optional dict of server-specific auth headers {server_alias: auth_value}
            proxy_logging_obj: Optional ProxyLogging object for hook integration


        Returns:
            CallToolResult from the MCP server
        """
        start_time = datetime.datetime.now()

        # Get the MCP server
        prefixed_tool_name = add_server_prefix_to_tool_name(name, server_name)
        mcp_server = self._get_mcp_server_from_tool_name(prefixed_tool_name)
        if mcp_server is None:
            raise ValueError(f"Tool {name} not found")

        #########################################################
        # Pre MCP Tool Call Hook
        # Allow validation and modification of tool calls before execution
        # Using standard pre_call_hook with call_type="mcp_call"
        #########################################################
        if proxy_logging_obj:
            await self.pre_call_tool_check(
                name=name,
                arguments=arguments,
                server_name=server_name,
                user_api_key_auth=user_api_key_auth,
                proxy_logging_obj=proxy_logging_obj,
                server=mcp_server,
            )

        # Prepare tasks for during hooks
        tasks = []
        if proxy_logging_obj:
            during_hook_task = self._create_during_hook_task(
                name=name,
                arguments=arguments,
                server_name_from_prefix=server_name,
                user_api_key_auth=user_api_key_auth,
                proxy_logging_obj=proxy_logging_obj,
                start_time=start_time,
            )
            tasks.append(during_hook_task)

        # For OpenAPI servers, call the tool handler directly instead of via MCP client
        if mcp_server.spec_path:
            verbose_logger.debug(
                f"Calling OpenAPI tool {name} directly via HTTP handler"
            )
            tasks.append(
                asyncio.create_task(
                    self._call_openapi_tool_handler(mcp_server, name, arguments)
                )
            )
        else:
            # For regular MCP servers, use the MCP client
            return await self._call_regular_mcp_tool(
                mcp_server=mcp_server,
                original_tool_name=name,
                arguments=arguments,
                tasks=tasks,
                mcp_auth_header=mcp_auth_header,
                mcp_server_auth_headers=mcp_server_auth_headers,
                oauth2_headers=oauth2_headers,
                raw_headers=raw_headers,
                proxy_logging_obj=proxy_logging_obj,
            )

        # For OpenAPI tools, await outside the client context
        try:
            mcp_responses = await asyncio.gather(*tasks)

            # If proxy_logging_obj is None, the tool call result is at index 0
            # If proxy_logging_obj is not None, the tool call result is at index 1 (after the during hook task)
            result_index = 1 if proxy_logging_obj else 0
            result = mcp_responses[result_index]

            return cast(CallToolResult, result)
        except (
            BlockedPiiEntityError,
            GuardrailRaisedException,
            HTTPException,
        ) as e:
            # Re-raise guardrail exceptions to properly fail the MCP call
            verbose_logger.error(
                f"Guardrail blocked MCP tool call during result check: {str(e)}"
            )
            raise e

    #########################################################
    # End of Methods that call the upstream MCP servers
    #########################################################

    def initialize_tool_name_to_mcp_server_name_mapping(self):
        """
        On startup, initialize the tool name to MCP server name mapping
        """
        try:
            if asyncio.get_running_loop():
                asyncio.create_task(
                    self._initialize_tool_name_to_mcp_server_name_mapping()
                )
        except RuntimeError as e:  # no running event loop
            verbose_logger.exception(
                f"No running event loop - skipping tool name to MCP server name mapping initialization: {str(e)}"
            )

    async def _initialize_tool_name_to_mcp_server_name_mapping(self):
        """
        Call list_tools for each server and update the tool name to MCP server name mapping
        Note: This now handles prefixed tool names
        """
        for server in self.get_registry().values():
            tools = await self._get_tools_from_server(server)
            for tool in tools:
                # The tool.name here is already prefixed from _get_tools_from_server
                # Extract original name for mapping
                original_name, _ = get_server_name_prefix_tool_mcp(tool.name)
                self.tool_name_to_mcp_server_name_mapping[original_name] = server.name
                self.tool_name_to_mcp_server_name_mapping[tool.name] = server.name

    def _get_mcp_server_from_tool_name(self, tool_name: str) -> Optional[MCPServer]:
        """
        Get the MCP Server from the tool name (handles both prefixed and non-prefixed names)

        Args:
            tool_name: Tool name (can be prefixed or non-prefixed)

        Returns:
            MCPServer if found, None otherwise
        """
        # First try with the original tool name
        if tool_name in self.tool_name_to_mcp_server_name_mapping:
            server_name = self.tool_name_to_mcp_server_name_mapping[tool_name]
            for server in self.get_registry().values():
                if normalize_server_name(server.name) == normalize_server_name(
                    server_name
                ):
                    return server

        # If not found and tool name is prefixed, try extracting server name from prefix
        if is_tool_name_prefixed(tool_name):
            (
                original_tool_name,
                server_name_from_prefix,
            ) = get_server_name_prefix_tool_mcp(tool_name)
            if original_tool_name in self.tool_name_to_mcp_server_name_mapping:
                for server in self.get_registry().values():
                    if normalize_server_name(server.name) == normalize_server_name(
                        server_name_from_prefix
                    ):
                        return server

        return None

    async def _add_mcp_servers_from_db_to_in_memory_registry(self):
        from litellm.proxy._experimental.mcp_server.db import get_all_mcp_servers
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            get_prisma_client_or_throw,
        )

        verbose_logger.debug("Loading MCP servers from database into registry...")

        # perform authz check to filter the mcp servers user has access to
        prisma_client = get_prisma_client_or_throw(
            "Database not connected. Connect a database to your proxy"
        )
        db_mcp_servers = await get_all_mcp_servers(prisma_client)
        verbose_logger.info(f"Found {len(db_mcp_servers)} MCP servers in database")

        # ensure the global_mcp_server_manager is up to date with the db
        for server in db_mcp_servers:
            verbose_logger.debug(
                f"Adding server to registry: {server.server_id} ({server.server_name})"
            )
            self.add_update_server(server)

        verbose_logger.debug(
            f"Registry now contains {len(self.get_registry())} servers"
        )

    def get_mcp_server_by_id(self, server_id: str) -> Optional[MCPServer]:
        """
        Get the MCP Server from the server id
        """
        registry = self.get_registry()
        for server in registry.values():
            if server.server_id == server_id:
                return server
        return None

    def get_public_mcp_servers(self) -> List[MCPServer]:
        """
        Get the public MCP servers
        """
        servers: List[MCPServer] = []
        if litellm.public_mcp_servers is None:
            return servers
        for server_id in litellm.public_mcp_servers:
            server = self.get_mcp_server_by_id(server_id)
            if server:
                servers.append(server)
        return servers

    def get_mcp_server_by_name(self, server_name: str) -> Optional[MCPServer]:
        """
        Get the MCP Server from the server name
        """
        registry = self.get_registry()
        for server in registry.values():
            if server.server_name == server_name:
                return server
        return None

    def _generate_stable_server_id(
        self,
        server_name: str,
        url: str,
        transport: str,
        auth_type: Optional[str] = None,
        alias: Optional[str] = None,
    ) -> str:
        """
        Generate a stable server ID based on server parameters using a hash function.

        This is critical to ensure the server_id is stable across server restarts.
        Some users store MCPs on the config.yaml and permission management is based on server_ids.

        Eg a key might have mcp_servers = ["1234"], if the server_id changes across restarts, the key will no longer have access to the MCP.

        Args:
            server_name: Name of the server
            url: Server URL
            transport: Transport type (sse, http, etc.)
            auth_type: Authentication type (optional)
            alias: Server alias (optional)

        Returns:
            A deterministic server ID string
        """
        # Create a string from all the identifying parameters
        params_string = (
            f"{server_name}|{url}|{transport}|{auth_type or ''}|{alias or ''}"
        )

        # Generate SHA-256 hash
        hash_object = hashlib.sha256(params_string.encode("utf-8"))
        hash_hex = hash_object.hexdigest()

        # Take first 32 characters and format as UUID-like string
        return hash_hex[:32]

    async def health_check_server(
        self, server_id: str, mcp_auth_header: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform a health check on a specific MCP server.

        Args:
            server_id: The ID of the server to health check
            mcp_auth_header: Optional authentication header for the MCP server

        Returns:
            Dict containing health check results
        """
        import time
        from datetime import datetime

        server = self.get_mcp_server_by_id(server_id)
        if not server:
            return {
                "server_id": server_id,
                "server_name": None,
                "status": "unknown",
                "error": "Server not found",
                "last_health_check": datetime.now().isoformat(),
                "response_time_ms": None,
            }

        start_time = time.time()
        try:
            # Try to get tools from the server as a health check
            tools = await self._get_tools_from_server(server, mcp_auth_header)
            response_time = (time.time() - start_time) * 1000

            return {
                "server_id": server_id,
                "server_name": server.name,
                "status": "healthy",
                "tools_count": len(tools),
                "last_health_check": datetime.now().isoformat(),
                "response_time_ms": round(response_time, 2),
                "error": None,
            }
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            error_message = str(e)

            return {
                "server_id": server_id,
                "server_name": server.name,
                "status": "unhealthy",
                "last_health_check": datetime.now().isoformat(),
                "response_time_ms": round(response_time, 2),
                "error": error_message,
            }

    async def health_check_all_servers(
        self, mcp_auth_header: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform health checks on all MCP servers.

        Args:
            mcp_auth_header: Optional authentication header for the MCP servers

        Returns:
            Dict containing health check results for all servers
        """
        all_servers = self.get_registry()
        results = {}

        for server_id, server in all_servers.items():
            results[server_id] = await self.health_check_server(
                server_id, mcp_auth_header
            )

        return results

    async def health_check_allowed_servers(
        self,
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
        mcp_auth_header: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Perform health checks on all MCP servers that the user has access to.

        Args:
            user_api_key_auth: User authentication info for access control
            mcp_auth_header: Optional authentication header for the MCP servers

        Returns:
            Dict containing health check results for accessible servers
        """
        # Get allowed servers for the user
        allowed_server_ids = await self.get_allowed_mcp_servers(user_api_key_auth)

        # Perform health checks on allowed servers
        results = {}
        for server_id in allowed_server_ids:
            results[server_id] = await self.health_check_server(
                server_id, mcp_auth_header
            )

        return results

    async def get_all_mcp_servers_with_health_and_teams(
        self,
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
        include_health: bool = True,
    ) -> List[LiteLLM_MCPServerTable]:
        """
        Get all MCP servers that the user has access to, with health status and team information.

        Args:
            user_api_key_auth: User authentication info for access control
            include_health: Whether to include health check information

        Returns:
            List of MCP server objects with health and team data
        """
        from litellm.proxy._experimental.mcp_server.db import (
            get_all_mcp_servers,
            get_mcp_servers,
        )
        from litellm.proxy.management_endpoints.common_utils import _user_has_admin_view
        from litellm.proxy.proxy_server import prisma_client

        # Get allowed server IDs
        allowed_server_ids = await self.get_allowed_mcp_servers(user_api_key_auth)

        # Get servers from database
        list_mcp_servers: List[LiteLLM_MCPServerTable] = []
        if prisma_client is not None:
            list_mcp_servers = await get_mcp_servers(prisma_client, allowed_server_ids)

            # If admin, also get all servers from database
            if user_api_key_auth and _user_has_admin_view(user_api_key_auth):
                all_mcp_servers = await get_all_mcp_servers(prisma_client)
                for server in all_mcp_servers:
                    if server.server_id not in allowed_server_ids:
                        list_mcp_servers.append(server)

        # Add config.yaml servers
        for _server_id, _server_config in self.config_mcp_servers.items():
            if _server_id in allowed_server_ids:
                list_mcp_servers.append(
                    LiteLLM_MCPServerTable(
                        **{
                            **_server_config.model_dump(),
                            "created_at": datetime.datetime.now(),
                            "updated_at": datetime.datetime.now(),
                            "description": (
                                _server_config.mcp_info.get("description")
                                if _server_config.mcp_info
                                else None
                            ),
                            "allowed_tools": _server_config.allowed_tools or [],
                            "mcp_info": _server_config.mcp_info,
                            "mcp_access_groups": _server_config.access_groups or [],
                            "extra_headers": _server_config.extra_headers or [],
                            "command": getattr(_server_config, "command", None),
                            "args": getattr(_server_config, "args", None) or [],
                            "env": getattr(_server_config, "env", None) or {},
                        }
                    )
                )

        # Get team information for non-admin users
        server_to_teams_map: Dict[str, List[Dict[str, str]]] = {}
        if (
            user_api_key_auth
            and not _user_has_admin_view(user_api_key_auth)
            and prisma_client is not None
        ):
            teams = await prisma_client.db.litellm_teamtable.find_many(
                include={"object_permission": True}
            )

            user_teams = []
            for team in teams:
                if team.members_with_roles:
                    for member in team.members_with_roles:
                        if (
                            "user_id" in member
                            and member["user_id"] is not None
                            and member["user_id"] == user_api_key_auth.user_id
                        ):
                            user_teams.append(team)

            # Create a mapping of server_id to teams that have access to it
            for team in user_teams:
                if team.object_permission and team.object_permission.mcp_servers:
                    for server_id in team.object_permission.mcp_servers:
                        if server_id not in server_to_teams_map:
                            server_to_teams_map[server_id] = []
                        server_to_teams_map[server_id].append(
                            {
                                "team_id": team.team_id,
                                "team_alias": team.team_alias,
                                "organization_id": team.organization_id,
                            }
                        )

        ## mark invalid servers w/ reason for being invalid
        valid_server_ids = self.get_all_mcp_server_ids()
        for server in list_mcp_servers:
            if server.server_id not in valid_server_ids:
                server.status = "unhealthy"
                ## try adding server to registry to get error
                try:
                    self.add_update_server(server)
                except Exception as e:
                    server.health_check_error = str(e)
                server.health_check_error = "Server is not in in memory registry yet. This could be a temporary sync issue."

        return list_mcp_servers

    async def reload_servers_from_database(self):
        """
        Public method to reload all MCP servers from database into registry.
        This can be called from management endpoints to ensure registry is up to date.
        """
        await self._add_mcp_servers_from_db_to_in_memory_registry()


global_mcp_server_manager: MCPServerManager = MCPServerManager()
