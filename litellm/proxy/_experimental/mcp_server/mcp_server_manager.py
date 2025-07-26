"""
MCP Client Manager

This class is responsible for managing MCP clients with support for both SSE and HTTP streamable transports.

This is a Proxy
"""

import asyncio
import hashlib
import json
from typing import Any, Dict, List, Optional, cast

from mcp.types import CallToolRequestParams as MCPCallToolRequestParams
from mcp.types import CallToolResult
from mcp.types import Tool as MCPTool

from litellm._logging import verbose_logger
from litellm.experimental_mcp_client.client import MCPClient
from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
    MCPRequestHandler,
)
from litellm.proxy._experimental.mcp_server.utils import (
    add_server_prefix_to_tool_name,
    get_server_name_prefix_tool_mcp,
    is_tool_name_prefixed,
    normalize_server_name,
    validate_mcp_server_name,
    get_server_prefix,
)
from litellm.proxy._types import (
    LiteLLM_MCPServerTable,
    MCPAuthType,
    MCPSpecVersion,
    MCPSpecVersionType,
    MCPTransport,
    MCPTransportType,
    UserAPIKeyAuth,
)
from litellm.types.mcp import MCPStdioConfig
from litellm.types.mcp_server.mcp_server_manager import MCPInfo, MCPServer


def _deserialize_env_dict(env_data: Any) -> Optional[Dict[str, str]]:
    """
    Helper function to deserialize environment dictionary from database storage.
    Handles both JSON string and dictionary formats.
    
    Args:
        env_data: The environment data from database (could be JSON string or dict)
        
    Returns:
        Dict[str, str] or None: Deserialized environment dictionary
    """
    if not env_data:
        return None
        
    if isinstance(env_data, str):
        try:
            return json.loads(env_data)
        except (json.JSONDecodeError, TypeError):
            # If it's not valid JSON, return as-is (shouldn't happen but safety)
            return None
    else:
        # Already a dictionary
        return env_data


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
                "auth_type": "api_key",
                "spec_version": "2025-03-26"
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

    def load_servers_from_config(self, mcp_servers_config: Dict[str, Any], mcp_aliases: Optional[Dict[str, str]] = None):
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
            # Convert Dict[str, Any] to MCPInfo properly
            mcp_info: MCPInfo = {
                "server_name": _mcp_info.get("server_name", server_name),
                "description": _mcp_info.get("description", server_config.get("description", None)),
                "logo_url": _mcp_info.get("logo_url", None),
                "mcp_server_cost_info": _mcp_info.get("mcp_server_cost_info", None),
            }

            # Use alias for name if present, else server_name
            alias = server_config.get("alias", None)
            
            # Apply mcp_aliases mapping if provided
            if mcp_aliases and alias is None:
                # Check if this server_name has an alias in mcp_aliases
                for alias_name, target_server_name in mcp_aliases.items():
                    if target_server_name == server_name and alias_name not in used_aliases:
                        alias = alias_name
                        used_aliases.add(alias_name)
                        verbose_logger.debug(f"Mapped alias '{alias_name}' to server '{server_name}'")
                        break
            
            # Create a temporary server object to use with get_server_prefix utility
            temp_server = type('TempServer', (), {
                'alias': alias,
                'server_name': server_name,
                'server_id': None
            })()
            name_for_prefix = get_server_prefix(temp_server)

            # Use alias for name if present, else server_name
            alias = server_config.get("alias", None)
            
            # Apply mcp_aliases mapping if provided
            if mcp_aliases and alias is None:
                # Check if this server_name has an alias in mcp_aliases
                for alias_name, target_server_name in mcp_aliases.items():
                    if target_server_name == server_name and alias_name not in used_aliases:
                        alias = alias_name
                        used_aliases.add(alias_name)
                        verbose_logger.debug(f"Mapped alias '{alias_name}' to server '{server_name}'")
                        break
            
            # Create a temporary server object to use with get_server_prefix utility
            temp_server = type('TempServer', (), {
                'alias': alias,
                'server_name': server_name,
                'server_id': None
            })()
            name_for_prefix = get_server_prefix(temp_server)

            # Generate stable server ID based on parameters
            server_id = self._generate_stable_server_id(
                server_name=server_name,
                url=server_config.get("url", None) or "",
                transport=server_config.get("transport", MCPTransport.http),
                spec_version=server_config.get("spec_version", MCPSpecVersion.mar_2025),
                auth_type=server_config.get("auth_type", None),
                alias=alias,
            )

            new_server = MCPServer(
                server_id=server_id,
                name=name_for_prefix,
                alias=alias,
                server_name=server_name,
                url=server_config.get("url", None) or "",
                command=server_config.get("command", None) or "",
                args=server_config.get("args", None) or [],
                env=server_config.get("env", None) or {},
                # TODO: utility fn the default values
                transport=server_config.get("transport", MCPTransport.http),
                spec_version=server_config.get("spec_version", MCPSpecVersion.mar_2025),
                auth_type=server_config.get("auth_type", None),
                mcp_info=mcp_info,
                access_groups=server_config.get("access_groups", None),
            )
            self.config_mcp_servers[server_id] = new_server
        verbose_logger.debug(
            f"Loaded MCP Servers: {json.dumps(self.config_mcp_servers, indent=4, default=str)}"
        )

        self.initialize_tool_name_to_mcp_server_name_mapping()

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
        if mcp_server.server_id not in self.get_registry():
            _mcp_info: MCPInfo = mcp_server.mcp_info or {}
            # Use helper to deserialize environment dictionary
            # Safely access env field which may not exist on Prisma model objects
            env_data = getattr(mcp_server, 'env', None)
            env_dict = _deserialize_env_dict(env_data)
            # Use alias for name if present, else server_name
            name_for_prefix = mcp_server.alias or mcp_server.server_name or mcp_server.server_id
            new_server = MCPServer(
                server_id=mcp_server.server_id,
                name=name_for_prefix,
                alias=getattr(mcp_server, 'alias', None),
                server_name=getattr(mcp_server, 'server_name', None),
                url=mcp_server.url,
                transport=cast(MCPTransportType, mcp_server.transport),
                spec_version=cast(MCPSpecVersionType, mcp_server.spec_version),
                auth_type=cast(MCPAuthType, mcp_server.auth_type),
                mcp_info=MCPInfo(
                    server_name=mcp_server.server_name or mcp_server.server_id,
                    description=mcp_server.description,
                    mcp_server_cost_info=_mcp_info.get("mcp_server_cost_info", None),
                ),
                # Stdio-specific fields
                command=getattr(mcp_server, 'command', None),
                args=getattr(mcp_server, 'args', None) or [],
                env=env_dict,
            )
            self.registry[mcp_server.server_id] = new_server
            verbose_logger.debug(
                f"Added MCP Server: {name_for_prefix}"
            )

    async def get_allowed_mcp_servers(
        self, user_api_key_auth: Optional[UserAPIKeyAuth] = None
    ) -> List[str]:
        """
        Get the allowed MCP Servers for the user
        """
        allowed_mcp_servers = await MCPRequestHandler.get_allowed_mcp_servers(
            user_api_key_auth
        )
        verbose_logger.debug(
            f"Allowed MCP Servers for user api key auth: {allowed_mcp_servers}"
        )
        if len(allowed_mcp_servers) > 0:
            return allowed_mcp_servers
        else:
            verbose_logger.debug(
                "No allowed MCP Servers found for user api key auth, returning default registry servers"
            )
            return list(self.get_registry().keys())


    async def get_tools_for_server(self, server_id: str) -> List[MCPTool]:
        """
        Get the tools for a given server
        """
        server = self.get_mcp_server_by_id(server_id)
        if server is None:
            return []
        return await self._get_tools_from_server(server)
        

    async def list_tools(
        self, 
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
        mcp_auth_header: Optional[str] = None,
        mcp_server_auth_headers: Optional[Dict[str, str]] = None,
    ) -> List[MCPTool]:
        """
        List all tools available across all MCP Servers.

        Args:
            user_api_key_auth: User authentication
            mcp_auth_header: MCP auth header (deprecated)
            mcp_server_auth_headers: Optional dict of server-specific auth headers {server_alias: auth_value}

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
                verbose_logger.info(f"Successfully fetched {len(tools)} tools from server {server.name}")
            except Exception as e:
                verbose_logger.warning(
                    f"Failed to list tools from server {server.name}: {str(e)}. Continuing with other servers."
                )
                # Continue with other servers instead of failing completely

        verbose_logger.info(f"Successfully fetched {len(list_tools_result)} tools total from all servers")
        return list_tools_result

    #########################################################
    # Methods that call the upstream MCP servers
    #########################################################
    def _create_mcp_client(self, server: MCPServer, mcp_auth_header: Optional[str] = None) -> MCPClient:
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
                    command=server.command,
                    args=server.args,
                    env=server.env or {}
                )
            
            return MCPClient(
                server_url="",  # Not used for stdio
                transport_type=transport,
                auth_type=server.auth_type,
                auth_value=mcp_auth_header or server.authentication_token,
                timeout=60.0,
                stdio_config=stdio_config,
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
            )

    async def _get_tools_from_server(self, server: MCPServer, mcp_auth_header: Optional[str] = None) -> List[MCPTool]:
        """
        Helper method to get tools from a single MCP server with prefixed names.

        Args:
            server (MCPServer): The server to query tools from
            mcp_auth_header: Optional auth header for MCP server

        Returns:
            List[MCPTool]: List of tools available on the server with prefixed names
        """
        verbose_logger.debug(f"Connecting to url: {server.url}")
        verbose_logger.info(f"_get_tools_from_server for {server.name}...")

        client = None
        try:
            client = self._create_mcp_client(
                server=server,
                mcp_auth_header=mcp_auth_header,
            )

            # Create a task for the client operations to ensure proper cancellation handling
            async def _list_tools_task():
                try:
                    async with client:
                        tools = await client.list_tools()
                        verbose_logger.debug(f"Tools from {server.name}: {tools}")
                        return tools
                except asyncio.CancelledError:
                    verbose_logger.warning(f"Client operation cancelled for {server.name}")
                    return []
                except Exception as e:
                    verbose_logger.warning(f"Client operation failed for {server.name}: {str(e)}")
                    return []

            try:
                # Add timeout to prevent hanging
                tools = await asyncio.wait_for(_list_tools_task(), timeout=30.0)

                # Create new tools with prefixed names
                prefixed_tools = []
                for tool in tools:
                    # Always use alias for prefixing if present
                    prefix = get_server_prefix(server)
                    prefixed_name = add_server_prefix_to_tool_name(tool.name, prefix)

                    # Create new tool with prefixed name
                    prefixed_tool = MCPTool(
                        name=prefixed_name,
                        description=tool.description,
                        inputSchema=tool.inputSchema
                    )
                    prefixed_tools.append(prefixed_tool)

                    # Update tool to server mapping with both original and prefixed names
                    self.tool_name_to_mcp_server_name_mapping[tool.name] = prefix
                    self.tool_name_to_mcp_server_name_mapping[prefixed_name] = prefix

                verbose_logger.info(f"Successfully fetched {len(prefixed_tools)} tools from server {server.name}")
                return prefixed_tools
            except asyncio.TimeoutError:
                verbose_logger.warning(f"Timeout while listing tools from {server.name}")
                # Don't re-raise the exception, just return empty list
                return []
            except asyncio.CancelledError:
                verbose_logger.warning(f"Task cancelled while listing tools from {server.name}")
                # Don't re-raise cancellation, just return empty list
                return []
            except ConnectionError as e:
                verbose_logger.warning(f"Connection error while listing tools from {server.name}: {str(e)}")
                # Don't re-raise the exception, just return empty list
                return []
            except Exception as e:
                verbose_logger.warning(f"Error listing tools from {server.name}: {str(e)}")
                # Don't re-raise the exception, just return empty list
                return []
        except Exception as e:
            verbose_logger.warning(f"Failed to get tools from server {server.name}: {str(e)}")
            return []  # Return empty list on failure
        finally:
            if client:
                try:
                    await client.disconnect()
                except Exception:
                    pass

    async def call_tool(
            self,
            name: str,
            arguments: Dict[str, Any],
            user_api_key_auth: Optional[UserAPIKeyAuth] = None,
            mcp_auth_header: Optional[str] = None,
            mcp_server_auth_headers: Optional[Dict[str, str]] = None,
    ) -> CallToolResult:
        """
        Call a tool with the given name and arguments (handles prefixed tool names)

        Args:
            name: Tool name (can be prefixed with server name)
            arguments: Tool arguments
            user_api_key_auth: User authentication
            mcp_auth_header: MCP auth header (deprecated)
            mcp_server_auth_headers: Optional dict of server-specific auth headers {server_alias: auth_value}

        Returns:
            CallToolResult from the MCP server
        """
        # Remove prefix if present to get the original tool name
        original_tool_name, server_name_from_prefix = get_server_name_prefix_tool_mcp(name)

        # Get the MCP server
        mcp_server = self._get_mcp_server_from_tool_name(name)
        if mcp_server is None:
            raise ValueError(f"Tool {name} not found")

        # Validate that the server from prefix matches the actual server (if prefix was used)
        if server_name_from_prefix:
            expected_prefix = get_server_prefix(mcp_server)
            if normalize_server_name(server_name_from_prefix) != normalize_server_name(expected_prefix):
                raise ValueError(
                    f"Tool {name} server prefix mismatch: expected {expected_prefix}, got {server_name_from_prefix}")

        # Get server-specific auth header if available
        server_auth_header = None
        if mcp_server_auth_headers and mcp_server.alias:
            server_auth_header = mcp_server_auth_headers.get(mcp_server.alias)
        elif mcp_server_auth_headers and mcp_server.server_name:
            server_auth_header = mcp_server_auth_headers.get(mcp_server.server_name)
        
        # Fall back to deprecated mcp_auth_header if no server-specific header found
        if server_auth_header is None:
            server_auth_header = mcp_auth_header

        client = self._create_mcp_client(
            server=mcp_server,
            mcp_auth_header=server_auth_header,
        )
        async with client:
            # Use the original tool name (without prefix) for the actual call
            call_tool_params = MCPCallToolRequestParams(
                name=original_tool_name,
                arguments=arguments,
            )
            return await client.call_tool(call_tool_params)

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
                if normalize_server_name(server.name) == normalize_server_name(server_name):
                    return server

        # If not found and tool name is prefixed, try extracting server name from prefix
        if is_tool_name_prefixed(tool_name):
            _, server_name_from_prefix = get_server_name_prefix_tool_mcp(tool_name)
            for server in self.get_registry().values():
                if normalize_server_name(server.name) == normalize_server_name(server_name_from_prefix):
                    return server

        return None

    async def _add_mcp_servers_from_db_to_in_memory_registry(self):
        from litellm.proxy._experimental.mcp_server.db import get_all_mcp_servers
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            get_prisma_client_or_throw,
        )

        # perform authz check to filter the mcp servers user has access to
        prisma_client = get_prisma_client_or_throw(
            "Database not connected. Connect a database to your proxy"
        )
        db_mcp_servers = await get_all_mcp_servers(prisma_client)
        # ensure the global_mcp_server_manager is up to date with the db
        for server in db_mcp_servers:
            self.add_update_server(server)

    def get_mcp_server_by_id(self, server_id: str) -> Optional[MCPServer]:
        """
        Get the MCP Server from the server id
        """
        for server in self.get_registry().values():
            if server.server_id == server_id:
                return server
        return None

    def _generate_stable_server_id(
        self,
        server_name: str,
        url: str,
        transport: str,
        spec_version: str,
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
            spec_version: MCP spec version
            auth_type: Authentication type (optional)
            alias: Server alias (optional)

        Returns:
            A deterministic server ID string
        """
        # Create a string from all the identifying parameters
        params_string = (
            f"{server_name}|{url}|{transport}|{spec_version}|{auth_type or ''}|{alias or ''}"
        )

        # Generate SHA-256 hash
        hash_object = hashlib.sha256(params_string.encode("utf-8"))
        hash_hex = hash_object.hexdigest()

        # Take first 32 characters and format as UUID-like string
        return hash_hex[:32]


global_mcp_server_manager: MCPServerManager = MCPServerManager()
