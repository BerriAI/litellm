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
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple, Union, cast
from urllib.parse import urlparse

import anyio
from fastapi import HTTPException
from httpx import HTTPStatusError
from mcp import ReadResourceResult, Resource
from mcp.types import CallToolRequestParams as MCPCallToolRequestParams
from mcp.types import (
    CallToolResult,
    GetPromptRequestParams,
    GetPromptResult,
    Prompt,
    ResourceTemplate,
)
from mcp.types import Tool as MCPTool
from pydantic import AnyUrl

import litellm
from litellm._logging import verbose_logger
from litellm.exceptions import BlockedPiiEntityError, GuardrailRaisedException
from litellm.experimental_mcp_client.client import MCPClient
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
    MCPRequestHandler,
)
from litellm.proxy._experimental.mcp_server.oauth2_token_cache import resolve_mcp_auth
from litellm.proxy._experimental.mcp_server.utils import (
    MCP_TOOL_PREFIX_SEPARATOR,
    add_server_prefix_to_name,
    get_server_prefix,
    is_tool_name_prefixed,
    merge_mcp_headers,
    normalize_server_name,
    split_server_prefix_from_name,
    validate_mcp_server_name,
)
from litellm.proxy._types import (
    LiteLLM_MCPServerTable,
    MCPAuthType,
    MCPTransport,
    MCPTransportType,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.ip_address_utils import IPAddressUtils
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper
from litellm.proxy.utils import ProxyLogging
from litellm.types.llms.custom_http import httpxSpecialProvider
from litellm.types.mcp import MCPAuth, MCPStdioConfig
from litellm.types.mcp_server.mcp_server_manager import (
    MCPInfo,
    MCPOAuthMetadata,
    MCPServer,
)
from litellm.types.utils import CallTypes

try:
    from mcp.shared.tool_name_validation import (
        validate_tool_name,  # pyright: ignore[reportAssignmentType]
    )
    from mcp.shared.tool_name_validation import SEP_986_URL
except ImportError:
    from pydantic import BaseModel

    SEP_986_URL = "https://github.com/modelcontextprotocol/protocol/blob/main/proposals/0001-tool-name-validation.md"

    class _ToolNameValidationResult(BaseModel):
        is_valid: bool = True
        warnings: list = []

    def validate_tool_name(name: str) -> _ToolNameValidationResult:  # type: ignore[misc]
        return _ToolNameValidationResult()


# Probe includes characters on both sides of the separator to mimic real prefixed tool names.
_separator_probe_tool_name = f"litellm{MCP_TOOL_PREFIX_SEPARATOR}probe"
_separator_probe = validate_tool_name(_separator_probe_tool_name)
if not _separator_probe.is_valid:
    verbose_logger.warning(
        "MCP tool prefix separator '%s' violates SEP-986. See %s",
        MCP_TOOL_PREFIX_SEPARATOR,
        SEP_986_URL,
    )


def _warn_on_server_name_fields(
    *,
    server_id: str,
    alias: Optional[str],
    server_name: Optional[str],
):
    def _warn(field_name: str, value: Optional[str]) -> None:
        if not value:
            return
        result = validate_tool_name(value)
        if result.is_valid:
            return

        warning_text = (
            "; ".join(result.warnings) if result.warnings else "Validation failed"
        )
        verbose_logger.warning(
            "MCP server '%s' has invalid %s '%s': %s",
            server_id,
            field_name,
            value,
            warning_text,
        )

    _warn("alias", alias)
    _warn("server_name", server_name)


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
    _STDIO_ENV_TEMPLATE_PATTERN = re.compile(r"^\$\{(X-[^}]+)\}$")

    def __init__(self):
        self.registry: Dict[str, MCPServer] = {}
        self.config_mcp_servers: Dict[str, MCPServer] = {}
        """
        eg.
        [
            "server-1": {
                "name": "zapier_mcp_server",
                "url": "https://actions.zapier.com/mcp/<your-api-key>/sse"
                "transport": "sse",
                "auth_type": "api_key"
            },
            "uuid-2": {
                "name": "google_drive_mcp_server",
                "url": "https://actions.zapier.com/mcp/<your-api-key>/sse"
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

            _warn_on_server_name_fields(
                server_id=server_id,
                alias=alias,
                server_name=server_name,
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
                allow_all_keys=bool(server_config.get("allow_all_keys", False)),
                available_on_public_internet=bool(
                    server_config.get("available_on_public_internet", False)
                ),
            )
            self.config_mcp_servers[server_id] = new_server

            # Check if this is an OpenAPI-based server
            spec_path = server_config.get("spec_path", None)
            if spec_path:
                verbose_logger.info(
                    f"Loading OpenAPI spec from {spec_path} for server {server_name}"
                )
                await self._register_openapi_tools(
                    spec_path=spec_path,
                    server=new_server,
                    base_url=server_config.get("url", ""),
                )

        verbose_logger.debug(
            f"Loaded MCP Servers: {json.dumps(self.config_mcp_servers, indent=4, default=str)}"
        )

        self.initialize_tool_name_to_mcp_server_name_mapping()

    async def _register_openapi_tools(
        self, spec_path: str, server: MCPServer, base_url: str
    ):
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
            load_openapi_spec_async,
        )
        from litellm.proxy._experimental.mcp_server.tool_registry import (
            global_mcp_tool_registry,
        )

        try:
            # Load OpenAPI spec (async to avoid "called from within a running event loop")
            spec = await load_openapi_spec_async(spec_path)

            # Use base_url from config if provided, otherwise extract from spec
            if not base_url:
                base_url = get_openapi_base_url(spec)

            verbose_logger.info(
                f"Registering OpenAPI tools for server {server.name} with base URL: {base_url}"
            )

            # Get server prefix for tool naming
            server_prefix = get_server_prefix(server)

            # Build headers from server configuration
            headers: Dict[str, str] = {}

            # Add authentication headers if configured
            if server.authentication_token:
                from litellm.types.mcp import MCPAuth

                if server.auth_type == MCPAuth.bearer_token:
                    headers["Authorization"] = f"Bearer {server.authentication_token}"
                elif server.auth_type == MCPAuth.api_key:
                    headers["Authorization"] = f"ApiKey {server.authentication_token}"
                elif server.auth_type == MCPAuth.basic:
                    headers["Authorization"] = f"Basic {server.authentication_token}"

            # Add any static headers from server config.
            #
            # Note: `extra_headers` on MCPServer is a List[str] of header names to forward
            # from the client request (not available in this OpenAPI tool generation step).
            # `static_headers` is a dict of concrete headers to always send.
            headers = (
                merge_mcp_headers(
                    extra_headers=headers,
                    static_headers=server.static_headers,
                )
                or {}
            )

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
                    prefixed_tool_name = add_server_prefix_to_name(
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

    async def build_mcp_server_from_table(
        self,
        mcp_server: LiteLLM_MCPServerTable,
        *,
        credentials_are_encrypted: bool = True,
    ) -> MCPServer:
        _mcp_info: MCPInfo = mcp_server.mcp_info or {}
        env_dict = _deserialize_json_dict(getattr(mcp_server, "env", None))
        static_headers_dict = _deserialize_json_dict(
            getattr(mcp_server, "static_headers", None)
        )
        credentials_dict = _deserialize_json_dict(
            getattr(mcp_server, "credentials", None)
        )

        encrypted_auth_value: Optional[str] = None
        encrypted_client_id: Optional[str] = None
        encrypted_client_secret: Optional[str] = None
        if credentials_dict:
            encrypted_auth_value = credentials_dict.get("auth_value")
            encrypted_client_id = credentials_dict.get("client_id")
            encrypted_client_secret = credentials_dict.get("client_secret")

        auth_value: Optional[str] = None
        if encrypted_auth_value:
            if credentials_are_encrypted:
                auth_value = decrypt_value_helper(
                    value=encrypted_auth_value,
                    key="auth_value",
                    exception_type="debug",
                    return_original_value=True,
                )
            else:
                auth_value = encrypted_auth_value

        client_id_value: Optional[str] = None
        if encrypted_client_id:
            if credentials_are_encrypted:
                client_id_value = decrypt_value_helper(
                    value=encrypted_client_id,
                    key="client_id",
                    exception_type="debug",
                    return_original_value=True,
                )
            else:
                client_id_value = encrypted_client_id

        client_secret_value: Optional[str] = None
        if encrypted_client_secret:
            if credentials_are_encrypted:
                client_secret_value = decrypt_value_helper(
                    value=encrypted_client_secret,
                    key="client_secret",
                    exception_type="debug",
                    return_original_value=True,
                )
            else:
                client_secret_value = encrypted_client_secret

        scopes: Optional[List[str]] = None
        if credentials_dict:
            scopes_value = credentials_dict.get("scopes")
            if scopes_value is not None:
                scopes = self._extract_scopes(scopes_value)

        name_for_prefix = (
            mcp_server.alias or mcp_server.server_name or mcp_server.server_id
        )

        mcp_info: MCPInfo = _mcp_info.copy()
        if "server_name" not in mcp_info:
            mcp_info["server_name"] = mcp_server.server_name or mcp_server.server_id
        if "description" not in mcp_info and mcp_server.description:
            mcp_info["description"] = mcp_server.description

        auth_type = cast(MCPAuthType, mcp_server.auth_type)
        if mcp_server.url and auth_type == MCPAuth.oauth2:
            mcp_oauth_metadata = await self._descovery_metadata(
                server_url=mcp_server.url,
            )
        else:
            mcp_oauth_metadata = None

        resolved_scopes = scopes or (
            mcp_oauth_metadata.scopes if mcp_oauth_metadata else None
        )

        new_server = MCPServer(
            server_id=mcp_server.server_id,
            name=name_for_prefix,
            alias=getattr(mcp_server, "alias", None),
            server_name=getattr(mcp_server, "server_name", None),
            url=mcp_server.url,
            transport=cast(MCPTransportType, mcp_server.transport),
            auth_type=auth_type,
            authentication_token=auth_value,
            mcp_info=mcp_info,
            extra_headers=getattr(mcp_server, "extra_headers", None),
            static_headers=static_headers_dict,
            client_id=client_id_value or getattr(mcp_server, "client_id", None),
            client_secret=client_secret_value
            or getattr(mcp_server, "client_secret", None),
            scopes=resolved_scopes,
            authorization_url=mcp_server.authorization_url
            or getattr(mcp_oauth_metadata, "authorization_url", None),
            token_url=mcp_server.token_url
            or getattr(mcp_oauth_metadata, "token_url", None),
            registration_url=mcp_server.registration_url
            or getattr(mcp_oauth_metadata, "registration_url", None),
            command=getattr(mcp_server, "command", None),
            args=getattr(mcp_server, "args", None) or [],
            env=env_dict,
            access_groups=getattr(mcp_server, "mcp_access_groups", None),
            allowed_tools=getattr(mcp_server, "allowed_tools", None),
            disallowed_tools=getattr(mcp_server, "disallowed_tools", None),
            allow_all_keys=mcp_server.allow_all_keys,
            available_on_public_internet=bool(
                getattr(mcp_server, "available_on_public_internet", False)
            ),
            updated_at=getattr(mcp_server, "updated_at", None),
        )
        return new_server

    async def add_server(self, mcp_server: LiteLLM_MCPServerTable):
        try:
            if mcp_server.server_id not in self.registry:
                new_server = await self.build_mcp_server_from_table(mcp_server)
                self.registry[mcp_server.server_id] = new_server
                verbose_logger.debug(f"Added MCP Server: {new_server.name}")

        except Exception as e:
            verbose_logger.debug(f"Failed to add MCP server: {str(e)}")
            raise e

    async def update_server(self, mcp_server: LiteLLM_MCPServerTable):
        try:
            if mcp_server.server_id in self.registry:
                new_server = await self.build_mcp_server_from_table(mcp_server)
                self.registry[mcp_server.server_id] = new_server
                verbose_logger.debug(f"Updated MCP Server: {new_server.name}")

        except Exception as e:
            verbose_logger.debug(f"Failed to udpate MCP server: {str(e)}")
            raise e

    def get_all_mcp_server_ids(self) -> Set[str]:
        """
        Get all MCP server IDs
        """
        all_servers = list(self.get_registry().values())
        return {server.server_id for server in all_servers}

    def get_allow_all_keys_server_ids(self) -> List[str]:
        """Return server IDs that bypass per-key restrictions."""
        return [
            server.server_id
            for server in self.get_registry().values()
            if server.allow_all_keys is True
        ]

    async def get_allowed_mcp_servers(
        self, user_api_key_auth: Optional[UserAPIKeyAuth] = None
    ) -> List[str]:
        """
        Get the allowed MCP Servers for the user.

        Priority:
        1. If object_permission.mcp_servers is explicitly set, use it (even for admins)
        2. If admin and no object_permission, return all servers
        3. Otherwise, use standard permission checks
        """
        from litellm.proxy.management_endpoints.common_utils import _user_has_admin_view

        allow_all_server_ids = self.get_allow_all_keys_server_ids()

        try:
            # Check if object_permission.mcp_servers is explicitly set
            has_explicit_object_permission = False
            if user_api_key_auth and user_api_key_auth.object_permission:
                # Check if mcp_servers is explicitly set (not None, empty list is valid)
                if user_api_key_auth.object_permission.mcp_servers is not None:
                    has_explicit_object_permission = True
                    verbose_logger.debug(
                        f"Object permission mcp_servers explicitly set: {user_api_key_auth.object_permission.mcp_servers}"
                    )

            # If admin but NO explicit object permission, get all servers
            if (
                user_api_key_auth
                and _user_has_admin_view(user_api_key_auth)
                and not has_explicit_object_permission
            ):
                verbose_logger.debug(
                    "Admin user without explicit object_permission - returning all servers"
                )
                return list(self.get_registry().keys())

            # Get allowed servers from object permissions (respects object_permission even for admins)
            allowed_mcp_servers = await MCPRequestHandler.get_allowed_mcp_servers(
                user_api_key_auth
            )
            verbose_logger.debug(
                f"Allowed MCP Servers for user api key auth: {allowed_mcp_servers}"
            )
            combined_servers = set(allowed_mcp_servers)
            combined_servers.update(allow_all_server_ids)

            if len(combined_servers) == 0:
                verbose_logger.debug(
                    "No allowed MCP Servers found for user api key auth."
                )
            return list(combined_servers)
        except Exception as e:
            verbose_logger.warning(f"Failed to get allowed MCP servers: {str(e)}.")
            return allow_all_server_ids

    def filter_server_ids_by_ip(
        self, server_ids: List[str], client_ip: Optional[str]
    ) -> List[str]:
        """
        Filter server IDs by client IP — external callers only see public servers.

        Returns server_ids unchanged when client_ip is None (no filtering).
        """
        if client_ip is None:
            return server_ids
        return [
            sid
            for sid in server_ids
            if (s := self.get_mcp_server_by_id(sid)) is not None
            and self._is_server_accessible_from_ip(s, client_ip)
        ]

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

        verbose_logger.debug("SERVER MANAGER LISTING TOOLS")

        async def _fetch_server_tools(server_id: str) -> List[MCPTool]:
            """Fetch tools from a single server with error handling."""
            server = self.get_mcp_server_by_id(server_id)
            if server is None:
                verbose_logger.warning(f"MCP Server {server_id} not found")
                return []

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
                return tools
            except Exception as e:
                verbose_logger.warning(
                    f"Failed to list tools from server {server.name}: {str(e)}. Continuing with other servers."
                )
                return []

        # Fetch tools from all servers in parallel
        tasks = [_fetch_server_tools(server_id) for server_id in allowed_mcp_servers]
        results = await asyncio.gather(*tasks)

        # Flatten results into single list
        list_tools_result: List[MCPTool] = [tool for tools in results for tool in tools]

        verbose_logger.info(
            f"Successfully fetched {len(list_tools_result)} tools total from all servers"
        )
        return list_tools_result

    #########################################################
    # Methods that call the upstream MCP servers
    #########################################################
    def _build_stdio_env(
        self,
        server: MCPServer,
        raw_headers: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, str]]:
        """Resolve stdio env values, supporting header-driven placeholders."""

        if server.transport != MCPTransport.stdio or not server.env:
            return None

        resolved_env: Dict[str, str] = {}
        normalized_headers = {k.lower(): v for k, v in (raw_headers or {}).items()}

        for env_key, env_value in server.env.items():
            stripped_value = env_value.strip()
            match = self._STDIO_ENV_TEMPLATE_PATTERN.match(stripped_value)
            if match:
                header_name = match.group(1)
                header_value = normalized_headers.get(header_name.lower())
                if header_value is None:
                    continue
                resolved_env[env_key] = header_value
            else:
                resolved_env[env_key] = env_value

        return resolved_env

    async def _create_mcp_client(
        self,
        server: MCPServer,
        mcp_auth_header: Optional[Union[str, Dict[str, str]]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        stdio_env: Optional[Dict[str, str]] = None,
    ) -> MCPClient:
        """
        Create an MCPClient instance for the given server.

        Auth resolution (single place for all auth logic):
        1. ``mcp_auth_header`` — per-request/per-user override
        2. OAuth2 client_credentials token — auto-fetched and cached
        3. ``server.authentication_token`` — static token from config/DB

        Args:
            server: The server configuration.
            mcp_auth_header: Optional per-request auth override.
            extra_headers: Additional headers to forward.
            stdio_env: Environment variables for stdio transport.

        Returns:
            Configured MCP client instance.
        """
        auth_value = await resolve_mcp_auth(server, mcp_auth_header)

        transport = server.transport or MCPTransport.sse

        # Handle stdio transport
        if transport == MCPTransport.stdio:
            resolved_env = stdio_env if stdio_env is not None else dict(server.env or {})

            # Ensure npm-based STDIO MCP servers have a writable cache dir.
            # In containers the default (~/.npm or /app/.npm) may not exist
            # or be read-only, causing npx to fail with ENOENT.
            if "NPM_CONFIG_CACHE" not in resolved_env:
                from litellm.constants import MCP_NPM_CACHE_DIR

                resolved_env["NPM_CONFIG_CACHE"] = MCP_NPM_CACHE_DIR
            stdio_config: Optional[MCPStdioConfig] = None
            if server.command and server.args is not None:
                stdio_config = MCPStdioConfig(
                    command=server.command,
                    args=server.args,
                    env=resolved_env,
                )

            return MCPClient(
                server_url="",  # Not used for stdio
                transport_type=transport,
                auth_type=server.auth_type,
                auth_value=auth_value,
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
                auth_value=auth_value,
                timeout=60.0,
                extra_headers=extra_headers,
            )

    async def _get_tools_from_server(
        self,
        server: MCPServer,
        mcp_auth_header: Optional[Union[str, Dict[str, str]]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        add_prefix: bool = True,
        raw_headers: Optional[Dict[str, str]] = None,
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

            stdio_env = self._build_stdio_env(server, raw_headers)

            client = await self._create_mcp_client(
                server=server,
                mcp_auth_header=mcp_auth_header,
                extra_headers=extra_headers,
                stdio_env=stdio_env,
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

    async def get_prompts_from_server(
        self,
        server: MCPServer,
        mcp_auth_header: Optional[Union[str, Dict[str, str]]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        add_prefix: bool = True,
        raw_headers: Optional[Dict[str, str]] = None,
    ) -> List[Prompt]:
        """
        Helper method to get prompts from a single MCP server with prefixed names.

        Args:
            server (MCPServer): The server to query prompts from
            mcp_auth_header: Optional auth header for MCP server

        Returns:
            List[Prompt]: List of prompts available on the server with prefixed names
        """

        verbose_logger.debug(f"Connecting to url: {server.url}")
        verbose_logger.info(f"get_prompts_from_server for {server.name}...")

        client = None

        try:
            if server.static_headers:
                if extra_headers is None:
                    extra_headers = {}
                extra_headers.update(server.static_headers)

            stdio_env = self._build_stdio_env(server, raw_headers)

            client = await self._create_mcp_client(
                server=server,
                mcp_auth_header=mcp_auth_header,
                extra_headers=extra_headers,
                stdio_env=stdio_env,
            )

            prompts = await client.list_prompts()

            prefixed_or_original_prompts = self._create_prefixed_prompts(
                prompts, server, add_prefix=add_prefix
            )

            return prefixed_or_original_prompts

        except Exception as e:
            verbose_logger.warning(
                f"Failed to get prompts from server {server.name}: {str(e)}"
            )
            return []

    async def get_resources_from_server(
        self,
        server: MCPServer,
        mcp_auth_header: Optional[Union[str, Dict[str, str]]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        add_prefix: bool = True,
        raw_headers: Optional[Dict[str, str]] = None,
    ) -> List[Resource]:
        """Fetch available resources from a single MCP server."""

        verbose_logger.debug(f"Connecting to url: {server.url}")
        verbose_logger.info(f"get_resources_from_server for {server.name}...")

        client = None

        try:
            if server.static_headers:
                if extra_headers is None:
                    extra_headers = {}
                extra_headers.update(server.static_headers)

            stdio_env = self._build_stdio_env(server, raw_headers)

            client = await self._create_mcp_client(
                server=server,
                mcp_auth_header=mcp_auth_header,
                extra_headers=extra_headers,
                stdio_env=stdio_env,
            )

            resources = await client.list_resources()

            prefixed_resources = self._create_prefixed_resources(
                resources, server, add_prefix=add_prefix
            )

            return prefixed_resources

        except Exception as e:
            verbose_logger.warning(
                f"Failed to get resources from server {server.name}: {str(e)}"
            )
            return []

    async def get_resource_templates_from_server(
        self,
        server: MCPServer,
        mcp_auth_header: Optional[Union[str, Dict[str, str]]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        add_prefix: bool = True,
        raw_headers: Optional[Dict[str, str]] = None,
    ) -> List[ResourceTemplate]:
        """Fetch available resource templates from a single MCP server."""

        verbose_logger.debug(f"Connecting to url: {server.url}")
        verbose_logger.info(f"get_resource_templates_from_server for {server.name}...")

        client = None

        try:
            if server.static_headers:
                if extra_headers is None:
                    extra_headers = {}
                extra_headers.update(server.static_headers)

            stdio_env = self._build_stdio_env(server, raw_headers)

            client = await self._create_mcp_client(
                server=server,
                mcp_auth_header=mcp_auth_header,
                extra_headers=extra_headers,
                stdio_env=stdio_env,
            )

            resource_templates = await client.list_resource_templates()

            prefixed_templates = self._create_prefixed_resource_templates(
                resource_templates, server, add_prefix=add_prefix
            )

            return prefixed_templates

        except Exception as e:
            verbose_logger.warning(
                f"Failed to get resource templates from server {server.name}: {str(e)}"
            )
            return []

    async def read_resource_from_server(
        self,
        server: MCPServer,
        url: AnyUrl,
        mcp_auth_header: Optional[Union[str, Dict[str, str]]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        raw_headers: Optional[Dict[str, str]] = None,
    ) -> ReadResourceResult:
        """Read resource contents from a specific MCP server."""

        verbose_logger.debug(f"Connecting to url: {server.url}")
        verbose_logger.info(f"read_resource_from_server for {server.name}...")

        if server.static_headers:
            if extra_headers is None:
                extra_headers = {}
            extra_headers.update(server.static_headers)

        stdio_env = self._build_stdio_env(server, raw_headers)

        client = await self._create_mcp_client(
            server=server,
            mcp_auth_header=mcp_auth_header,
            extra_headers=extra_headers,
            stdio_env=stdio_env,
        )

        return await client.read_resource(url)

    async def get_prompt_from_server(
        self,
        server: MCPServer,
        prompt_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        mcp_auth_header: Optional[Union[str, Dict[str, str]]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        raw_headers: Optional[Dict[str, str]] = None,
    ) -> GetPromptResult:
        """Fetch a specific prompt definition from a single MCP server."""

        verbose_logger.debug(f"Connecting to url: {server.url}")
        verbose_logger.info(f"get_prompt_from_server for {server.name}...")

        if server.static_headers:
            if extra_headers is None:
                extra_headers = {}
            extra_headers.update(server.static_headers)

        stdio_env = self._build_stdio_env(server, raw_headers)

        client = await self._create_mcp_client(
            server=server,
            mcp_auth_header=mcp_auth_header,
            extra_headers=extra_headers,
            stdio_env=stdio_env,
        )

        get_prompt_request_params = GetPromptRequestParams(
            name=prompt_name,
            arguments=arguments,
        )
        return await client.get_prompt(get_prompt_request_params)

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

        Uses anyio.fail_after() instead of asyncio.wait_for() to avoid conflicts
        with the MCP SDK's anyio TaskGroup. See GitHub issue #20715 for details.

        Args:
            client: MCP client instance
            server_name: Name of the server for logging

        Returns:
            List of tools from the server
        """
        try:
            with anyio.fail_after(30.0):
                tools = await client.list_tools()
                verbose_logger.debug(f"Tools from {server_name}: {tools}")
                return tools
        except TimeoutError:
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
            tool_copy = tool.model_copy(deep=True)

            original_name = tool_copy.name
            prefixed_name = add_server_prefix_to_name(original_name, prefix)

            name_to_use = prefixed_name if add_prefix else original_name

            # Preserve all tool fields including metadata/_meta by avoiding mutation
            tool_copy.name = name_to_use
            prefixed_tools.append(tool_copy)

            # Update tool to server mapping for resolution (support both forms)
            self.tool_name_to_mcp_server_name_mapping[original_name] = prefix
            self.tool_name_to_mcp_server_name_mapping[prefixed_name] = prefix

        verbose_logger.info(
            f"Successfully fetched {len(prefixed_tools)} tools from server {server.name}"
        )
        return prefixed_tools

    def _create_prefixed_prompts(
        self, prompts: List[Prompt], server: MCPServer, add_prefix: bool = True
    ) -> List[Prompt]:
        """
        Create prefixed prompts and update prompt mapping.

        Args:
            prompts: List of original prompts from server
            server: Server instance

        Returns:
            List of prompts with prefixed names
        """
        prefixed_prompts = []
        prefix = get_server_prefix(server)

        for prompt in prompts:
            prefixed_name = add_server_prefix_to_name(prompt.name, prefix)

            name_to_use = prefixed_name if add_prefix else prompt.name

            prompt.name = name_to_use
            prefixed_prompts.append(prompt)

        verbose_logger.info(
            f"Successfully fetched {len(prefixed_prompts)} prompts from server {server.name}"
        )
        return prefixed_prompts

    def _create_prefixed_resources(
        self, resources: List[Resource], server: MCPServer, add_prefix: bool = True
    ) -> List[Resource]:
        """Prefix resource names and track origin server for read requests."""

        prefixed_resources: List[Resource] = []
        prefix = get_server_prefix(server)

        for resource in resources:
            name_to_use = (
                add_server_prefix_to_name(resource.name, prefix)
                if add_prefix
                else resource.name
            )
            resource.name = name_to_use
            prefixed_resources.append(resource)

        verbose_logger.info(
            f"Successfully fetched {len(prefixed_resources)} resources from server {server.name}"
        )
        return prefixed_resources

    def _create_prefixed_resource_templates(
        self,
        resource_templates: List[ResourceTemplate],
        server: MCPServer,
        add_prefix: bool = True,
    ) -> List[ResourceTemplate]:
        """Prefix resource template names for multi-server scenarios."""

        prefixed_templates: List[ResourceTemplate] = []
        prefix = get_server_prefix(server)

        for resource_template in resource_templates:
            name_to_use = (
                add_server_prefix_to_name(resource_template.name, prefix)
                if add_prefix
                else resource_template.name
            )
            resource_template.name = name_to_use
            prefixed_templates.append(resource_template)

        verbose_logger.info(
            f"Successfully fetched {len(prefixed_templates)} resource templates from server {server.name}"
        )
        return prefixed_templates

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
            split_server_prefix_from_name,
        )

        # If no allowed_params configured, return all arguments
        if not server.allowed_params:
            return

        # Get the unprefixed tool name to match against config
        unprefixed_tool_name, _ = split_server_prefix_from_name(tool_name)

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
            # Use standard pre_call_hook
            modified_data = await proxy_logging_obj.pre_call_hook(
                user_api_key_dict=user_api_key_auth,  # type: ignore
                data=synthetic_llm_data,
                call_type=CallTypes.call_mcp_tool.value,
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
                call_type=CallTypes.call_mcp_tool.value,
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
        host_progress_callback: Optional[Callable] = None,
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

            normalized_raw_headers = {
                str(k).lower(): v for k, v in raw_headers.items() if isinstance(k, str)
            }
            for header in mcp_server.extra_headers:
                if not isinstance(header, str):
                    continue
                header_value = normalized_raw_headers.get(header.lower())
                if header_value is None:
                    continue
                extra_headers[header] = header_value

        if mcp_server.static_headers:
            if extra_headers is None:
                extra_headers = {}
            extra_headers.update(mcp_server.static_headers)

        stdio_env = self._build_stdio_env(mcp_server, raw_headers)

        client = await self._create_mcp_client(
            server=mcp_server,
            mcp_auth_header=server_auth_header,
            extra_headers=extra_headers,
            stdio_env=stdio_env,
        )

        call_tool_params = MCPCallToolRequestParams(
            name=original_tool_name,
            arguments=arguments,
        )

        async def _call_tool_via_client(client, params):
            return await client.call_tool(
                params, host_progress_callback=host_progress_callback
            )

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
        host_progress_callback: Optional[Callable] = None,
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
        prefixed_tool_name = add_server_prefix_to_name(name, server_name)
        mcp_server = self._get_mcp_server_from_tool_name(prefixed_tool_name)
        if mcp_server is None:
            raise ValueError(f"Tool {name} not found")

        #########################################################
        # Pre MCP Tool Call Hook
        # Allow validation and modification of tool calls before execution
        # Using standard pre_call_hook
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
                host_progress_callback=host_progress_callback,
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
            if server.needs_user_oauth_token:
                # Skip OAuth2 servers that rely on user-provided tokens
                continue
            tools = await self._get_tools_from_server(server)
            for tool in tools:
                # The tool.name here is already prefixed from _get_tools_from_server
                # Extract original name for mapping
                original_name, _ = split_server_prefix_from_name(tool.name)
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
            ) = split_server_prefix_from_name(tool_name)
            if original_tool_name in self.tool_name_to_mcp_server_name_mapping:
                for server in self.get_registry().values():
                    if server.server_name is None:
                        if normalize_server_name(server.name) == normalize_server_name(
                            server_name_from_prefix
                        ):
                            return server
                    elif normalize_server_name(
                        server.server_name
                    ) == normalize_server_name(server_name_from_prefix):
                        return server

        return None

    async def reload_servers_from_database(self):
        """Re-synchronize the in-memory MCP server registry with the database."""
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

        previous_registry = self.registry
        new_registry: Dict[str, MCPServer] = {}

        for server in db_mcp_servers:
            existing_server = previous_registry.get(server.server_id)

            if (
                existing_server is not None
                and existing_server.updated_at is not None
                and server.updated_at is not None
                and existing_server.updated_at == server.updated_at
            ):
                # Re-use existing server instance to avoid re-running build_mcp_server_from_table()
                # which can perform network discovery for OAuth2 servers.
                new_registry[server.server_id] = existing_server
                continue

            _warn_on_server_name_fields(
                server_id=server.server_id,
                alias=getattr(server, "alias", None),
                server_name=getattr(server, "server_name", None),
            )
            verbose_logger.debug(
                f"Building server from DB: {server.server_id} ({server.server_name})"
            )
            new_registry[server.server_id] = await self.build_mcp_server_from_table(
                server
            )

        self.registry = new_registry

        verbose_logger.debug(
            "MCP registry refreshed (%s servers in registry)", len(new_registry)
        )

    def get_mcp_servers_from_ids(self, server_ids: List[str]) -> List[MCPServer]:
        servers = []
        registry = self.get_registry()
        for server in registry.values():
            if server.server_id in server_ids:
                servers.append(server)
        return servers

    def _get_general_settings(self) -> Dict[str, Any]:
        """Get general_settings, importing lazily to avoid circular imports."""
        try:
            from litellm.proxy.proxy_server import (
                general_settings as proxy_general_settings,
            )

            return proxy_general_settings
        except ImportError:
            # Fallback if proxy_server not available
            return {}

    def _is_server_accessible_from_ip(
        self, server: MCPServer, client_ip: Optional[str]
    ) -> bool:
        """
        Check if a server is accessible from the given client IP.

        - If client_ip is None, no IP filtering is applied (internal callers).
        - If the server has available_on_public_internet=True, it's always accessible.
        - Otherwise, only internal/private IPs can access it.
        """
        if client_ip is None:
            return True
        if server.available_on_public_internet:
            return True
        # Check backwards compat: litellm.public_mcp_servers
        public_ids = set(litellm.public_mcp_servers or [])
        if server.server_id in public_ids:
            return True
        # Non-public server: only accessible from internal IPs
        general_settings = self._get_general_settings()
        internal_networks = IPAddressUtils.parse_internal_networks(
            general_settings.get("mcp_internal_ip_ranges")
        )
        return IPAddressUtils.is_internal_ip(client_ip, internal_networks)

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
        Get the public MCP servers (available_on_public_internet=True flag on server).
        Also includes servers from litellm.public_mcp_servers for backwards compat.
        """
        servers: List[MCPServer] = []
        public_ids = set(litellm.public_mcp_servers or [])
        for server in self.get_registry().values():
            if server.available_on_public_internet or server.server_id in public_ids:
                servers.append(server)
        return servers

    def get_mcp_server_by_name(
        self, server_name: str, client_ip: Optional[str] = None
    ) -> Optional[MCPServer]:
        """
        Get the MCP Server from the server name.

        Uses priority-based matching to avoid collisions:
        1. First pass: exact alias match (highest priority)
        2. Second pass: exact server_name match
        3. Third pass: exact name match (lowest priority)

        Args:
            server_name: The server name to look up.
            client_ip: Optional client IP for access control. When provided,
                       non-public servers are hidden from external IPs.
        """
        registry = self.get_registry()
        # Pass 1: Match by alias (highest priority)
        for server in registry.values():
            if server.alias == server_name:
                if not self._is_server_accessible_from_ip(server, client_ip):
                    return None
                return server
        # Pass 2: Match by server_name
        for server in registry.values():
            if server.server_name == server_name:
                if not self._is_server_accessible_from_ip(server, client_ip):
                    return None
                return server
        # Pass 3: Match by name (lowest priority)
        for server in registry.values():
            if server.name == server_name:
                if not self._is_server_accessible_from_ip(server, client_ip):
                    return None
                return server
        return None

    def get_filtered_registry(
        self, client_ip: Optional[str] = None
    ) -> Dict[str, MCPServer]:
        """
        Get registry filtered by client IP access control.

        Args:
            client_ip: Optional client IP. When provided, non-public servers
                       are hidden from external IPs. When None, returns all servers.
        """
        registry = self.get_registry()
        if client_ip is None:
            return registry
        return {
            k: v
            for k, v in registry.items()
            if self._is_server_accessible_from_ip(v, client_ip)
        }

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
    ) -> LiteLLM_MCPServerTable:
        """
        Perform a health check on a specific MCP server.

        Args:
            server_id: The ID of the server to health check
            mcp_auth_header: Optional authentication header for the MCP server

        Returns:
            Dict containing health check results
        """
        from datetime import datetime

        server = self.get_mcp_server_by_id(server_id)
        if not server:
            verbose_logger.warning(f"MCP Server {server_id} not found")
            return LiteLLM_MCPServerTable(
                server_id=server_id,
                server_name=None,
                transport=MCPTransport.http,  # Default transport for not found servers
                status="unknown",
                health_check_error="Server not found",
                last_health_check=datetime.now(),
            )

        status: Literal["healthy", "unhealthy", "unknown"] = "unknown"
        health_check_error = None

        # Check if we should skip health check based on auth configuration
        should_skip_health_check = False

        # Skip if auth_type is oauth2
        if server.needs_user_oauth_token:
            should_skip_health_check = True
        # Skip if auth_type is not none and authentication_token is missing
        elif (
            server.auth_type
            and server.auth_type != MCPAuth.none
            and not server.authentication_token
        ):
            should_skip_health_check = True

        if not should_skip_health_check:
            extra_headers = {}
            if server.static_headers:
                extra_headers.update(server.static_headers)

            client = await self._create_mcp_client(
                server=server,
                mcp_auth_header=None,
                extra_headers=extra_headers,
                stdio_env=None,
            )

            try:

                async def _noop(session):
                    return "ok"

                # Add timeout wrapper to prevent hanging
                await asyncio.wait_for(client.run_with_session(_noop), timeout=10.0)
                status = "healthy"
            except asyncio.TimeoutError:
                health_check_error = "Health check timed out after 10 seconds"
                status = "unhealthy"
            except asyncio.CancelledError:
                health_check_error = "Health check was cancelled"
                status = "unknown"
            except Exception as e:
                health_check_error = str(e)
                status = "unhealthy"

        return LiteLLM_MCPServerTable(
            server_id=server.server_id,
            server_name=server.server_name,
            alias=server.alias,
            description=(
                server.mcp_info.get("description") if server.mcp_info else None
            ),
            url=server.url,
            transport=server.transport,
            auth_type=server.auth_type,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            teams=[],
            mcp_access_groups=server.access_groups or [],
            allowed_tools=server.allowed_tools or [],
            extra_headers=server.extra_headers or [],
            mcp_info=server.mcp_info,
            static_headers=server.static_headers,
            status=status,
            last_health_check=datetime.now(),
            health_check_error=health_check_error,
            command=getattr(server, "command", None),
            args=getattr(server, "args", None) or [],
            env=getattr(server, "env", None) or {},
            authorization_url=server.authorization_url,
            token_url=server.token_url,
            registration_url=server.registration_url,
            allow_all_keys=server.allow_all_keys,
        )

    async def get_all_mcp_servers_with_health_and_teams(
        self,
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
        server_ids: Optional[List[str]] = None,
    ) -> List[LiteLLM_MCPServerTable]:
        """
        Get all MCP servers that the user has access to, with health status and team information.

        Args:
            user_api_key_auth: User authentication info for access control
            server_ids: Optional list of server IDs to filter. If provided, only these servers
                       will be checked (subject to access control). If None, all accessible servers are checked.

        Returns:
            List of MCP server objects with health and team data
        """

        # Get allowed server IDs
        allowed_server_ids = await self.get_allowed_mcp_servers(user_api_key_auth)

        # Filter by requested server_ids if provided
        if server_ids:
            # Only check servers that are both requested AND accessible
            target_server_ids = [sid for sid in server_ids if sid in allowed_server_ids]
        else:
            # Check all accessible servers
            target_server_ids = allowed_server_ids

        return await self._run_health_checks(target_server_ids)

    async def get_all_allowed_mcp_servers(
        self,
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ) -> List[LiteLLM_MCPServerTable]:
        """
        Get all MCP servers that the user has access to.

        Args:
            user_api_key_auth: User authentication info for access control

        Returns:
            List of MCP server objects without health status
        """
        # Get allowed server IDs
        allowed_server_ids = await self.get_allowed_mcp_servers(user_api_key_auth)

        list_mcp_servers: List[LiteLLM_MCPServerTable] = []

        for server_id in allowed_server_ids:
            server = self.get_mcp_server_by_id(server_id)
            if not server:
                verbose_logger.warning(f"MCP Server {server_id} not found in registry")
                continue

            mcp_server_table = self._build_mcp_server_table(server)
            list_mcp_servers.append(mcp_server_table)

        return list_mcp_servers

    def _build_mcp_server_table(self, server: MCPServer) -> LiteLLM_MCPServerTable:
        from datetime import datetime

        return LiteLLM_MCPServerTable(
            server_id=server.server_id,
            server_name=server.server_name,
            alias=server.alias,
            description=(
                server.mcp_info.get("description") if server.mcp_info else None
            ),
            url=server.url,
            transport=server.transport,
            auth_type=server.auth_type,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            teams=[],
            mcp_access_groups=server.access_groups or [],
            allowed_tools=server.allowed_tools or [],
            extra_headers=server.extra_headers or [],
            mcp_info=server.mcp_info,
            static_headers=server.static_headers,
            status=None,  # No health check performed
            last_health_check=None,  # No health check performed
            health_check_error=None,
            command=getattr(server, "command", None),
            args=getattr(server, "args", None) or [],
            env=getattr(server, "env", None) or {},
            authorization_url=server.authorization_url,
            token_url=server.token_url,
            registration_url=server.registration_url,
            allow_all_keys=server.allow_all_keys,
            available_on_public_internet=server.available_on_public_internet,
        )

    async def get_all_mcp_servers_unfiltered(self) -> List[LiteLLM_MCPServerTable]:
        """Return all MCP servers from registry without applying access controls."""

        registry = self.get_registry()
        if not registry:
            return []

        servers: List[LiteLLM_MCPServerTable] = []
        for server in registry.values():
            servers.append(self._build_mcp_server_table(server))
        return servers

    async def get_all_mcp_servers_with_health_unfiltered(
        self, server_ids: Optional[List[str]] = None
    ) -> List[LiteLLM_MCPServerTable]:
        """Return health info for all servers in registry regardless of user access."""

        registry = self.get_registry()
        if not registry:
            return []

        if server_ids:
            target_server_ids = [sid for sid in server_ids if sid in registry]
        else:
            target_server_ids = list(registry.keys())

        if not target_server_ids:
            return []

        return await self._run_health_checks(target_server_ids)

    async def _run_health_checks(
        self, target_server_ids: List[str]
    ) -> List[LiteLLM_MCPServerTable]:
        if not target_server_ids:
            return []

        tasks = [self.health_check_server(server_id) for server_id in target_server_ids]
        results = await asyncio.gather(*tasks)
        return [server for server in results if server is not None]


global_mcp_server_manager: MCPServerManager = MCPServerManager()
