"""
LiteLLM Proxy uses this MCP Client to connnect to other MCP servers.
"""

import asyncio
import base64
from datetime import timedelta
from typing import Awaitable, Callable, Dict, List, Optional, TypeVar, Union

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolRequestParams as MCPCallToolRequestParams
from mcp.types import CallToolResult as MCPCallToolResult
from mcp.types import TextContent
from mcp.types import Tool as MCPTool

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import get_ssl_configuration
from litellm.types.llms.custom_http import VerifyTypes
from litellm.types.mcp import (
    MCPAuth,
    MCPAuthType,
    MCPStdioConfig,
    MCPTransport,
    MCPTransportType,
)


def to_basic_auth(auth_value: str) -> str:
    """Convert auth value to Basic Auth format."""
    return base64.b64encode(auth_value.encode("utf-8")).decode()


TSessionResult = TypeVar("TSessionResult")


class MCPClient:
    """
    MCP Client supporting:
      SSE and HTTP transports
      Authentication via Bearer token, Basic Auth, or API Key
      Tool calling with error handling and result parsing
    """

    def __init__(
        self,
        server_url: str = "",
        transport_type: MCPTransportType = MCPTransport.http,
        auth_type: MCPAuthType = None,
        auth_value: Optional[Union[str, Dict[str, str]]] = None,
        timeout: float = 60.0,
        stdio_config: Optional[MCPStdioConfig] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        ssl_verify: Optional[VerifyTypes] = None,
    ):
        self.server_url: str = server_url
        self.transport_type: MCPTransport = transport_type
        self.auth_type: MCPAuthType = auth_type
        self.timeout: float = timeout
        self._mcp_auth_value: Optional[Union[str, Dict[str, str]]] = None
        self.stdio_config: Optional[MCPStdioConfig] = stdio_config
        self.extra_headers: Optional[Dict[str, str]] = extra_headers
        self.ssl_verify: Optional[VerifyTypes] = ssl_verify
        # handle the basic auth value if provided
        if auth_value:
            self.update_auth_value(auth_value)

    async def run_with_session(
        self, operation: Callable[[ClientSession], Awaitable[TSessionResult]]
    ) -> TSessionResult:
        """Open a session, run the provided coroutine, and clean up."""
        transport_ctx = None

        try:
            if self.transport_type == MCPTransport.stdio:
                if not self.stdio_config:
                    raise ValueError("stdio_config is required for stdio transport")

                server_params = StdioServerParameters(
                    command=self.stdio_config.get("command", ""),
                    args=self.stdio_config.get("args", []),
                    env=self.stdio_config.get("env", {}),
                )
                transport_ctx = stdio_client(server_params)
            elif self.transport_type == MCPTransport.sse:
                headers = self._get_auth_headers()
                httpx_client_factory = self._create_httpx_client_factory()
                transport_ctx = sse_client(
                    url=self.server_url,
                    timeout=self.timeout,
                    headers=headers,
                    httpx_client_factory=httpx_client_factory,
                )
            else:
                headers = self._get_auth_headers()
                httpx_client_factory = self._create_httpx_client_factory()
                verbose_logger.debug(
                    "litellm headers for streamablehttp_client: %s", headers
                )
                transport_ctx = streamablehttp_client(
                    url=self.server_url,
                    timeout=timedelta(seconds=self.timeout),
                    headers=headers,
                    httpx_client_factory=httpx_client_factory,
                )

            if transport_ctx is None:
                raise RuntimeError("Failed to create transport context")

            async with transport_ctx as transport:
                read_stream, write_stream = transport[0], transport[1]
                session_ctx = ClientSession(read_stream, write_stream)
                async with session_ctx as session:
                    await session.initialize()
                    return await operation(session)
        except Exception:
            verbose_logger.warning(
                "MCP client run_with_session failed for %s", self.server_url or "stdio"
            )
            raise

    def update_auth_value(self, mcp_auth_value: Union[str, Dict[str, str]]):
        """
        Set the authentication header for the MCP client.
        """
        if isinstance(mcp_auth_value, dict):
            self._mcp_auth_value = mcp_auth_value
        else:
            if self.auth_type == MCPAuth.basic:
                # Assuming mcp_auth_value is in format "username:password", convert it when updating
                mcp_auth_value = to_basic_auth(mcp_auth_value)
            self._mcp_auth_value = mcp_auth_value

    def _get_auth_headers(self) -> dict:
        """Generate authentication headers based on auth type."""
        headers = {}

        if self._mcp_auth_value:
            if isinstance(self._mcp_auth_value, str):
                if self.auth_type == MCPAuth.bearer_token:
                    headers["Authorization"] = f"Bearer {self._mcp_auth_value}"
                elif self.auth_type == MCPAuth.basic:
                    headers["Authorization"] = f"Basic {self._mcp_auth_value}"
                elif self.auth_type == MCPAuth.api_key:
                    headers["X-API-Key"] = self._mcp_auth_value
                elif self.auth_type == MCPAuth.authorization:
                    headers["Authorization"] = self._mcp_auth_value
            elif isinstance(self._mcp_auth_value, dict):
                headers.update(self._mcp_auth_value)

        # update the headers with the extra headers
        if self.extra_headers:
            headers.update(self.extra_headers)

        return headers

    def _create_httpx_client_factory(self) -> Callable[..., httpx.AsyncClient]:
        """
        Create a custom httpx client factory that uses LiteLLM's SSL configuration.

        This factory follows the same CA bundle path logic as http_handler.py:
        1. Check ssl_verify parameter (can be SSLContext, bool, or path to CA bundle)
        2. Check SSL_VERIFY environment variable
        3. Check SSL_CERT_FILE environment variable
        4. Fall back to certifi CA bundle
        """

        def factory(
            *,
            headers: Optional[Dict[str, str]] = None,
            timeout: Optional[httpx.Timeout] = None,
            auth: Optional[httpx.Auth] = None,
        ) -> httpx.AsyncClient:
            """Create an httpx.AsyncClient with LiteLLM's SSL configuration."""
            # Get unified SSL configuration using the same logic as http_handler.py
            ssl_config = get_ssl_configuration(self.ssl_verify)

            verbose_logger.debug(
                f"MCP client using SSL configuration: {type(ssl_config).__name__}"
            )

            return httpx.AsyncClient(
                headers=headers,
                timeout=timeout,
                auth=auth,
                verify=ssl_config,
                follow_redirects=True,
            )

        return factory

    async def list_tools(self) -> List[MCPTool]:
        """List available tools from the server."""
        verbose_logger.debug(
            f"MCP client listing tools from {self.server_url or 'stdio'}"
        )

        async def _list_tools_operation(session: ClientSession):
            return await session.list_tools()

        try:
            result = await self.run_with_session(_list_tools_operation)
            tool_count = len(result.tools)
            tool_names = [tool.name for tool in result.tools]
            verbose_logger.info(
                f"MCP client listed {tool_count} tools from {self.server_url or 'stdio'}: {tool_names}"
            )
            return result.tools
        except asyncio.CancelledError:
            verbose_logger.warning("MCP client list_tools was cancelled")
            raise
        except Exception as e:
            error_type = type(e).__name__
            verbose_logger.error(
                f"MCP client list_tools failed - "
                f"Error Type: {error_type}, "
                f"Error: {str(e)}, "
                f"Server: {self.server_url or 'stdio'}, "
                f"Transport: {self.transport_type}"
            )

            # Check if it's a stream/connection error
            if "BrokenResourceError" in error_type or "Broken" in error_type:
                verbose_logger.error(
                    "MCP client detected broken connection/stream during list_tools - "
                    "the MCP server may have crashed, disconnected, or timed out"
                )

            # Return empty list instead of raising to allow graceful degradation
            return []

    async def call_tool(
        self, call_tool_request_params: MCPCallToolRequestParams
    ) -> MCPCallToolResult:
        """
        Call an MCP Tool.
        """
        verbose_logger.info(
            f"MCP client calling tool '{call_tool_request_params.name}' with arguments: {call_tool_request_params.arguments}"
        )

        async def _call_tool_operation(session: ClientSession):
            verbose_logger.debug("MCP client sending tool call to session")
            return await session.call_tool(
                name=call_tool_request_params.name,
                arguments=call_tool_request_params.arguments,
            )

        try:
            tool_result = await self.run_with_session(_call_tool_operation)
            verbose_logger.info(
                f"MCP client tool call '{call_tool_request_params.name}' completed successfully"
            )
            return tool_result
        except asyncio.CancelledError:
            verbose_logger.warning("MCP client tool call was cancelled")
            raise
        except Exception as e:
            import traceback

            error_trace = traceback.format_exc()
            verbose_logger.debug(f"MCP client tool call traceback:\n{error_trace}")

            # Log detailed error information
            error_type = type(e).__name__
            verbose_logger.error(
                f"MCP client call_tool failed - "
                f"Error Type: {error_type}, "
                f"Error: {str(e)}, "
                f"Tool: {call_tool_request_params.name}, "
                f"Server: {self.server_url or 'stdio'}, "
                f"Transport: {self.transport_type}"
            )

            # Check if it's a stream/connection error
            if "BrokenResourceError" in error_type or "Broken" in error_type:
                verbose_logger.error(
                    "MCP client detected broken connection/stream - "
                    "the MCP server may have crashed, disconnected, or timed out."
                )

            # Return a default error result instead of raising
            return MCPCallToolResult(
                content=[
                    TextContent(type="text", text=f"{error_type}: {str(e)}")
                ],  # Empty content for error case
                isError=True,
            )
