"""
LiteLLM Proxy uses this MCP Client to connnect to other MCP servers.
"""
import base64
from datetime import timedelta
from typing import List, Optional

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolRequestParams as MCPCallToolRequestParams
from mcp.types import CallToolResult as MCPCallToolResult
from mcp.types import Tool as MCPTool

from litellm.types.mcp import MCPAuth, MCPAuthType, MCPTransport, MCPTransportType


def to_basic_auth(auth_value: str) -> str:
    """Convert auth value to Basic Auth format."""
    return base64.b64encode(auth_value.encode("utf-8")).decode()


class MCPClient:
    """
    MCP Client supporting:
      SSE and HTTP transports
      Authentication via Bearer token, Basic Auth, or API Key
      Tool calling with error handling and result parsing
    """

    def __init__(
        self,
        server_url: str,
        transport_type: MCPTransportType = MCPTransport.http,
        auth_type: MCPAuthType = None,
        auth_value: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.server_url: str = server_url
        self.transport_type: MCPTransport = transport_type
        self.auth_type: MCPAuthType = auth_type
        self.timeout: float = timeout
        self._mcp_auth_value: Optional[str] = None
        self._session: Optional[ClientSession] = None
        self._context = None
        self._transport_ctx = None
        self._transport = None
        self._session_ctx = None

        # handle the basic auth value if provided
        if auth_value:
            self.update_auth_value(auth_value)

    async def __aenter__(self):
        """
        Enable async context manager support.
          Initializes the transport and session.
        """
        await self.connect()
        return self

    async def connect(self):
        """Initialize the transport and session."""
        if self._session:
            return  # Already connected
            
        headers = self._get_auth_headers()

        if self.transport_type == MCPTransport.sse:
            self._transport_ctx = sse_client(
                url=self.server_url,
                timeout=self.timeout,
                headers=headers,
            )
            self._transport = await self._transport_ctx.__aenter__()
            self._session_ctx = ClientSession(self._transport[0], self._transport[1])
            self._session = await self._session_ctx.__aenter__()
            await self._session.initialize()
        else:
            self._transport_ctx = streamablehttp_client(
                url=self.server_url,
                timeout=timedelta(seconds=self.timeout),
                headers=headers,
            )
            self._transport = await self._transport_ctx.__aenter__()
            self._session_ctx = ClientSession(self._transport[0], self._transport[1])
            self._session = await self._session_ctx.__aenter__()
            await self._session.initialize()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup when exiting context manager."""
        if self._session:
            await self._session_ctx.__aexit__(exc_type, exc_val, exc_tb) # type: ignore
        if self._transport_ctx:
            await self._transport_ctx.__aexit__(exc_type, exc_val, exc_tb)

    async def disconnect(self):
        """Clean up session and connections."""
        if self._session:
            try:
                # Ensure session is properly closed
                await self._session.close()  # type: ignore 
            except Exception:
                pass
            self._session = None

        if self._context:
            try:
                await self._context.__aexit__(None, None, None) # type: ignore
            except Exception:
                pass
            self._context = None

    def update_auth_value(self, mcp_auth_value: str):
        """
        Set the authentication header for the MCP client.
        """
        if self.auth_type == MCPAuth.basic:
            # Assuming mcp_auth_value is in format "username:password", convert it when updating
            mcp_auth_value = to_basic_auth(mcp_auth_value)
        self._mcp_auth_value = mcp_auth_value

    def _get_auth_headers(self) -> dict:
        """Generate authentication headers based on auth type."""
        if not self._mcp_auth_value:
            return {}

        if self.auth_type == MCPAuth.bearer_token:
            return {"Authorization": f"Bearer {self._mcp_auth_value}"}
        elif self.auth_type == MCPAuth.basic:
            return {"Authorization": f"Basic {self._mcp_auth_value}"}
        elif self.auth_type == MCPAuth.api_key:
            return {"X-API-Key": self._mcp_auth_value}
        return {}

    async def list_tools(self) -> List[MCPTool]:
        """List available tools from the server."""
        if not self._session:
            await self.connect()
        if self._session is None:
            raise ValueError("Session is not initialized")

        result = await self._session.list_tools()
        return result.tools

    async def call_tool(
        self, call_tool_request_params: MCPCallToolRequestParams
    ) -> MCPCallToolResult:
        """
        Call an MCP Tool.
        """
        if not self._session:
            await self.connect()

        if self._session is None:
            raise ValueError("Session is not initialized")
        
        tool_result = await self._session.call_tool(
            name=call_tool_request_params.name,
            arguments=call_tool_request_params.arguments,
        )
        return tool_result
        

