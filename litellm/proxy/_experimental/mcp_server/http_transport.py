"""
HTTP Streamable Transport for MCP Server

This implements the Streamable HTTP transport mechanism for MCP as defined in the
specification: https://modelcontextprotocol.io/specification/2025-03-26/basic/transports

Based on the MCP Streamable HTTP specification and using StreamableHTTPSessionManager.
"""

import contextlib
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

from litellm._logging import verbose_logger

# Check if MCP streamable http is available
try:
    from mcp.server import Server
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

    STREAMABLE_HTTP_AVAILABLE = True
except ImportError as e:
    verbose_logger.debug(f"MCP StreamableHTTPSessionManager not found: {e}")
    STREAMABLE_HTTP_AVAILABLE = False
    # Define dummy types for when not available
    StreamableHTTPSessionManager = None
    Server = None


class HttpServerTransport:
    """
    HTTP Streamable server transport for MCP. This class provides an ASGI application
    suitable to be used with FastAPI that handles both POST and GET requests
    for MCP communication according to the Streamable HTTP specification.
    """

    def __init__(self, mcp_server: Any, endpoint: str = "/mcp", stateless: bool = True):
        """
        Creates a new HTTP Streamable server transport.

        Args:
            mcp_server: The MCP Server instance to handle requests
            endpoint: The endpoint path for MCP communication
            stateless: Whether to run in stateless mode (default: True)
        """
        if not STREAMABLE_HTTP_AVAILABLE:
            raise ImportError(
                "StreamableHTTPSessionManager not available. Please install mcp with streamable http support."
            )

        self.mcp_server = mcp_server
        self.endpoint = endpoint
        self.stateless = stateless
        self._session_manager: Optional[Any] = None
        self._initialized = False

        verbose_logger.debug(
            f"HttpServerTransport initialized with endpoint: {endpoint}, stateless: {stateless}"
        )

    @property
    def session_manager(self) -> Any:
        """Get the session manager, creating it if necessary."""
        if self._session_manager is None:
            if not STREAMABLE_HTTP_AVAILABLE:
                raise ImportError("StreamableHTTPSessionManager not available")
            # Create the session manager with JSON responses (stateless HTTP)
            # StreamableHTTPSessionManager is guaranteed to be available here due to the check above
            from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

            self._session_manager = StreamableHTTPSessionManager(
                app=self.mcp_server,
                event_store=None,
                json_response=True,  # Use JSON responses for HTTP transport
                stateless=self.stateless,
            )
            verbose_logger.debug("Created StreamableHTTPSessionManager")
        return self._session_manager

    @contextlib.asynccontextmanager
    async def run_session_manager(self) -> AsyncIterator[None]:
        """
        Context manager to run the session manager.
        """
        async with self.session_manager.run():
            verbose_logger.info("HTTP Streamable session manager started")
            try:
                yield
            finally:
                verbose_logger.info("HTTP Streamable session manager shutting down")

    async def handle_request(
        self,
        scope: dict,
        receive: Callable[[], Awaitable[dict]],
        send: Callable[[dict], Awaitable[None]],
    ) -> None:
        """
        Handle MCP requests through Streamable HTTP.

        This method handles both POST and GET requests according to the
        Streamable HTTP specification:
        - POST: Client sends JSON-RPC messages to server
        - GET: Optional SSE stream for server-to-client communication

        Args:
            scope: ASGI scope dictionary
            receive: ASGI receive callable
            send: ASGI send callable
        """
        try:
            await self.session_manager.handle_request(scope, receive, send)
        except Exception as e:
            verbose_logger.error(f"Error handling HTTP request: {e}")
            try:
                # Send error response if possible
                await send(
                    {
                        "type": "http.response.start",
                        "status": 500,
                        "headers": [(b"content-type", b"application/json")],
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": b'{"error": "Internal server error"}',
                    }
                )
            except Exception as send_error:
                verbose_logger.error(f"Failed to send error response: {send_error}")
                # If we can't send the error response, there's not much more we can do
