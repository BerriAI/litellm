"""
LiteLLM Proxy uses this MCP Client to connnect to other MCP servers.
"""

import asyncio
import base64
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import httpx
from mcp import ClientSession, ReadResourceResult, Resource, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import (
    CallToolRequestParams as MCPCallToolRequestParams,
    GetPromptRequestParams,
    GetPromptResult,
    Prompt,
    ResourceTemplate,
)
from mcp.types import CallToolResult as MCPCallToolResult
from mcp.types import TextContent
from mcp.types import Tool as MCPTool
from pydantic import AnyUrl

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
        use_session_cache: bool = False,
        session_cache_ttl: float = 300.0,  # 5 minutes default
    ):
        self.server_url: str = server_url
        self.transport_type: MCPTransport = transport_type
        self.auth_type: MCPAuthType = auth_type
        self.timeout: float = timeout
        self._mcp_auth_value: Optional[Union[str, Dict[str, str]]] = None
        self.stdio_config: Optional[MCPStdioConfig] = stdio_config
        self.extra_headers: Optional[Dict[str, str]] = extra_headers
        self.ssl_verify: Optional[VerifyTypes] = ssl_verify

        # Session caching configuration
        self.use_session_cache: bool = use_session_cache
        self.session_cache_ttl: float = session_cache_ttl

        # Cached session state
        self._cached_session: Optional[ClientSession] = None
        self._cached_transport_ctx: Optional[Any] = None
        self._cached_http_client: Optional[httpx.AsyncClient] = None
        self._session_last_used_at: Optional[float] = None
        self._session_lock: asyncio.Lock = asyncio.Lock()

        # handle the basic auth value if provided
        if auth_value:
            self.update_auth_value(auth_value)

    async def run_with_session(
        self, operation: Callable[[ClientSession], Awaitable[TSessionResult]]
    ) -> TSessionResult:
        """Open a session, run the provided coroutine, and clean up."""
        transport_ctx, http_client = self._create_transport_context()
        try:
            async with transport_ctx as transport:
                read_stream, write_stream = transport[0], transport[1]
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    return await operation(session)
        except Exception:
            verbose_logger.warning(
                "MCP client run_with_session failed for %s", self.server_url or "stdio"
            )
            raise
        finally:
            if http_client is not None:
                await http_client.aclose()

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

    def _create_transport_context(self) -> Tuple[Any, Optional[httpx.AsyncClient]]:
        """Create transport context based on transport type."""
        http_client: Optional[httpx.AsyncClient] = None

        if self.transport_type == MCPTransport.stdio:
            if not self.stdio_config:
                raise ValueError("stdio_config is required for stdio transport")
            server_params = StdioServerParameters(
                command=self.stdio_config.get("command", ""),
                args=self.stdio_config.get("args", []),
                env=self.stdio_config.get("env", {}),
            )
            return stdio_client(server_params), None

        headers = self._get_auth_headers()
        httpx_client_factory = self._create_httpx_client_factory()

        if self.transport_type == MCPTransport.sse:
            return sse_client(
                url=self.server_url,
                timeout=self.timeout,
                headers=headers,
                httpx_client_factory=httpx_client_factory,
            ), None

        verbose_logger.debug("litellm headers for streamable_http_client: %s", headers)
        http_client = httpx_client_factory(
            headers=headers,
            timeout=httpx.Timeout(self.timeout),
        )
        return streamable_http_client(url=self.server_url, http_client=http_client), http_client

    def _is_session_valid(self) -> bool:
        """Check if the cached session is still valid (not idle too long)."""
        if self._cached_session is None:
            return False

        if self._session_last_used_at is None:
            return False

        # Check idle timeout
        idle_time = time.time() - self._session_last_used_at
        if idle_time > self.session_cache_ttl:
            verbose_logger.debug(
                f"MCP client cached session idle timeout (TTL={self.session_cache_ttl}s, idle={idle_time:.1f}s)"
            )
            return False

        return True

    async def _create_and_cache_session(self) -> ClientSession:
        """Create a new session and cache it."""
        await self._cleanup_cached_session()

        transport_ctx, http_client = self._create_transport_context()
        session = None
        try:
            transport = await transport_ctx.__aenter__()
            read_stream, write_stream = transport[0], transport[1]
            session = ClientSession(read_stream, write_stream)
            await session.__aenter__()
            await session.initialize()
        except Exception:
            # Clean up on failure to prevent resource leak
            if session is not None:
                try:
                    await session.__aexit__(None, None, None)
                except Exception:
                    pass
            try:
                await transport_ctx.__aexit__(None, None, None)
            except Exception:
                pass
            if http_client is not None:
                try:
                    await http_client.aclose()
                except Exception:
                    pass
            raise

        self._cached_transport_ctx = transport_ctx
        self._cached_session = session
        self._cached_http_client = http_client
        self._session_last_used_at = time.time()

        verbose_logger.debug(
            f"MCP client created and cached new session for {self.server_url or 'stdio'}"
        )
        return session

    async def _cleanup_cached_session(self) -> None:
        """Clean up any cached session resources."""
        if self._cached_session is not None:
            try:
                await self._cached_session.__aexit__(None, None, None)
            except Exception as e:
                verbose_logger.debug(f"Error closing cached session: {e}")
            self._cached_session = None

        if self._cached_transport_ctx is not None:
            try:
                await self._cached_transport_ctx.__aexit__(None, None, None)
            except Exception as e:
                verbose_logger.debug(f"Error closing cached transport: {e}")
            self._cached_transport_ctx = None

        if self._cached_http_client is not None:
            try:
                await self._cached_http_client.aclose()
            except Exception as e:
                verbose_logger.debug(f"Error closing cached http client: {e}")
            self._cached_http_client = None

        self._session_last_used_at = None

    async def _get_or_create_session(self) -> ClientSession:
        """Get a cached session or create a new one."""
        async with self._session_lock:
            if self._is_session_valid():
                verbose_logger.debug(
                    f"MCP client reusing cached session for {self.server_url or 'stdio'}"
                )
                return self._cached_session  # type: ignore

            return await self._create_and_cache_session()

    def _is_connection_error(self, e: Exception) -> bool:
        """Check if exception indicates a broken/closed connection."""
        if isinstance(e, (ConnectionError, ConnectionResetError, TimeoutError)):
            return True
        error_str = str(e).lower()
        return "broken" in error_str or "closed" in error_str

    async def run_with_cached_session(
        self, operation: Callable[[ClientSession], Awaitable[TSessionResult]]
    ) -> TSessionResult:
        """Run an operation using a cached session (connection pooling enabled)."""
        try:
            session = await self._get_or_create_session()
            result = await operation(session)
            self._session_last_used_at = time.time()
            return result
        except Exception as e:
            if self._is_connection_error(e):
                verbose_logger.warning(
                    f"MCP client cached session appears broken, retrying: {e}"
                )
                async with self._session_lock:
                    await self._cleanup_cached_session()
                    session = await self._create_and_cache_session()
                result = await operation(session)
                self._session_last_used_at = time.time()
                return result
            raise

    async def close(self) -> None:
        """Close the client and clean up any cached sessions."""
        async with self._session_lock:
            await self._cleanup_cached_session()
        verbose_logger.info(
            f"MCP client closed for {self.server_url or 'stdio'}"
        )

    async def _run_operation(
        self, operation: Callable[[ClientSession], Awaitable[TSessionResult]]
    ) -> TSessionResult:
        """Run an operation, using cached session if enabled."""
        if self.use_session_cache:
            return await self.run_with_cached_session(operation)
        return await self.run_with_session(operation)

    async def list_tools(self) -> List[MCPTool]:
        """List available tools from the server."""
        verbose_logger.debug(
            f"MCP client listing tools from {self.server_url or 'stdio'}"
        )

        async def _list_tools_operation(session: ClientSession):
            return await session.list_tools()

        try:
            result = await self._run_operation(_list_tools_operation)
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
            verbose_logger.exception(
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
            tool_result = await self._run_operation(_call_tool_operation)
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

    async def list_prompts(self) -> List[Prompt]:
        """List available prompts from the server."""
        verbose_logger.debug(
            f"MCP client listing tools from {self.server_url or 'stdio'}"
        )

        async def _list_prompts_operation(session: ClientSession):
            return await session.list_prompts()

        try:
            result = await self._run_operation(_list_prompts_operation)
            prompt_count = len(result.prompts)
            prompt_names = [prompt.name for prompt in result.prompts]
            verbose_logger.info(
                f"MCP client listed {prompt_count} tools from {self.server_url or 'stdio'}: {prompt_names}"
            )
            return result.prompts
        except asyncio.CancelledError:
            verbose_logger.warning("MCP client list_prompts was cancelled")
            raise
        except Exception as e:
            error_type = type(e).__name__
            verbose_logger.error(
                f"MCP client list_prompts failed - "
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

    async def get_prompt(
        self, get_prompt_request_params: GetPromptRequestParams
    ) -> GetPromptResult:
        """Fetch a prompt definition from the MCP server."""
        verbose_logger.info(
            f"MCP client fetching prompt '{get_prompt_request_params.name}' with arguments: {get_prompt_request_params.arguments}"
        )

        async def _get_prompt_operation(session: ClientSession):
            verbose_logger.debug("MCP client sending get_prompt request to session")
            return await session.get_prompt(
                name=get_prompt_request_params.name,
                arguments=get_prompt_request_params.arguments,
            )

        try:
            get_prompt_result = await self._run_operation(_get_prompt_operation)
            verbose_logger.info(
                f"MCP client get_prompt '{get_prompt_request_params.name}' completed successfully"
            )
            return get_prompt_result
        except asyncio.CancelledError:
            verbose_logger.warning("MCP client get_prompt was cancelled")
            raise
        except Exception as e:
            import traceback

            error_trace = traceback.format_exc()
            verbose_logger.debug(f"MCP client get_prompt traceback:\n{error_trace}")

            # Log detailed error information
            error_type = type(e).__name__
            verbose_logger.error(
                f"MCP client get_prompt failed - "
                f"Error Type: {error_type}, "
                f"Error: {str(e)}, "
                f"Prompt: {get_prompt_request_params.name}, "
                f"Server: {self.server_url or 'stdio'}, "
                f"Transport: {self.transport_type}"
            )

            # Check if it's a stream/connection error
            if "BrokenResourceError" in error_type or "Broken" in error_type:
                verbose_logger.error(
                    "MCP client detected broken connection/stream during get_prompt - "
                    "the MCP server may have crashed, disconnected, or timed out."
                )

            raise

    async def list_resources(self) -> list[Resource]:
        """List available resources from the server."""
        verbose_logger.debug(
            f"MCP client listing resources from {self.server_url or 'stdio'}"
        )

        async def _list_resources_operation(session: ClientSession):
            return await session.list_resources()

        try:
            result = await self._run_operation(_list_resources_operation)
            resource_count = len(result.resources)
            resource_names = [resource.name for resource in result.resources]
            verbose_logger.info(
                f"MCP client listed {resource_count} resources from {self.server_url or 'stdio'}: {resource_names}"
            )
            return result.resources
        except asyncio.CancelledError:
            verbose_logger.warning("MCP client list_resources was cancelled")
            raise
        except Exception as e:
            error_type = type(e).__name__
            verbose_logger.error(
                f"MCP client list_resources failed - "
                f"Error Type: {error_type}, "
                f"Error: {str(e)}, "
                f"Server: {self.server_url or 'stdio'}, "
                f"Transport: {self.transport_type}"
            )

            # Check if it's a stream/connection error
            if "BrokenResourceError" in error_type or "Broken" in error_type:
                verbose_logger.error(
                    "MCP client detected broken connection/stream during list_resources - "
                    "the MCP server may have crashed, disconnected, or timed out"
                )

            # Return empty list instead of raising to allow graceful degradation
            return []

    async def list_resource_templates(self) -> list[ResourceTemplate]:
        """List available resource templates from the server."""
        verbose_logger.debug(
            f"MCP client listing resource templates from {self.server_url or 'stdio'}"
        )

        async def _list_resource_templates_operation(session: ClientSession):
            return await session.list_resource_templates()

        try:
            result = await self._run_operation(_list_resource_templates_operation)
            resource_template_count = len(result.resourceTemplates)
            resource_template_names = [
                resourceTemplate.name for resourceTemplate in result.resourceTemplates
            ]
            verbose_logger.info(
                f"MCP client listed {resource_template_count} resource templates from {self.server_url or 'stdio'}: {resource_template_names}"
            )
            return result.resourceTemplates
        except asyncio.CancelledError:
            verbose_logger.warning("MCP client list_resource_templates was cancelled")
            raise
        except Exception as e:
            error_type = type(e).__name__
            verbose_logger.error(
                f"MCP client list_resource_templates failed - "
                f"Error Type: {error_type}, "
                f"Error: {str(e)}, "
                f"Server: {self.server_url or 'stdio'}, "
                f"Transport: {self.transport_type}"
            )

            # Check if it's a stream/connection error
            if "BrokenResourceError" in error_type or "Broken" in error_type:
                verbose_logger.error(
                    "MCP client detected broken connection/stream during list_resource_templates - "
                    "the MCP server may have crashed, disconnected, or timed out"
                )

            # Return empty list instead of raising to allow graceful degradation
            return []

    async def read_resource(self, url: AnyUrl) -> ReadResourceResult:
        """Fetch resource contents from the MCP server."""
        verbose_logger.info(f"MCP client fetching resource '{url}'")

        async def _read_resource_operation(session: ClientSession):
            verbose_logger.debug("MCP client sending read_resource request to session")
            return await session.read_resource(url)

        try:
            read_resource_result = await self._run_operation(_read_resource_operation)
            verbose_logger.info(
                f"MCP client read_resource '{url}' completed successfully"
            )
            return read_resource_result
        except asyncio.CancelledError:
            verbose_logger.warning("MCP client read_resource was cancelled")
            raise
        except Exception as e:
            import traceback

            error_trace = traceback.format_exc()
            verbose_logger.debug(f"MCP client read_resource traceback:\n{error_trace}")

            # Log detailed error information
            error_type = type(e).__name__
            verbose_logger.error(
                f"MCP client read_resource failed - "
                f"Error Type: {error_type}, "
                f"Error: {str(e)}, "
                f"Url: {url}, "
                f"Server: {self.server_url or 'stdio'}, "
                f"Transport: {self.transport_type}"
            )

            # Check if it's a stream/connection error
            if "BrokenResourceError" in error_type or "Broken" in error_type:
                verbose_logger.error(
                    "MCP client detected broken connection/stream during read_resource - "
                    "the MCP server may have crashed, disconnected, or timed out."
                )

            raise
