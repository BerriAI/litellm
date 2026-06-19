"""v2 MCP egress transport (the chokepoint): scaffolding.

This phase makes v2 own the upstream MCP connection. When ``LITELLM_USE_V2_MCP_EGRESS`` is enabled,
a v2 manager (built in later steps) implements the handler-facing egress surface via an
``UpstreamConnection`` that attaches ``resolve()``'s ``httpx.Auth`` plus resolved static/env-var
headers directly to the SDK client, replacing v1's ``_create_mcp_client`` and the
``resolve_mcp_auth`` header graft.

Step 1 lands only the flag and the egress contract (``MCPEgressManager``). The implementation
(``UpstreamConnection``, the static-headers resolver, the per-user token bridges,
``MCPServerManagerV2``) arrives in later steps; the contract grows with the surface as it is
implemented (call_tool/dispatch and the reused registry/RBAC lookups are added when the v2 manager
is assembled). The CLI flag wiring lands at the cutover step.
"""

from __future__ import annotations

import contextlib
import os
from typing import (
    TYPE_CHECKING,
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    TypeVar,
    Union,
)

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from pydantic import BaseModel, ConfigDict

from litellm.llms.custom_httpx.http_handler import get_ssl_configuration
from litellm.proxy._types import MCPTransport
from litellm.proxy.gateway.mcp.result import Error, Ok, Result

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from anyio.streams.memory import (
        MemoryObjectReceiveStream,
        MemoryObjectSendStream,
    )
    from mcp import ReadResourceResult, Resource
    from mcp.shared.message import SessionMessage
    from mcp.types import CallToolResult, GetPromptResult, Prompt
    from mcp.types import Tool as MCPTool
    from pydantic import AnyUrl

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    _Streams = tuple[
        MemoryObjectReceiveStream[SessionMessage | Exception],
        MemoryObjectSendStream[SessionMessage],
    ]

_T = TypeVar("_T")

_V2_EGRESS_ENV_FLAG = "LITELLM_USE_V2_MCP_EGRESS"


def v2_egress_enabled() -> bool:
    """True when v2 owns the MCP egress transport (set via the env flag)."""
    return os.getenv(_V2_EGRESS_ENV_FLAG, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


class MCPEgressManager(Protocol):
    """The per-server egress operations the inbound handler invokes on the manager.

    v1's ``MCPServerManager`` satisfies this today; ``MCPServerManagerV2`` (later steps) will
    implement it via the ``UpstreamConnection``. Registry/RBAC lookups
    (``get_allowed_mcp_servers``, ``get_registry``, ...) are reused from v1 and are intentionally
    not part of this egress contract; ``call_tool``/dispatch is added when the v2 manager is
    assembled.
    """

    async def _get_tools_from_server(
        self,
        server: MCPServer,
        mcp_auth_header: Optional[Union[str, Dict[str, str]]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        add_prefix: bool = True,
        raw_headers: Optional[Dict[str, str]] = None,
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ) -> List[MCPTool]: ...

    async def get_prompts_from_server(
        self,
        server: MCPServer,
        mcp_auth_header: Optional[Union[str, Dict[str, str]]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        add_prefix: bool = True,
        raw_headers: Optional[Dict[str, str]] = None,
    ) -> List[Prompt]: ...

    async def get_resources_from_server(
        self,
        server: MCPServer,
        mcp_auth_header: Optional[Union[str, Dict[str, str]]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        add_prefix: bool = True,
        raw_headers: Optional[Dict[str, str]] = None,
    ) -> List[Resource]: ...

    async def read_resource_from_server(
        self,
        server: MCPServer,
        url: AnyUrl,
        mcp_auth_header: Optional[Union[str, Dict[str, str]]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        raw_headers: Optional[Dict[str, str]] = None,
    ) -> ReadResourceResult: ...

    async def get_prompt_from_server(
        self,
        server: MCPServer,
        prompt_name: str,
        arguments: Optional[Dict[str, object]] = None,
        mcp_auth_header: Optional[Union[str, Dict[str, str]]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        raw_headers: Optional[Dict[str, str]] = None,
    ) -> GetPromptResult: ...


class ConnError(BaseModel):
    """A connection/transport failure to an upstream MCP server, modeled as a value.

    Discriminated on ``tag``: an upstream 401/403 is surfaced to the client to trigger the
    upstream OAuth flow; any other transport failure maps to a 503.
    """

    model_config = ConfigDict(frozen=True)
    tag: Literal["unauthorized", "upstream_unavailable"]
    summary: str

    @classmethod
    def of_unauthorized(cls, summary: str) -> "ConnError":
        return cls(tag="unauthorized", summary=summary)

    @classmethod
    def of_upstream_unavailable(cls, summary: str) -> "ConnError":
        return cls(tag="upstream_unavailable", summary=summary)


def _classify_conn_error(error: Exception) -> ConnError:
    # Reuse v1's exception-tree walk (it handles the SDK's anyio ExceptionGroup and the
    # __cause__/__context__ chains portably) to spot an upstream 401/403; anything else is a
    # transport failure. Imported lazily to avoid an import cycle at the manager swap.
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        extract_upstream_auth_failure,
    )

    auth_failure = extract_upstream_auth_failure(error)
    if auth_failure is not None:
        status_code, _ = auth_failure
        return ConnError.of_unauthorized(f"upstream returned {status_code}")
    return ConnError.of_upstream_unavailable(f"upstream connection failed: {error}")


class UpstreamConnection:
    """Opens the SDK client connection to one upstream MCP server and runs an operation.

    The v2 egress transport: attaches ``resolve()``'s ``httpx.Auth`` (and any static/env-var
    headers) to the connection through litellm's httpx client (SSL/proxy config), opens a
    ``ClientSession`` per request over the server's transport (streamable-http, sse, or stdio),
    and returns typed results (errors-as-values). Replaces v1's ``MCPClient`` for the modes routed
    through the v2 manager.
    """

    def __init__(
        self,
        server_url: Optional[str] = None,
        *,
        transport: MCPTransport = MCPTransport.http,
        auth: Optional[httpx.Auth] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        timeout: float = 60.0,
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        self._server_url = server_url
        self._transport = transport
        self._auth = auth
        self._extra_headers = extra_headers
        self._timeout = timeout
        self._command = command
        self._args = args
        self._env = env

    def _http_client_factory(self) -> Callable[..., httpx.AsyncClient]:
        def factory(
            *,
            headers: Optional[Dict[str, str]] = None,
            timeout: Optional[httpx.Timeout] = None,
            auth: Optional[httpx.Auth] = None,
        ) -> httpx.AsyncClient:
            return httpx.AsyncClient(
                headers=headers,
                timeout=timeout,
                auth=auth,
                verify=get_ssl_configuration(None),
                follow_redirects=True,
            )

        return factory

    @contextlib.asynccontextmanager
    async def _session_streams(self) -> AsyncGenerator[_Streams, None]:
        # The resolved auth and static/env-var headers ride on the httpx client (SDK 1.26: not the
        # transport kwargs); SSL/proxy config comes from litellm's get_ssl_configuration. Normalizes
        # all three transports to a (read, write) stream pair (streamable-http yields a third value).
        if self._transport == MCPTransport.stdio:
            params = StdioServerParameters(
                command=self._command or "", args=self._args or [], env=self._env
            )
            async with stdio_client(params) as (read_stream, write_stream, *_):
                yield read_stream, write_stream
            return
        if self._transport == MCPTransport.sse:
            async with sse_client(
                url=self._server_url or "",
                timeout=self._timeout,
                headers=self._extra_headers,
                auth=self._auth,
                httpx_client_factory=self._http_client_factory(),
            ) as (read_stream, write_stream, *_):
                yield read_stream, write_stream
            return
        http_client = httpx.AsyncClient(
            headers=self._extra_headers,
            timeout=httpx.Timeout(self._timeout),
            auth=self._auth,
            verify=get_ssl_configuration(None),
            follow_redirects=True,
        )
        try:
            async with streamable_http_client(
                url=self._server_url or "", http_client=http_client
            ) as (read_stream, write_stream, *_):
                yield read_stream, write_stream
        finally:
            with contextlib.suppress(Exception):
                await http_client.aclose()

    async def _run(
        self, operation: Callable[[ClientSession], Awaitable[_T]]
    ) -> Result[_T, ConnError]:
        try:
            async with self._session_streams() as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await operation(session)
            return Ok(result)
        except Exception as e:  # transport / protocol failures -> ConnError
            return Error(_classify_conn_error(e))

    async def list_tools(self) -> Result[List[MCPTool], ConnError]:
        async def op(session: ClientSession) -> List[MCPTool]:
            return (await session.list_tools()).tools

        return await self._run(op)

    async def call_tool(
        self, name: str, arguments: Dict[str, object]
    ) -> Result[CallToolResult, ConnError]:
        async def op(session: ClientSession) -> CallToolResult:
            return await session.call_tool(name, arguments)

        return await self._run(op)

    async def list_prompts(self) -> Result[List[Prompt], ConnError]:
        async def op(session: ClientSession) -> List[Prompt]:
            return (await session.list_prompts()).prompts

        return await self._run(op)

    async def get_prompt(
        self, name: str, arguments: Optional[Dict[str, str]] = None
    ) -> Result[GetPromptResult, ConnError]:
        async def op(session: ClientSession) -> GetPromptResult:
            return await session.get_prompt(name, arguments)

        return await self._run(op)

    async def list_resources(self) -> Result[List[Resource], ConnError]:
        async def op(session: ClientSession) -> List[Resource]:
            return (await session.list_resources()).resources

        return await self._run(op)

    async def read_resource(self, uri: AnyUrl) -> Result[ReadResourceResult, ConnError]:
        async def op(session: ClientSession) -> ReadResourceResult:
            return await session.read_resource(uri)

        return await self._run(op)
