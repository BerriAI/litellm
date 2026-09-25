"""
LiteLLM Proxy uses this MCP Client to connnect to other MCP servers.
"""

import asyncio
import base64
import hashlib
import json
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Generator, Sequence
from contextlib import AbstractAsyncContextManager
from functools import partial
from types import MappingProxyType
from typing import Final, TypeAlias, TypeVar, cast

import anyio
import httpx2
from httpx2._client import UseClientDefault
from httpx2._types import AuthTypes
from mcp import ClientSession, MCPError, ReadResourceResult, Resource, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._stream_protocols import ReadStream, WriteStream
from mcp.shared.message import SessionMessage

_TransportStreams: TypeAlias = tuple[
    ReadStream[SessionMessage | Exception],
    WriteStream[SessionMessage],
]
_TransportContext: TypeAlias = AbstractAsyncContextManager[_TransportStreams]


from mcp.types import (
    METHOD_NOT_FOUND,
    REQUEST_TIMEOUT,
    GetPromptRequestParams,
    GetPromptResult,
    InputRequiredResult,
    ListPromptsResult,
    ListResourcesResult,
    ListResourceTemplatesResult,
    PaginatedRequestParams,
    PaginatedResult,
    Prompt,
    ResourceTemplate,
    ServerNotification,
)
from mcp.types import CallToolRequestParams as MCPCallToolRequestParams
from mcp.types import CallToolResult as MCPCallToolResult
from mcp.types import Tool as MCPTool
from pydantic import AnyUrl

from litellm._logging import verbose_logger
from litellm.constants import (
    MCP_CLIENT_TIMEOUT,
    MCP_NPM_CACHE_DIR,
    MCP_TOOL_LISTING_MAX_PAGES,
    MCP_TOOL_LISTING_TIMEOUT,
)
from litellm.experimental_mcp_client.tools import list_tools_with_pagination
from litellm.llms.custom_httpx.http_handler import get_ssl_configuration
from litellm.proxy._experimental.mcp_server.mcp_debug import capture_upstream_error_response
from litellm.proxy._experimental.mcp_server.result_conversion import error_text_result
from litellm.types.llms.custom_http import VerifyTypes
from litellm.types.mcp import (
    MCPAuth,
    MCPAuthType,
    MCPStdioConfig,
    MCPTransport,
    MCPTransportType,
    credential_redirect_hook,
    has_header,
    without_header,
)


def to_basic_auth(auth_value: str) -> str:
    """Convert auth value to Basic Auth format."""
    return base64.b64encode(auth_value.encode("utf-8")).decode()


def strip_auth_scheme(auth_value: str, scheme: str) -> str:
    """Return ``auth_value`` with a leading ``<scheme>`` and separator removed, or unchanged when absent.

    Callers supply both a bare credential and a complete header value, so prefixing
    unconditionally yields ``Bearer Bearer <jwt>``. Scheme names are case-insensitive per
    RFC 7235. A credential is required after the scheme, so both a token that merely begins
    with the scheme text and a scheme with nothing behind it are returned untouched.
    Surrounding whitespace is left to ``_strip_header_whitespace`` at header-build time.
    """
    parts: Final = auth_value.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == scheme.lower():
        return parts[1]
    return auth_value


def to_basic_credentials(auth_value: str) -> str:
    """Return the base64 credentials for a ``Basic`` header, encoding only when needed.

    ``Basic <credentials>`` carries credentials that are already encoded, so encoding the whole
    value again would bury the scheme inside the payload. This has to run before
    :func:`to_basic_auth` rather than at header-build time, where no prefix is left to find.
    A schemed value whose remainder does not decode is the bare ``username:password`` shape with
    the scheme written in front of it, and is encoded rather than forwarded as an invalid header;
    a pair always contains ``:``, which is outside the base64 alphabet, so the two never collide.
    """
    credentials: Final = strip_auth_scheme(auth_value, "Basic")
    if credentials == auth_value:
        return to_basic_auth(auth_value)
    try:
        base64.b64decode(credentials, validate=True)
    except ValueError:
        return to_basic_auth(credentials)
    return credentials


def _strip_header_whitespace(headers: dict[str, str]) -> dict[str, str]:
    return {
        (key.strip() if isinstance(key, str) else key): (value.strip() if isinstance(value, str) else value)
        for key, value in headers.items()
    }


def _first_non_cancelled_cause(exc: BaseException, cleanup_errors: tuple[Exception, ...] = ()) -> BaseException | None:
    queue: Final[list[BaseException]] = [exc]
    while queue:
        current = queue.pop(0)
        nested = getattr(current, "exceptions", None)
        if nested:
            queue.extend(nested)
        elif not isinstance(current, asyncio.CancelledError) and not any(current is error for error in cleanup_errors):
            return current
    return None


_SDK_READ_TIMEOUT_CODE: Final = REQUEST_TIMEOUT
"""The code the MCP SDK puts on its own elapsed read timeout."""


def as_mcp_read_timeout(exc: BaseException) -> TimeoutError | None:
    """Normalize an MCP SDK read timeout for client and gateway diagnostics, or return ``None``.

    The SDK reports its own elapsed read timeout as ``MCPError`` carrying ``REQUEST_TIMEOUT`` in a
    field that also carries relayed upstream JSON-RPC errors. The numeric code alone therefore
    cannot separate the two, and an upstream answering with the same application code would be
    reported as a gateway timeout it never caused. The SDK raises its own from inside an ``except TimeoutError``, so the elapsed timeout is
    on the context chain, while a relayed error is built from a received message and has no such
    chain; that is the discriminator.
    """
    if not isinstance(exc, MCPError) or exc.error.code != _SDK_READ_TIMEOUT_CODE:
        return None
    if not isinstance(exc.__context__, TimeoutError):
        return None
    return TimeoutError(exc.error.message)


TSessionResult = TypeVar("TSessionResult")
_ListPage = TypeVar("_ListPage", bound=PaginatedResult)
_ListItem = TypeVar("_ListItem")


async def _run_bounded_cleanup(operation: Callable[[], Awaitable[TSessionResult]], deadline: float) -> TSessionResult:
    async def run() -> TSessionResult:
        with anyio.fail_after(max(0, deadline - anyio.current_time()), shield=True):
            return await operation()

    # A cancelled asyncio.gather repeatedly forwards Task.cancel, bypassing AnyIO shields.
    # Isolate only cleanup, and drain it before propagating the caller's cancellation.
    task: Final = asyncio.create_task(run())
    interrupted: asyncio.CancelledError | None = None
    with anyio.CancelScope(shield=True):
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as exc:
                interrupted = exc
            except Exception:
                break
        if interrupted is not None:
            if not task.cancelled():
                task.exception()
            raise interrupted
        return task.result()


class _MCPResponseStream(httpx2.AsyncByteStream):
    def __init__(self, stream: httpx2.AsyncByteStream, record_error: Callable[[Exception], None]) -> None:
        self._stream: Final = stream
        self._record_error: Final = record_error

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self._stream:
                yield chunk
        except Exception as error:
            self._record_error(error)
            raise

    async def aclose(self) -> None:
        try:
            await self._stream.aclose()
        except Exception as error:
            self._record_error(error)
            raise


class _MCPHTTPClient(httpx2.AsyncClient):
    cleanup_scope: anyio.CancelScope | None = None
    cleanup_errors: tuple[Exception, ...] = ()

    def _record_cleanup_error(self, error: Exception) -> None:
        if self.cleanup_scope is not None and self.cleanup_scope.shield:
            self.cleanup_errors += (error,)

    async def send(
        self,
        request: httpx2.Request,
        *,
        stream: bool = False,
        auth: AuthTypes | UseClientDefault | None = httpx2.USE_CLIENT_DEFAULT,
        follow_redirects: bool | UseClientDefault = httpx2.USE_CLIENT_DEFAULT,
    ) -> httpx2.Response:
        if request.method == "DELETE" and self.cleanup_scope is not None:

            async def terminate() -> httpx2.Response:
                termination: Final = await super(_MCPHTTPClient, self).send(
                    request, stream=stream, auth=auth, follow_redirects=follow_redirects
                )
                await termination.aread()
                return termination

            return await _run_bounded_cleanup(terminate, self.cleanup_scope.deadline)
        try:
            response: Final = await super().send(request, stream=stream, auth=auth, follow_redirects=follow_redirects)
            if request.method == "POST" and response.is_error and response.status_code != 404:
                await response.aclose()
                response.raise_for_status()
            if stream:
                response.stream = _MCPResponseStream(
                    cast(httpx2.AsyncByteStream, response.stream), self._record_cleanup_error
                )
            return response
        except Exception as error:
            self._record_cleanup_error(error)
            raise


class MCPSigV4Auth(httpx2.Auth):
    """
    httpx2 Auth class that signs each request with AWS SigV4.
    This is used for MCP servers that require AWS SigV4 authentication,
    such as AWS Bedrock AgentCore MCP servers. httpx calls auth_flow()
    for every outgoing request, enabling per-request signature computation.
    """

    requires_request_body = True

    def __init__(
        self,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        aws_region_name: str | None = None,
        aws_service_name: str | None = None,
        aws_role_name: str | None = None,
        aws_session_name: str | None = None,
    ):
        try:
            from botocore.credentials import Credentials
        except ImportError:
            raise ImportError("Missing botocore to use AWS SigV4 authentication. Run 'pip install boto3'.")
        self.service_name = aws_service_name or "bedrock-agentcore"
        self.region_name = aws_region_name or "us-east-1"
        # Note: os.environ/ prefixed values are already resolved by
        # ProxyConfig._check_for_os_environ_vars() at config load time.
        # Values arrive here as plain strings.
        if aws_role_name:
            self.credentials = self._assume_role(
                aws_role_name=aws_role_name,
                aws_session_name=aws_session_name,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                aws_region_name=self.region_name,
            )
        elif aws_access_key_id and aws_secret_access_key:
            self.credentials = Credentials(
                access_key=aws_access_key_id,
                secret_key=aws_secret_access_key,
                token=aws_session_token,
            )
        else:
            # Fall back to default boto3 credential chain
            import botocore.session

            session: Final = botocore.session.get_session()
            self.credentials = session.get_credentials()
            if self.credentials is None:
                raise ValueError(
                    "No AWS credentials found. Provide aws_access_key_id and "
                    "aws_secret_access_key, or configure default credentials "
                    "(env vars, ~/.aws/credentials, instance profile)."
                )

    @staticmethod
    def _assume_role(
        aws_role_name: str,
        aws_session_name: str | None,
        aws_access_key_id: str | None,
        aws_secret_access_key: str | None,
        aws_session_token: str | None,
        aws_region_name: str,
    ):
        """Call STS AssumeRole and return temporary credentials."""
        import time

        import boto3
        from botocore.credentials import Credentials

        session_name: Final = aws_session_name or f"litellm-mcp-{int(time.time())}"
        sts_kwargs: Final[dict] = {"region_name": aws_region_name}
        if aws_access_key_id and aws_secret_access_key:
            sts_kwargs["aws_access_key_id"] = aws_access_key_id
            sts_kwargs["aws_secret_access_key"] = aws_secret_access_key
            if aws_session_token:
                sts_kwargs["aws_session_token"] = aws_session_token
        sts_client: Final = boto3.client("sts", **sts_kwargs)
        sts_response: Final = sts_client.assume_role(
            RoleArn=aws_role_name,
            RoleSessionName=session_name,
        )
        sts_creds: Final = sts_response["Credentials"]
        return Credentials(
            access_key=sts_creds["AccessKeyId"],
            secret_key=sts_creds["SecretAccessKey"],
            token=sts_creds["SessionToken"],
        )

    def auth_flow(self, request: httpx2.Request) -> Generator[httpx2.Request, httpx2.Response, None]:
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest

        # Build AWSRequest from the httpx Request.
        # Pass all request headers so the canonical SigV4 signature covers them.
        aws_request: Final = AWSRequest(
            method=request.method,
            url=str(request.url),
            data=request.content,
            headers=dict(request.headers),
        )
        # Sign the request — SigV4Auth.add_auth() adds Authorization,
        # X-Amz-Date, and X-Amz-Security-Token (if session token present).
        # Host header is derived automatically from the URL.
        sigv4: Final = SigV4Auth(self.credentials, self.service_name, self.region_name)
        sigv4.add_auth(aws_request)
        # Copy SigV4 headers back to the httpx request
        for header_name, header_value in aws_request.headers.items():
            request.headers[header_name] = header_value
        yield request


class MCPClient:
    """
    MCP Client supporting:
      SSE and HTTP transports
      Authentication via Bearer token, Basic Auth, or API Key
      Tool calling with error handling and result parsing
      Sampling callbacks for upstream server LLM requests
      Elicitation callbacks for upstream server user-input requests
    """

    def __init__(
        self,
        server_url: str = "",
        transport_type: MCPTransportType = MCPTransport.http,
        auth_type: MCPAuthType = None,
        auth_value: str | dict[str, str] | None = None,
        auth_header_name: str | None = None,
        timeout: float | None = None,
        stdio_config: MCPStdioConfig | None = None,
        extra_headers: dict[str, str] | None = None,
        ssl_verify: VerifyTypes | None = None,
        aws_auth: httpx2.Auth | None = None,
        resolved_auth: httpx2.Auth | None = None,
        sampling_callback: Callable | None = None,
        elicitation_callback: Callable | None = None,
        logging_callback: Callable | None = None,
    ):
        self.server_url: str = server_url
        self.transport_type: MCPTransport = transport_type
        self.auth_type: MCPAuthType = auth_type
        self.timeout: float = timeout if timeout is not None else MCP_CLIENT_TIMEOUT
        self._mcp_auth_value: str | dict[str, str] | None = None
        # The one place this client decides which header its credential occupies: the operator's
        # configured slot on the v1 path, or the slot the v2 resolver's auth object already owns.
        # Every consumer reads this rather than re-deriving it, since each re-derivation so far
        # picked up a different bug.
        self._credential_slot: str | None = auth_header_name or getattr(resolved_auth, "header_name", None)
        self.stdio_config: MCPStdioConfig | None = stdio_config
        self.extra_headers: dict[str, str] | None = extra_headers
        self.ssl_verify: VerifyTypes | None = ssl_verify
        self._aws_auth: httpx2.Auth | None = aws_auth
        # A pre-resolved httpx2.Auth (e.g. from the v2 credential resolver) attached to the
        # upstream client's auth= slot, taking precedence over the SigV4 aws_auth.
        self._resolved_auth: httpx2.Auth | None = resolved_auth
        self._last_initialize_instructions: str | None = None
        self._sampling_callback: Callable | None = sampling_callback
        self._elicitation_callback: Callable | None = elicitation_callback
        self._logging_callback: Callable | None = logging_callback
        # handle the basic auth value if provided
        if auth_value:
            self.update_auth_value(auth_value)

    async def discovery_auth_fingerprint(self) -> str:
        return self._hash_discovery_auth(await self.prepare_request_auth())

    async def prepare_request_auth(self) -> httpx2.Request:
        """Preview the authenticated request without sending it, closing the auth flow afterwards."""
        request: Final = httpx2.Request(
            "POST", self.server_url or "http://localhost/", headers=self._get_auth_headers()
        )
        if self._resolved_auth is None:
            return request
        flow: Final = self._resolved_auth.async_auth_flow(request)
        try:
            authenticated: Final = await flow.__anext__()
            return authenticated
        finally:
            await flow.aclose()

    @staticmethod
    def _hash_discovery_auth(request: httpx2.Request) -> str:
        material: Final = json.dumps((str(request.url), tuple(sorted(request.headers.multi_items()))))
        return hashlib.sha256(material.encode()).hexdigest()

    def _create_transport_context(
        self,
    ) -> tuple[_TransportContext, httpx2.AsyncClient | None]:
        """
        Create the appropriate transport context based on transport type.
        Returns:
            Tuple of (transport_context, http_client).
            http_client is only set for HTTP transport and needs cleanup.
        """
        http_client: httpx2.AsyncClient | None = None
        if self.transport_type == MCPTransport.stdio:
            if not self.stdio_config:
                raise ValueError("stdio_config is required for stdio transport")
            server_params: Final = StdioServerParameters(
                command=self.stdio_config.get("command", ""),
                args=self.stdio_config.get("args", []),
                env=self._get_safe_stdio_env(self.stdio_config.get("env")),
            )
            return stdio_client(server_params), None
        if self.transport_type == MCPTransport.sse:
            headers = self._get_auth_headers()
            httpx_client_factory = self._create_httpx_client_factory()
            return (
                sse_client(
                    url=self.server_url,
                    timeout=self.timeout,
                    headers=headers,
                    httpx_client_factory=httpx_client_factory,
                ),
                None,
            )
        # HTTP transport (default)
        headers = self._get_auth_headers()
        httpx_client_factory = self._create_httpx_client_factory()
        verbose_logger.debug("litellm headers for streamable_http_client: %s", headers)
        http_client = httpx_client_factory(
            headers=headers,
            timeout=httpx2.Timeout(self.timeout),
        )
        transport_ctx: Final = streamable_http_client(
            url=self.server_url,
            http_client=http_client,
        )
        return transport_ctx, http_client

    def _get_safe_stdio_env(self, provided_env: dict[str, str] | None) -> dict[str, str] | None:
        """
        Return a safe environment for the stdio subprocess.

        If provided_env is set, we use it as-is.
        If provided_env is None, we return a minimal allowlist from the parent environment
        to avoid leaking sensitive LiteLLM keys (OPENAI_API_KEY, etc.) to sub-processes.
        """
        if provided_env is not None:
            return provided_env

        # Minimal allowlist of safe/standard environment variables
        safe_keys: Final = {
            "PATH",
            "HOME",
            "USER",
            "LOGNAME",
            "TMPDIR",
            "TMP",
            "TEMP",
            "SHELL",
            "LANG",
            "LC_ALL",
            # Node/Package manager caches
            "NPM_CONFIG_CACHE",
            "PNPM_HOME",
            "XDG_CACHE_HOME",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
            # System info
            "SYSTEMROOT",
            "COMSPEC",
            "PATHEXT",
            "WINDIR",
        }

        safe_env: Final = {}
        for key in safe_keys:
            if key in os.environ:
                safe_env[key] = os.environ[key]

        if "NPM_CONFIG_CACHE" not in safe_env:
            safe_env["NPM_CONFIG_CACHE"] = MCP_NPM_CACHE_DIR

        return safe_env

    async def _execute_session_operation(
        self,
        transport_ctx: _TransportContext,
        operation: Callable[[ClientSession], Awaitable[TSessionResult]],
        http_client: httpx2.AsyncClient | None = None,
    ) -> TSessionResult:
        """
        Execute an operation within a transport and session context.
        Handles entering/exiting contexts and running the operation.
        Passes sampling/elicitation/logging callbacks to the ClientSession
        so that upstream MCP servers can request LLM inference (sampling),
        user input (elicitation), or send log messages.
        """
        in_flight_error: BaseException | None = None
        with anyio.CancelScope() as cleanup_scope:
            if isinstance(http_client, _MCPHTTPClient):
                http_client.cleanup_scope = cleanup_scope
            try:
                transport: Final = await transport_ctx.__aenter__()
                try:
                    read_stream: Final = transport[0]
                    write_stream: Final = transport[1]
                    stream_error: Final[asyncio.Future[Exception]] = asyncio.get_running_loop().create_future()

                    async def receive_message(
                        message: ServerNotification | Exception,
                    ) -> None:
                        if not isinstance(message, (ValueError, httpx2.HTTPError, OSError)):
                            return
                        if not stream_error.done():
                            stream_error.set_result(message)
                        # The SDK closes pending requests when its message handler raises.
                        raise RuntimeError("MCP response stream failed")

                    session_kwargs: Final = {
                        name: callback
                        for name, callback in (
                            ("sampling_callback", self._sampling_callback),
                            ("elicitation_callback", self._elicitation_callback),
                            ("logging_callback", self._logging_callback),
                        )
                        if callback is not None
                    }
                    # The SDK drops a response stream that ends without a JSON-RPC reply, so nothing else
                    # ever fails the request.
                    session_ctx: Final = ClientSession(
                        read_stream,
                        write_stream,
                        read_timeout_seconds=self.timeout,
                        message_handler=receive_message,
                        **session_kwargs,
                    )
                    session: Final = await session_ctx.__aenter__()
                    try:
                        init_result: Final = await session.initialize()
                        instructions: Final = getattr(init_result, "instructions", None)
                        self._last_initialize_instructions = (
                            instructions.strip() or None if isinstance(instructions, str) else None
                        )
                        result: Final = await operation(session)
                    except BaseException as operation_error:
                        in_flight_error = operation_error
                        if isinstance(operation_error, MCPError) and stream_error.done():
                            raise stream_error.result()
                        raise
                    finally:
                        cleanup_scope.shield = True
                        cleanup_scope.deadline = anyio.current_time() + 5
                        try:
                            await session_ctx.__aexit__(None, None, None)
                        except (Exception, asyncio.CancelledError) as e:
                            verbose_logger.debug("Error during session context exit: %s", e)
                            if in_flight_error is None and isinstance(e, asyncio.CancelledError):
                                raise
                except BaseException as e:
                    in_flight_error = e
                    raise
                finally:
                    cleanup_scope.shield = True
                    cleanup_scope.deadline = min(cleanup_scope.deadline, anyio.current_time() + 5)
                    try:
                        await transport_ctx.__aexit__(None, None, None)
                    except (Exception, asyncio.CancelledError) as exit_error:
                        verbose_logger.debug("Error during transport context exit: %s", exit_error)
                        if in_flight_error is None and isinstance(exit_error, asyncio.CancelledError):
                            raise
                        root_cause: Final = _first_non_cancelled_cause(
                            exit_error, http_client.cleanup_errors if isinstance(http_client, _MCPHTTPClient) else ()
                        )
                        if root_cause is not None and isinstance(in_flight_error, asyncio.CancelledError):
                            raise root_cause from in_flight_error
            finally:
                cleanup_scope.shield = False
                if isinstance(http_client, _MCPHTTPClient):
                    http_client.cleanup_errors = ()
                    http_client.cleanup_scope = None
        await anyio.lowlevel.checkpoint_if_cancelled()
        if cleanup_scope.cancel_called:
            raise (
                in_flight_error
                if in_flight_error is not None
                else asyncio.CancelledError("MCP session cleanup timed out")
            )
        return result

    async def run_with_session(
        self,
        operation: Callable[[ClientSession], Awaitable[TSessionResult]],
        *,
        quiet_on_error: bool = False,
    ) -> TSessionResult:
        """Open a session, run the provided coroutine, and clean up.

        quiet_on_error demotes the failure line to debug for callers that own the exception
        (call_tool / list_tools under raise_on_error), so an expected pass-through re-auth does
        not emit a warning per call; every other caller keeps the operator-visible warning."""
        http_client: httpx2.AsyncClient | None = None
        close_cancellation: asyncio.CancelledError | None = None
        try:
            self._last_initialize_instructions = None
            transport_ctx, http_client = self._create_transport_context()
            result: Final = await self._execute_session_operation(transport_ctx, operation, http_client=http_client)
        except Exception as e:
            read_timeout: Final = as_mcp_read_timeout(e)
            if read_timeout is not None:
                verbose_logger.warning(
                    "MCP client timed out after %ss waiting for a valid MCP response from %s",
                    self.timeout,
                    self.server_url or "stdio",
                )
                raise read_timeout from e
            _log: Final = verbose_logger.debug if quiet_on_error else verbose_logger.warning
            _log("MCP client run_with_session failed for %s", self.server_url or "stdio")
            raise
        finally:
            if http_client is not None:
                try:
                    await _run_bounded_cleanup(http_client.aclose, anyio.current_time() + 1)
                except (Exception, asyncio.CancelledError) as e:
                    verbose_logger.debug("Error during http_client cleanup: %s", e)
                    if isinstance(e, asyncio.CancelledError):
                        close_cancellation = e

        if close_cancellation is not None:
            raise close_cancellation
        await anyio.lowlevel.checkpoint_if_cancelled()
        return result

    def update_auth_value(self, mcp_auth_value: str | dict[str, str]) -> None:
        """
        Set the authentication header for the MCP client.
        """
        if isinstance(mcp_auth_value, dict):
            self._mcp_auth_value = mcp_auth_value
        elif self.auth_type == MCPAuth.basic:
            self._mcp_auth_value = to_basic_credentials(mcp_auth_value)
        else:
            self._mcp_auth_value = mcp_auth_value

    def _header_slot(self, default: str) -> str:
        return self._credential_slot or default

    def _get_auth_headers(self) -> dict:
        """Generate authentication headers based on auth type."""
        headers: Final = {}
        if self._mcp_auth_value:
            if isinstance(self._mcp_auth_value, str):
                if self.auth_type == MCPAuth.bearer_token:
                    static_bearer: Final = strip_auth_scheme(self._mcp_auth_value, "Bearer")
                    headers[self._header_slot("Authorization")] = f"Bearer {static_bearer}"
                elif self.auth_type == MCPAuth.basic:
                    headers[self._header_slot("Authorization")] = f"Basic {self._mcp_auth_value}"
                elif self.auth_type == MCPAuth.api_key:
                    headers[self._header_slot("X-API-Key")] = self._mcp_auth_value
                elif self.auth_type == MCPAuth.authorization:
                    # This auth type means the caller owns the whole header value.
                    headers[self._header_slot("Authorization")] = self._mcp_auth_value
                elif self.auth_type == MCPAuth.oauth2:
                    oauth2_bearer: Final = strip_auth_scheme(self._mcp_auth_value, "Bearer")
                    headers[self._header_slot("Authorization")] = f"Bearer {oauth2_bearer}"
                elif self.auth_type == MCPAuth.token:
                    scheme_token: Final = strip_auth_scheme(self._mcp_auth_value, "token")
                    headers[self._header_slot("Authorization")] = f"token {scheme_token}"
                elif self.auth_type == MCPAuth.oauth2_token_exchange:
                    exchanged_bearer: Final = strip_auth_scheme(self._mcp_auth_value, "Bearer")
                    headers[self._header_slot("Authorization")] = f"Bearer {exchanged_bearer}"
            elif isinstance(self._mcp_auth_value, dict):
                headers.update(self._mcp_auth_value)
        # Note: aws_sigv4 auth is not handled here — SigV4 requires per-request
        # signing (including the body hash), so it uses httpx2.Auth flow instead
        # of static headers. See MCPSigV4Auth and _create_httpx_client_factory().
        # update the headers with the extra headers
        if self.extra_headers:
            # Mirrors _resolve_v2_auth: when the operator named a slot for the credential the
            # gateway resolved, no injected header may shadow it, case-insensitively, since HTTP
            # header names are. Without a configured slot the old precedence stands unchanged.
            slot: Final = self._credential_slot
            injected: Final = (
                without_header(self.extra_headers, slot) if slot and has_header(headers, slot) else self.extra_headers
            )
            headers.update(injected or {})
        return _strip_header_whitespace(headers)

    def _create_httpx_client_factory(
        self, *, transport: httpx2.AsyncBaseTransport | None = None
    ) -> Callable[..., httpx2.AsyncClient]:
        """
        Create a custom httpx2 client factory that uses LiteLLM's SSL configuration.
        This factory follows the same CA bundle path logic as http_handler.py:
        1. Check ssl_verify parameter (can be SSLContext, bool, or path to CA bundle)
        2. Check SSL_VERIFY environment variable
        3. Check SSL_CERT_FILE environment variable
        4. Fall back to certifi CA bundle
        """

        def factory(
            *,
            headers: dict[str, str] | None = None,
            timeout: httpx2.Timeout | None = None,
            auth: httpx2.Auth | None = None,
        ) -> httpx2.AsyncClient:
            """Create an httpx2.AsyncClient with LiteLLM's SSL configuration."""
            # Get unified SSL configuration using the same logic as http_handler.py
            ssl_config: Final = get_ssl_configuration(self.ssl_verify)
            verbose_logger.debug("MCP client using SSL configuration: %s", type(ssl_config).__name__)
            # The MCP SDK's sse_client and streamable_http_client call this factory without
            # passing auth=, so the fallback is used: a v2-resolved auth if present, else the
            # SigV4 aws_auth. Both are None for the common case — no behavior change.
            fallback_auth: Final = self._resolved_auth if self._resolved_auth is not None else self._aws_auth
            effective_auth: Final = auth if auth is not None else fallback_auth
            guard: Final = credential_redirect_hook(self.server_url, self._credential_slot)
            return _MCPHTTPClient(
                transport=transport,
                headers=headers,
                timeout=timeout,
                auth=effective_auth,
                verify=ssl_config,
                follow_redirects=True,
                event_hooks=MappingProxyType(
                    {"response": [capture_upstream_error_response], "request": [guard] if guard else []}
                ),
            )

        return factory

    async def list_tools(self, raise_on_error: bool = False) -> list[MCPTool]:
        """List available tools from the server.

        Args:
            raise_on_error: When True, re-raise exceptions instead of returning
                an empty list. Used by the proxy's pass-through MCP flow so it
                can surface upstream HTTP 401 responses as a proper 401 to the
                MCP client (triggering the upstream OAuth flow) rather than
                masking them as "connected, no tools".
        """
        verbose_logger.debug("MCP client listing tools from %s", self.server_url or "stdio")

        try:
            # A per-server timeout above the global default extends the whole-walk deadline
            listing_deadline: Final = max(self.timeout, MCP_TOOL_LISTING_TIMEOUT)
            tools: Final = await self.run_with_session(
                partial(list_tools_with_pagination, listing_deadline=listing_deadline),
                quiet_on_error=raise_on_error,
            )
            tool_count: Final = len(tools)
            tool_names: Final = tuple(tool.name for tool in tools)
            verbose_logger.info(
                "MCP client listed %s tools from %s: %s", tool_count, self.server_url or "stdio", tool_names
            )
            return tools
        except asyncio.CancelledError:
            verbose_logger.warning("MCP client list_tools was cancelled")
            raise
        except Exception as e:
            error_type: Final = type(e).__name__
            # Mirror call_tool: when the caller opted into raise_on_error it owns the exception and
            # logs it at the fitting level (an expected pass-through re-auth 401 is info, not an
            # error), so log at debug here to avoid an error-level line + traceback that would trip
            # error-rate alerts on that expected signal. The swallow path still logs the full
            # exception because nothing downstream will surface the failure.
            _log: Final = verbose_logger.debug if raise_on_error else verbose_logger.exception
            _log(
                f"MCP client list_tools failed - "
                f"Error Type: {error_type}, "
                f"Error: {e}, "
                f"Server: {self.server_url or 'stdio'}, "
                f"Transport: {self.transport_type}"
            )
            # Check if it's a stream/connection error
            if "BrokenResourceError" in error_type or "Broken" in error_type:
                _log_broken: Final = verbose_logger.debug if raise_on_error else verbose_logger.error
                _log_broken(
                    "MCP client detected broken connection/stream during list_tools - "
                    "the MCP server may have crashed, disconnected, or timed out"
                )

            if raise_on_error:
                raise
            # Return empty list instead of raising to allow graceful degradation
            return []

    @staticmethod
    def error_tool_result(exc: Exception) -> MCPCallToolResult:
        """The error result ``call_tool`` returns when it swallows a failure (no re-execution)."""
        return error_text_result(exc)

    async def call_tool(
        self,
        call_tool_request_params: MCPCallToolRequestParams,
        host_progress_callback: Callable | None = None,
        raise_on_error: bool = False,
        allow_input_required: bool = False,
        on_dispatch: Callable[[], None] | None = None,
    ) -> MCPCallToolResult | InputRequiredResult:
        """
        Call an MCP Tool.

        Args:
            raise_on_error: When True, re-raise the underlying exception instead of returning an
                ``isError=True`` result. The token-exchange (OBO) tool-call path uses this to detect
                an upstream 401 so it can re-mint the exchanged token and retry once; every other
                caller keeps the default and gets graceful ``isError`` degradation.
            allow_input_required: When True, a 2026-07-28 upstream may answer with an interim
                ``InputRequiredResult`` and it is returned as is. The SDK rejects it otherwise, so
                callers only opt in when the downstream side can carry it.
            on_dispatch: Called once the session is ready, right before the tool call is sent,
                so a caller can tell a call the server may have run from one that never left.
        """
        verbose_logger.info("MCP client calling tool '%s'", call_tool_request_params.name)

        async def on_progress(progress: float, total: float | None, message: str | None):
            percentage: Final = (progress / total * 100) if total else 0
            verbose_logger.info(
                f"MCP Tool '{call_tool_request_params.name}' progress: "
                f"{progress}/{total} ({percentage:.0f}%) - {message or ''}"
            )
            # Forward to Host if callback provided
            if host_progress_callback:
                try:
                    await host_progress_callback(progress, total)
                except Exception as e:
                    verbose_logger.warning("Failed to forward to Host: %s", e)

        async def _call_tool_operation(session: ClientSession):
            verbose_logger.debug("MCP client sending tool call to session")
            if on_dispatch is not None:
                on_dispatch()
            return await session.call_tool(
                name=call_tool_request_params.name,
                arguments=call_tool_request_params.arguments,
                progress_callback=on_progress,
                allow_input_required=allow_input_required,
            )

        try:
            tool_result: Final = await self.run_with_session(_call_tool_operation, quiet_on_error=raise_on_error)
            verbose_logger.info("MCP client tool call '%s' completed successfully", call_tool_request_params.name)
            return tool_result
        except asyncio.CancelledError:
            verbose_logger.warning("MCP client tool call timed out after %ss for %s", self.timeout, self.server_url)
            raise
        except Exception as e:
            import traceback

            error_trace: Final = traceback.format_exc()
            verbose_logger.debug("MCP client tool call traceback:\n%s", error_trace)
            # Log detailed error information
            error_type: Final = type(e).__name__
            # When the caller opted into raise_on_error it owns the exception and logs it at the
            # level that fits (an expected pass-through re-auth 401 is info, not an operator-actionable
            # error), so log at debug here to avoid an error-level line that would trip error-rate
            # alerts on that expected signal. The swallow path (raise_on_error=False) still logs at
            # error because nothing downstream will surface the failure.
            _log: Final = verbose_logger.debug if raise_on_error else verbose_logger.error
            _log(
                f"MCP client call_tool failed - "
                f"Error Type: {error_type}, "
                f"Error: {e}, "
                f"Tool: {call_tool_request_params.name}, "
                f"Server: {self.server_url or 'stdio'}, "
                f"Transport: {self.transport_type}"
            )
            # Check if it's a stream/connection error
            if "BrokenResourceError" in error_type or "Broken" in error_type:
                _log(
                    "MCP client detected broken connection/stream - "
                    "the MCP server may have crashed, disconnected, or timed out."
                )
            if raise_on_error:
                raise
            # Return a default error result instead of raising
            return self.error_tool_result(e)

    async def _list_optional_pages(
        self,
        fetch_page: Callable[[PaginatedRequestParams | None], Awaitable[_ListPage]],
        items_of: Callable[[_ListPage], Sequence[_ListItem]],
    ) -> list[_ListItem]:  # mutable-ok: existing list discovery API
        items: Final[list[_ListItem]] = []  # mutable-ok: bounded iterative page accumulation
        cursors: Final[set[str]] = set()  # mutable-ok: constant-time detection of cursor cycles
        cursor: str | None = None  # rebind-ok: iterative traversal avoids recursion at the existing page cap
        with anyio.fail_after(max(self.timeout, MCP_TOOL_LISTING_TIMEOUT)):
            for page_index in range(MCP_TOOL_LISTING_MAX_PAGES):
                try:
                    page = await fetch_page(None if cursor is None else PaginatedRequestParams(cursor=cursor))
                except MCPError as error:
                    if page_index > 0 and error.error.code == METHOD_NOT_FOUND:
                        raise RuntimeError("MCP list operation became unavailable during pagination") from error
                    raise
                items.extend(items_of(page))
                if not page.next_cursor:
                    return items
                if page.next_cursor in cursors:
                    raise RuntimeError("MCP list pagination repeated a cursor")
                cursors.add(page.next_cursor)
                cursor = page.next_cursor
        raise RuntimeError(f"MCP list pagination exceeded {MCP_TOOL_LISTING_MAX_PAGES} pages")

    async def list_prompts(self, *, raise_on_error: bool = False) -> list[Prompt]:
        """List available prompts from the server."""
        verbose_logger.debug("MCP client listing tools from %s", self.server_url or "stdio")

        async def _list_prompts_operation(session: ClientSession) -> ListPromptsResult:
            capabilities: Final = session.server_capabilities
            if capabilities is not None and capabilities.prompts is None:
                return ListPromptsResult(prompts=[])
            try:
                return ListPromptsResult(
                    prompts=await self._list_optional_pages(
                        lambda params: session.list_prompts(params=params), lambda page: page.prompts
                    )
                )
            except MCPError as error:
                if error.error.code != METHOD_NOT_FOUND:
                    raise
                verbose_logger.debug(
                    "MCP client list_prompts is unsupported by %s: %s", self.server_url or "stdio", error
                )
                return ListPromptsResult(prompts=[])

        try:
            result: Final = await self.run_with_session(_list_prompts_operation)
            prompt_count: Final = len(result.prompts)
            prompt_names: Final = [prompt.name for prompt in result.prompts]
            verbose_logger.info(
                "MCP client listed %s tools from %s: %s", prompt_count, self.server_url or "stdio", prompt_names
            )
            return result.prompts
        except asyncio.CancelledError:
            verbose_logger.warning("MCP client list_prompts was cancelled")
            raise
        except Exception as e:
            if raise_on_error:
                raise
            error_type: Final = type(e).__name__
            verbose_logger.error(
                "MCP client list_prompts failed - Error Type: %s, Error: %s, Server: %s, Transport: %s",
                error_type,
                e,
                self.server_url or "stdio",
                self.transport_type,
            )
            # Check if it's a stream/connection error
            if "BrokenResourceError" in error_type or "Broken" in error_type:
                verbose_logger.error(
                    "MCP client detected broken connection/stream during list_tools - "
                    "the MCP server may have crashed, disconnected, or timed out"
                )
            # Return empty list instead of raising to allow graceful degradation
            return []

    async def get_prompt(self, get_prompt_request_params: GetPromptRequestParams) -> GetPromptResult:
        """Fetch a prompt definition from the MCP server."""
        verbose_logger.info("MCP client fetching prompt '%s'", get_prompt_request_params.name)

        async def _get_prompt_operation(session: ClientSession):
            verbose_logger.debug("MCP client sending get_prompt request to session")
            return await session.get_prompt(
                name=get_prompt_request_params.name,
                arguments=get_prompt_request_params.arguments,
            )

        try:
            get_prompt_result: Final = await self.run_with_session(_get_prompt_operation)
            verbose_logger.info("MCP client get_prompt '%s' completed successfully", get_prompt_request_params.name)
            return get_prompt_result
        except asyncio.CancelledError:
            verbose_logger.warning("MCP client get_prompt was cancelled")
            raise
        except Exception as e:
            import traceback

            error_trace: Final = traceback.format_exc()
            verbose_logger.debug("MCP client get_prompt traceback:\n%s", error_trace)
            # Log detailed error information
            error_type: Final = type(e).__name__
            verbose_logger.error(
                "MCP client get_prompt failed - Error Type: %s, Error: %s, Prompt: %s, Server: %s, Transport: %s",
                error_type,
                e,
                get_prompt_request_params.name,
                self.server_url or "stdio",
                self.transport_type,
            )
            # Check if it's a stream/connection error
            if "BrokenResourceError" in error_type or "Broken" in error_type:
                verbose_logger.error(
                    "MCP client detected broken connection/stream during get_prompt - "
                    "the MCP server may have crashed, disconnected, or timed out."
                )
            raise

    async def list_resources(self, *, raise_on_error: bool = False) -> list[Resource]:
        """List available resources from the server."""
        verbose_logger.debug("MCP client listing resources from %s", self.server_url or "stdio")

        async def _list_resources_operation(session: ClientSession) -> ListResourcesResult:
            capabilities: Final = session.server_capabilities
            if capabilities is not None and capabilities.resources is None:
                return ListResourcesResult(resources=[])
            try:
                return ListResourcesResult(
                    resources=await self._list_optional_pages(
                        lambda params: session.list_resources(params=params), lambda page: page.resources
                    )
                )
            except MCPError as error:
                if error.error.code != METHOD_NOT_FOUND:
                    raise
                verbose_logger.debug(
                    "MCP client list_resources is unsupported by %s: %s", self.server_url or "stdio", error
                )
                return ListResourcesResult(resources=[])

        try:
            result: Final = await self.run_with_session(_list_resources_operation)
            resource_count: Final = len(result.resources)
            resource_names: Final = [resource.name for resource in result.resources]
            verbose_logger.info(
                "MCP client listed %s resources from %s: %s", resource_count, self.server_url or "stdio", resource_names
            )
            return result.resources
        except asyncio.CancelledError:
            verbose_logger.warning("MCP client list_resources was cancelled")
            raise
        except Exception as e:
            if raise_on_error:
                raise
            error_type: Final = type(e).__name__
            verbose_logger.error(
                "MCP client list_resources failed - Error Type: %s, Error: %s, Server: %s, Transport: %s",
                error_type,
                e,
                self.server_url or "stdio",
                self.transport_type,
            )
            # Check if it's a stream/connection error
            if "BrokenResourceError" in error_type or "Broken" in error_type:
                verbose_logger.error(
                    "MCP client detected broken connection/stream during list_resources - "
                    "the MCP server may have crashed, disconnected, or timed out"
                )
            # Return empty list instead of raising to allow graceful degradation
            return []

    async def list_resource_templates(self, *, raise_on_error: bool = False) -> list[ResourceTemplate]:
        """List available resource templates from the server."""
        verbose_logger.debug("MCP client listing resource templates from %s", self.server_url or "stdio")

        async def _list_resource_templates_operation(session: ClientSession) -> ListResourceTemplatesResult:
            capabilities: Final = session.server_capabilities
            if capabilities is not None and capabilities.resources is None:
                return ListResourceTemplatesResult(resource_templates=[])  # mutable-ok: MCP result payload
            try:
                return ListResourceTemplatesResult(
                    resource_templates=await self._list_optional_pages(
                        lambda params: session.list_resource_templates(params=params),
                        lambda page: page.resource_templates,
                    )
                )
            except MCPError as error:
                if error.error.code != METHOD_NOT_FOUND:
                    raise
                verbose_logger.debug(
                    "MCP client list_resource_templates is unsupported by %s: %s", self.server_url or "stdio", error
                )
                return ListResourceTemplatesResult(resource_templates=[])  # mutable-ok: MCP result payload

        try:
            result: Final = await self.run_with_session(_list_resource_templates_operation)
            resource_template_count: Final = len(result.resource_templates)
            resource_template_names: Final = [resource_template.name for resource_template in result.resource_templates]
            verbose_logger.info(
                "MCP client listed %s resource templates from %s: %s",
                resource_template_count,
                self.server_url or "stdio",
                resource_template_names,
            )
            return result.resource_templates
        except asyncio.CancelledError:
            verbose_logger.warning("MCP client list_resource_templates was cancelled")
            raise
        except Exception as e:
            if raise_on_error:
                raise
            error_type: Final = type(e).__name__
            verbose_logger.error(
                "MCP client list_resource_templates failed - Error Type: %s, Error: %s, Server: %s, Transport: %s",
                error_type,
                e,
                self.server_url or "stdio",
                self.transport_type,
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
        verbose_logger.info("MCP client fetching resource '%s'", url)

        async def _read_resource_operation(session: ClientSession):
            verbose_logger.debug("MCP client sending read_resource request to session")
            return await session.read_resource(str(url))

        try:
            read_resource_result: Final = await self.run_with_session(_read_resource_operation)
            verbose_logger.info("MCP client read_resource '%s' completed successfully", url)
            return read_resource_result
        except asyncio.CancelledError:
            verbose_logger.warning("MCP client read_resource was cancelled")
            raise
        except Exception as e:
            import traceback

            error_trace: Final = traceback.format_exc()
            verbose_logger.debug("MCP client read_resource traceback:\n%s", error_trace)
            # Log detailed error information
            error_type: Final = type(e).__name__
            verbose_logger.error(
                "MCP client read_resource failed - Error Type: %s, Error: %s, Url: %s, Server: %s, Transport: %s",
                error_type,
                e,
                url,
                self.server_url or "stdio",
                self.transport_type,
            )
            # Check if it's a stream/connection error
            if "BrokenResourceError" in error_type or "Broken" in error_type:
                verbose_logger.error(
                    "MCP client detected broken connection/stream during read_resource - "
                    "the MCP server may have crashed, disconnected, or timed out."
                )
            raise
