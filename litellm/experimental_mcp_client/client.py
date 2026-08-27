"""
LiteLLM Proxy uses this MCP Client to connnect to other MCP servers.
"""

import asyncio
import base64
import os
from collections.abc import Awaitable, Callable, Generator
from datetime import timedelta
from importlib import metadata
from typing import Any, Final, TypeVar

import httpx
from mcp import ClientSession, McpError, ReadResourceResult, Resource, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client

streamable_http_client: Any | None = None
try:
    import mcp.client.streamable_http as streamable_http_module

    streamable_http_client = getattr(streamable_http_module, "streamable_http_client", None)
except ImportError:
    pass

MCP_STREAMABLE_HTTP_REQUIREMENT: Final = "mcp>=1.28.1"


def missing_streamable_http_client_error() -> ImportError:
    return ImportError(
        f"MCP streamable HTTP transport requires {MCP_STREAMABLE_HTTP_REQUIREMENT}, but the installed "
        f"mcp {metadata.version('mcp')} does not provide streamable_http_client. "
        "Fix with: pip install 'litellm[mcp]' (or upgrade mcp directly: pip install -U mcp)"
    )


from mcp.types import CallToolRequestParams as MCPCallToolRequestParams
from mcp.types import CallToolResult as MCPCallToolResult
from mcp.types import (
    GetPromptRequestParams,
    GetPromptResult,
    Prompt,
    ResourceTemplate,
    TextContent,
)
from mcp.types import Tool as MCPTool
from pydantic import AnyUrl

from litellm._logging import verbose_logger
from litellm.constants import MCP_CLIENT_TIMEOUT, MCP_NPM_CACHE_DIR
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


def strip_auth_scheme(auth_value: str, scheme: str) -> str:
    """Return ``auth_value`` with a leading ``<scheme> `` removed, or unchanged when absent.

    Callers supply both a bare credential and a complete header value, so prefixing
    unconditionally yields ``Bearer Bearer <jwt>``. Scheme names are case-insensitive per
    RFC 7235. A credential is required after the scheme, so both a token that merely begins
    with the scheme text and a scheme with nothing behind it are returned untouched.
    Surrounding whitespace is left to ``_strip_header_whitespace`` at header-build time.
    """
    scheme_name, _, remainder = auth_value.lstrip().partition(" ")
    credential: Final = remainder.lstrip()
    if credential and scheme_name.lower() == scheme.lower():
        return credential
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


def _first_non_cancelled_cause(exc: BaseException) -> BaseException | None:
    queue: Final[list[BaseException]] = [exc]
    while queue:
        current = queue.pop(0)
        nested = getattr(current, "exceptions", None)
        if nested:
            queue.extend(nested)
        elif not isinstance(current, asyncio.CancelledError):
            return current
    return None


_SDK_READ_TIMEOUT_CODE: Final = int(httpx.codes.REQUEST_TIMEOUT)
"""The code the MCP SDK puts on its own elapsed read timeout, an HTTP status in a field that
otherwise carries JSON-RPC error codes."""


def _as_read_timeout(exc: BaseException) -> TimeoutError | None:
    """The session read timeout elapsing, re-expressed as a ``TimeoutError``, or ``None``.

    The SDK reports its own elapsed read timeout as ``McpError`` carrying an HTTP status code in a
    field that otherwise holds JSON-RPC error codes, and it relays an upstream's JSON-RPC error
    through that same class and field. The numeric code alone therefore cannot separate the two, and
    an upstream answering with application code 408 would be reported as a gateway timeout it never
    caused. The SDK raises its own from inside an ``except TimeoutError``, so the elapsed timeout is
    on the context chain, while a relayed error is built from a received message and has no such
    chain; that is the discriminator.
    """
    if not isinstance(exc, McpError) or exc.error.code != _SDK_READ_TIMEOUT_CODE:
        return None
    if not isinstance(exc.__context__, TimeoutError):
        return None
    return TimeoutError(exc.error.message)


TSessionResult = TypeVar("TSessionResult")


class MCPSigV4Auth(httpx.Auth):
    """
    httpx Auth class that signs each request with AWS SigV4.
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
        import boto3
        from botocore.credentials import Credentials

        session_name: Final = aws_session_name or f"litellm-mcp-{int(__import__('time').time())}"
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

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
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
        timeout: float | None = None,
        stdio_config: MCPStdioConfig | None = None,
        extra_headers: dict[str, str] | None = None,
        ssl_verify: VerifyTypes | None = None,
        aws_auth: httpx.Auth | None = None,
        resolved_auth: httpx.Auth | None = None,
        sampling_callback: Callable | None = None,
        elicitation_callback: Callable | None = None,
        logging_callback: Callable | None = None,
    ):
        self.server_url: str = server_url
        self.transport_type: MCPTransport = transport_type
        self.auth_type: MCPAuthType = auth_type
        self.timeout: float = timeout if timeout is not None else MCP_CLIENT_TIMEOUT
        self._mcp_auth_value: str | dict[str, str] | None = None
        self.stdio_config: MCPStdioConfig | None = stdio_config
        self.extra_headers: dict[str, str] | None = extra_headers
        self.ssl_verify: VerifyTypes | None = ssl_verify
        self._aws_auth: httpx.Auth | None = aws_auth
        # A pre-resolved httpx.Auth (e.g. from the v2 credential resolver) attached to the
        # upstream client's auth= slot, taking precedence over the SigV4 aws_auth.
        self._resolved_auth: httpx.Auth | None = resolved_auth
        self._last_initialize_instructions: str | None = None
        self._sampling_callback: Callable | None = sampling_callback
        self._elicitation_callback: Callable | None = elicitation_callback
        self._logging_callback: Callable | None = logging_callback
        # handle the basic auth value if provided
        if auth_value:
            self.update_auth_value(auth_value)

    def _create_transport_context(
        self,
    ) -> tuple[Any, httpx.AsyncClient | None]:
        """
        Create the appropriate transport context based on transport type.
        Returns:
            Tuple of (transport_context, http_client).
            http_client is only set for HTTP transport and needs cleanup.
        """
        http_client: httpx.AsyncClient | None = None
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
        if streamable_http_client is None:
            raise missing_streamable_http_client_error()
        headers = self._get_auth_headers()
        httpx_client_factory = self._create_httpx_client_factory()
        verbose_logger.debug("litellm headers for streamable_http_client: %s", headers)
        http_client = httpx_client_factory(
            headers=headers,
            timeout=httpx.Timeout(self.timeout),
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
        transport_ctx: Any,
        operation: Callable[[ClientSession], Awaitable[TSessionResult]],
    ) -> TSessionResult:
        """
        Execute an operation within a transport and session context.
        Handles entering/exiting contexts and running the operation.
        Passes sampling/elicitation/logging callbacks to the ClientSession
        so that upstream MCP servers can request LLM inference (sampling),
        user input (elicitation), or send log messages.
        """
        transport: Final = await transport_ctx.__aenter__()
        in_flight_error: BaseException | None = None
        try:
            read_stream, write_stream = transport[0], transport[1]
            # Build session kwargs with optional callbacks
            session_kwargs: Final[dict[str, Any]] = {}
            if self._sampling_callback is not None:
                session_kwargs["sampling_callback"] = self._sampling_callback
            if self._elicitation_callback is not None:
                session_kwargs["elicitation_callback"] = self._elicitation_callback
            if self._logging_callback is not None:
                session_kwargs["logging_callback"] = self._logging_callback
            # The SDK drops a response stream that ends without a JSON-RPC reply, so nothing else
            # ever fails the request.
            session_ctx: Final = ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=timedelta(seconds=self.timeout),
                **session_kwargs,
            )
            session: Final = await session_ctx.__aenter__()
            try:
                init_result: Final = await session.initialize()
                self._last_initialize_instructions = None
                if init_result is not None:
                    ins: Final = getattr(init_result, "instructions", None)
                    if isinstance(ins, str) and ins.strip():
                        self._last_initialize_instructions = ins.strip()
                return await operation(session)
            finally:
                try:
                    await session_ctx.__aexit__(None, None, None)
                except BaseException as e:
                    verbose_logger.debug("Error during session context exit: %s", e)
        except BaseException as e:
            in_flight_error = e
            raise
        finally:
            try:
                await transport_ctx.__aexit__(None, None, None)
            except BaseException as exit_error:
                verbose_logger.debug("Error during transport context exit: %s", exit_error)
                root_cause: Final = _first_non_cancelled_cause(exit_error)
                if root_cause is not None and isinstance(in_flight_error, asyncio.CancelledError):
                    raise root_cause from in_flight_error

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
        http_client: httpx.AsyncClient | None = None
        try:
            self._last_initialize_instructions = None
            transport_ctx, http_client = self._create_transport_context()
            return await self._execute_session_operation(transport_ctx, operation)
        except Exception as e:
            read_timeout: Final = _as_read_timeout(e)
            if read_timeout is not None:
                verbose_logger.warning(
                    "MCP client timed out after %ss waiting for %s to answer; the server accepted the "
                    "request and ended its response stream without a JSON-RPC reply",
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
                    await http_client.aclose()
                except BaseException as e:
                    verbose_logger.debug("Error during http_client cleanup: %s", e)

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

    def _get_auth_headers(self) -> dict:
        """Generate authentication headers based on auth type."""
        headers: Final = {}
        if self._mcp_auth_value:
            if isinstance(self._mcp_auth_value, str):
                if self.auth_type == MCPAuth.bearer_token:
                    headers["Authorization"] = f"Bearer {strip_auth_scheme(self._mcp_auth_value, 'Bearer')}"
                elif self.auth_type == MCPAuth.basic:
                    headers["Authorization"] = f"Basic {self._mcp_auth_value}"
                elif self.auth_type == MCPAuth.api_key:
                    headers["X-API-Key"] = self._mcp_auth_value
                elif self.auth_type == MCPAuth.authorization:
                    # This auth type means the caller owns the whole header value.
                    headers["Authorization"] = self._mcp_auth_value
                elif self.auth_type == MCPAuth.oauth2:
                    headers["Authorization"] = f"Bearer {strip_auth_scheme(self._mcp_auth_value, 'Bearer')}"
                elif self.auth_type == MCPAuth.token:
                    headers["Authorization"] = f"token {strip_auth_scheme(self._mcp_auth_value, 'token')}"
                elif self.auth_type == MCPAuth.oauth2_token_exchange:
                    headers["Authorization"] = f"Bearer {strip_auth_scheme(self._mcp_auth_value, 'Bearer')}"
            elif isinstance(self._mcp_auth_value, dict):
                headers.update(self._mcp_auth_value)
        # Note: aws_sigv4 auth is not handled here — SigV4 requires per-request
        # signing (including the body hash), so it uses httpx.Auth flow instead
        # of static headers. See MCPSigV4Auth and _create_httpx_client_factory().
        # update the headers with the extra headers
        if self.extra_headers:
            headers.update(self.extra_headers)
        return _strip_header_whitespace(headers)

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
            headers: dict[str, str] | None = None,
            timeout: httpx.Timeout | None = None,
            auth: httpx.Auth | None = None,
        ) -> httpx.AsyncClient:
            """Create an httpx.AsyncClient with LiteLLM's SSL configuration."""
            # Get unified SSL configuration using the same logic as http_handler.py
            ssl_config: Final = get_ssl_configuration(self.ssl_verify)
            verbose_logger.debug("MCP client using SSL configuration: %s", type(ssl_config).__name__)
            # The MCP SDK's sse_client and streamable_http_client call this factory without
            # passing auth=, so the fallback is used: a v2-resolved auth if present, else the
            # SigV4 aws_auth. Both are None for the common case — no behavior change.
            fallback_auth: Final = self._resolved_auth if self._resolved_auth is not None else self._aws_auth
            effective_auth: Final = auth if auth is not None else fallback_auth
            return httpx.AsyncClient(
                headers=headers,
                timeout=timeout,
                auth=effective_auth,
                verify=ssl_config,
                follow_redirects=True,
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

        async def _list_tools_operation(session: ClientSession):
            return await session.list_tools()

        try:
            result: Final = await self.run_with_session(_list_tools_operation, quiet_on_error=raise_on_error)
            tool_count: Final = len(result.tools)
            tool_names: Final = [tool.name for tool in result.tools]
            verbose_logger.info(
                "MCP client listed %s tools from %s: %s", tool_count, self.server_url or "stdio", tool_names
            )
            return result.tools
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
        return MCPCallToolResult(
            content=[TextContent(type="text", text=f"{type(exc).__name__}: {exc}")],
            isError=True,
        )

    async def call_tool(
        self,
        call_tool_request_params: MCPCallToolRequestParams,
        host_progress_callback: Callable | None = None,
        raise_on_error: bool = False,
    ) -> MCPCallToolResult:
        """
        Call an MCP Tool.

        Args:
            raise_on_error: When True, re-raise the underlying exception instead of returning an
                ``isError=True`` result. The token-exchange (OBO) tool-call path uses this to detect
                an upstream 401 so it can re-mint the exchanged token and retry once; every other
                caller keeps the default and gets graceful ``isError`` degradation.
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
            return await session.call_tool(
                name=call_tool_request_params.name,
                arguments=call_tool_request_params.arguments,
                progress_callback=on_progress,
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

    async def list_prompts(self) -> list[Prompt]:
        """List available prompts from the server."""
        verbose_logger.debug("MCP client listing tools from %s", self.server_url or "stdio")

        async def _list_prompts_operation(session: ClientSession):
            return await session.list_prompts()

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

    async def list_resources(self) -> list[Resource]:
        """List available resources from the server."""
        verbose_logger.debug("MCP client listing resources from %s", self.server_url or "stdio")

        async def _list_resources_operation(session: ClientSession):
            return await session.list_resources()

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

    async def list_resource_templates(self) -> list[ResourceTemplate]:
        """List available resource templates from the server."""
        verbose_logger.debug("MCP client listing resource templates from %s", self.server_url or "stdio")

        async def _list_resource_templates_operation(session: ClientSession):
            return await session.list_resource_templates()

        try:
            result: Final = await self.run_with_session(_list_resource_templates_operation)
            resource_template_count: Final = len(result.resourceTemplates)
            resource_template_names: Final = [resourceTemplate.name for resourceTemplate in result.resourceTemplates]
            verbose_logger.info(
                "MCP client listed %s resource templates from %s: %s",
                resource_template_count,
                self.server_url or "stdio",
                resource_template_names,
            )
            return result.resourceTemplates
        except asyncio.CancelledError:
            verbose_logger.warning("MCP client list_resource_templates was cancelled")
            raise
        except Exception as e:
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
            return await session.read_resource(url)

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
