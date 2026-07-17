"""
LiteLLM MCP Server Routes
"""

# pyright: reportInvalidTypeForm=false, reportArgumentType=false, reportOptionalCall=false

import asyncio
import contextlib
import contextvars
import hashlib
import json
import time
import traceback
import types
import uuid
from datetime import datetime
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)
from urllib.parse import urlsplit, urlunsplit

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import AnyUrl, ConfigDict
from starlette.requests import Request as StarletteRequest
from starlette.responses import JSONResponse
from starlette.types import Message, Receive, Scope, Send

from litellm._logging import verbose_logger
from litellm.constants import MAXIMUM_TRACEBACK_LINES_TO_LOG
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
    MCPRequestHandler,
)
from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
    get_request_base_url,
)
from litellm.proxy._experimental.mcp_server.exceptions import (
    MCPToolResultError,
    MCPUpstreamAuthError,
)
from litellm.proxy._experimental.mcp_server.mcp_context import (
    _mcp_active_toolset_id,
    _mcp_gateway_initialize_instructions,
    _mcp_gateway_server_name,
)
from litellm.proxy._experimental.mcp_server.mcp_debug import MCPDebug
from litellm.proxy._experimental.mcp_server.utils import (
    LITELLM_MCP_SERVER_DESCRIPTION,
    LITELLM_MCP_SERVER_NAME,
    LITELLM_MCP_SERVER_VERSION,
    MCPMissingUserEnvVarsError,
    add_server_prefix_to_name,
    extract_mcp_tool_result_error_message,
    get_server_prefix,
    iter_known_server_prefixes,
)
from litellm.proxy._types import (
    ProxyException,
    SpecialMCPServerNames,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.ip_address_utils import IPAddressUtils
from litellm.proxy.litellm_pre_call_utils import (
    LiteLLMProxyRequestSetup,
    get_chain_id_from_headers,
)
from litellm.types.mcp import MCPAuth, MCPSpecVersion
from litellm.types.mcp_server.mcp_server_manager import MCPInfo, MCPServer
from litellm.types.utils import CallTypes, StandardLoggingMCPToolCall
from litellm.utils import Rules, client, function_setup

# Short-lived in-memory cache for BYOK credentials.
# Keyed by (user_id, server_id); value is (credential_or_None, monotonic_timestamp).
# Storing the credential value (not just a bool) means _get_byok_credential and
# _check_byok_credential share a single DB round-trip per TTL window.
_byok_cred_cache: Dict[Tuple[str, str], Tuple[Optional[str], float]] = {}
_BYOK_CRED_CACHE_TTL = 60  # seconds
_BYOK_CRED_CACHE_MAX_SIZE = 4096  # cap to prevent unbounded growth
_STATEFUL_SESSION_IDLE_TIMEOUT_SECONDS = 30 * 60
# Upper bound on concurrent stateful sessions a single caller may hold. Each
# `initialize` creates a session that survives until the idle timeout, so
# without a cap an authenticated client could spam `initialize` and exhaust
# memory. The caller's own oldest idle sessions are evicted to make room; if
# the cap is still hit (every session in flight), the new `initialize` is
# rejected with 429.
_MAX_STATEFUL_SESSIONS_PER_OWNER = 100
# Maximum bytes to peek when sniffing the JSON-RPC method on a POST.
# An `initialize` envelope is a few hundred bytes; capping the peek
# prevents an authenticated client from forcing the proxy to buffer an
# arbitrarily large body just to make a routing decision.
_MCP_ROUTING_PEEK_MAX_BYTES = 4096


def _redact_mcp_resource_url(url: Optional[str]) -> Optional[str]:
    """Reduce an MCP server URL to its origin (scheme + host + port) for logging.

    Everything else is dropped: userinfo (``user:pass@``), the query string, the
    fragment, and the path, because hosted MCP servers routinely embed the
    credential in the path (e.g. ``/mcp/s/<token>``) and this value is persisted
    in spend-log metadata that a caller who can invoke the tool can read back.
    Returns None when the URL has no host to identify (nothing safe to log).
    """
    if not isinstance(url, str) or not url:
        return None
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if not parts.hostname:
        return None
    netloc = f"{parts.hostname}:{parts.port}" if parts.port else parts.hostname
    return urlunsplit((parts.scheme, netloc, "", "", "")) or None


def _invalidate_byok_cred_cache(user_id: str, server_id: str) -> None:
    """Remove a (user_id, server_id) entry from the BYOK credential cache.

    Call this after storing or deleting a credential so subsequent calls
    see the fresh value rather than a stale cached result.
    """
    _byok_cred_cache.pop((user_id, server_id), None)


def _write_byok_cred_cache(user_id: str, server_id: str, credential: Optional[str]) -> None:
    """Write a credential value to the cache, evicting all entries if at capacity."""
    if len(_byok_cred_cache) >= _BYOK_CRED_CACHE_MAX_SIZE:
        _byok_cred_cache.clear()
    _byok_cred_cache[(user_id, server_id)] = (credential, time.monotonic())


# Check if MCP is available
# "mcp" requires python 3.10 or higher, but several litellm users use python 3.8
# We're making this conditional import to avoid breaking users who use python 3.8.
# TODO: Make this a util function for litellm client usage
MCP_AVAILABLE: bool = True
try:
    import weakref

    from mcp import ReadResourceResult, Resource
    from mcp.server import Server
    from mcp.server.lowlevel.helper_types import ReadResourceContents
    from mcp.server.session import ServerSession as _McpServerSession
    from mcp.types import (
        BlobResourceContents,
        GetPromptResult,
        ResourceTemplate,
        TextResourceContents,
        Tool,
    )

    # Robust auth lookup keyed by session_object.
    _session_obj_auth_storage: "weakref.WeakKeyDictionary[Any, MCPAuthenticatedUser]" = weakref.WeakKeyDictionary()

    active_mcp_session_var: contextvars.ContextVar[Optional[_McpServerSession]] = contextvars.ContextVar(
        "active_mcp_session", default=None
    )
except ImportError as e:
    verbose_logger.debug(f"MCP module not found: {e}")
    MCP_AVAILABLE = False
    # When MCP is not available, we set these to None at module level
    # All code using these types is inside `if MCP_AVAILABLE:` blocks
    # so they will never be accessed at runtime
    BlobResourceContents = None  # type: ignore
    GetPromptResult = None  # type: ignore
    ReadResourceContents = None  # type: ignore
    ReadResourceResult = None  # type: ignore
    Resource = None  # type: ignore
    ResourceTemplate = None  # type: ignore
    Server = None  # type: ignore
    TextResourceContents = None  # type: ignore


# Global variables to track initialization
_SESSION_MANAGERS_INITIALIZED = False
_INITIALIZATION_LOCK = asyncio.Lock()


def _mcp_session_id_from_headers(
    raw_headers: Optional[Dict[str, str]],
) -> Optional[str]:
    """The ``mcp-session-id`` of a stateful MCP session, read case-insensitively
    from the request headers. ``None`` for stateless calls (no such header)."""
    if not raw_headers:
        return None
    for key, value in raw_headers.items():
        if isinstance(key, str) and key.lower() == "mcp-session-id":
            return value or None
    return None


def _jsonrpc_text_has_top_level_method(text: str) -> bool:
    """Whether a (possibly truncated) JSON-RPC envelope has a ``method`` key at
    the root object's top level.

    Used to tell a request/notification (carries ``method``) apart from a
    response (carries ``result``/``error`` and no top-level ``method``). A
    response payload can itself nest a ``method`` field, so only keys at the
    root object's depth are inspected rather than searching the whole string.
    Returns ``True`` only when a top-level ``method`` key is positively found;
    truncation that hides it yields ``False``.
    """
    depth = 0
    in_string = False
    escaped = False
    in_object: List[bool] = []
    reading_key = False
    expect_key = False
    key_chars: List[str] = []
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
                if reading_key and depth == 1 and "".join(key_chars) == "method":
                    return True
            elif reading_key:
                key_chars.append(ch)
            continue
        if ch == '"':
            in_string = True
            reading_key = expect_key and depth >= 1 and in_object[-1]
            key_chars = []
            expect_key = False
        elif ch == "{" or ch == "[":
            depth += 1
            in_object.append(ch == "{")
            expect_key = ch == "{"
        elif ch == "}" or ch == "]":
            if in_object:
                in_object.pop()
            depth -= 1
            if depth <= 0:
                break
            expect_key = False
        elif ch == ",":
            expect_key = bool(in_object) and in_object[-1]
        elif ch == ":":
            expect_key = False
    return False


def _mcp_meta_trace_carrier(req_ctx: object) -> Optional[dict[str, str]]:
    """The W3C trace context (``traceparent``/``tracestate``) the MCP client
    propagated in the request's ``params._meta`` (SEP-414), or ``None``.

    Per the OTel MCP semconv the MCP span parents to this propagated context rather
    than to the HTTP/session transport (which is recorded as a link instead), so a
    streamable-HTTP session that multiplexes many messages does not glue every
    message under the session's first request. The client's W3C Baggage is
    deliberately excluded: it is caller-controlled, and the otel baggage processor
    stamps allowlisted baggage keys (``litellm.team.id``, ``litellm.metadata.*``,
    ...) onto the span, so honoring remote baggage would let a client spoof a
    span's identity attribution.
    """
    meta = getattr(req_ctx, "meta", None)
    extra = getattr(meta, "model_extra", None)
    if not isinstance(extra, dict):
        return None
    carrier = {key: extra[key] for key in ("traceparent", "tracestate") if isinstance(extra.get(key), str)}
    return carrier or None


def _otel_set_mcp_trace_carrier(carrier: Optional[dict[str, str]]) -> object:
    """Stash ``carrier`` for the otel_v2 MCP span and return a reset token, or
    ``None`` when otel_v2 is unavailable. Lazily imported so opentelemetry stays an
    optional dependency."""
    try:
        from litellm.integrations.otel.plumbing.context import (
            set_mcp_message_trace_carrier,
        )

        return set_mcp_message_trace_carrier(carrier)
    except ImportError:
        return None


def _otel_reset_mcp_trace_carrier(token: object) -> None:
    """Clear the per-message trace carrier so it never leaks to the next message on
    the same session task. Paired with ``_otel_set_mcp_trace_carrier``."""
    if token is None:
        return
    try:
        from litellm.integrations.otel.plumbing.context import (
            reset_mcp_message_trace_carrier,
        )

        reset_mcp_message_trace_carrier(token)
    except ImportError:
        return


def _proxy_exception_to_http_exception(exc: ProxyException) -> HTTPException:
    """Map a ``ProxyException`` to an ``HTTPException`` that preserves its real
    status code and headers.

    ``user_api_key_auth`` raises ``ProxyException`` (not ``HTTPException``) on
    auth failures. The MCP ASGI handlers re-raise ``HTTPException`` to keep the
    status and any ``WWW-Authenticate`` challenge, but a ``ProxyException`` would
    otherwise fall through to their generic handler and be flattened to a 500 —
    dropping the 401 + challenge an OAuth client needs to re-authenticate, so the
    tool call surfaces as a cancelled/terminated session instead.
    """
    try:
        status_code = int(exc.code)
    except (TypeError, ValueError):
        status_code = 500
    return HTTPException(
        status_code=status_code,
        detail=exc.message,
        headers=exc.headers or None,
    )


if MCP_AVAILABLE:
    from mcp.server import Server

    # Import auth context variables and middleware
    from mcp.server.auth.middleware.auth_context import (
        AuthContextMiddleware,
        auth_context_var,
    )
    from mcp.server.lowlevel.server import NotificationOptions
    from mcp.server.models import InitializationOptions

    try:
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    except ImportError:
        StreamableHTTPSessionManager = None  # type: ignore
    from mcp.types import (
        CallToolResult,
        EmbeddedResource,
        ImageContent,
        ListToolsResult,
        Prompt,
        TextContent,
    )
    from mcp.types import Tool as MCPTool

    from litellm.proxy._experimental.mcp_server.auth.litellm_auth_handler import (
        MCPAuthenticatedUser,
    )
    from litellm.proxy._experimental.mcp_server.faults.list_outcomes import (
        SERVER_OUTCOMES_META_KEY,
        AggregateToolListing,
        ServerListOk,
        ServerOutcome,
        classify_list_exception,
        outcome_wire_value,
    )
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        MCPServerManager,
        _caller_authorization_fans_out,
        _client_forwarded_authorization_headers,
        _should_strip_caller_authorization,
        _without_authorization,
        global_mcp_server_manager,
    )
    from litellm.proxy._experimental.mcp_server.openapi_to_mcp_generator import (
        _request_auth_header,
        _request_extra_headers,
    )
    from litellm.proxy._experimental.mcp_server.sse_transport import SseServerTransport
    from litellm.proxy._experimental.mcp_server.tool_registry import (
        global_mcp_tool_registry,
    )
    from litellm.proxy._experimental.mcp_server.utils import (
        MCP_TOOL_PREFIX_SEPARATOR,
        is_tool_name_prefixed,
        normalize_server_name,
        split_server_prefix_from_name,
        strip_known_server_prefix,
    )

    ######################################################
    ############ MCP Tools List REST API Response Object #
    # Defined here because we don't want to add `mcp` as a
    # required dependency for `litellm` pip package
    ######################################################
    class ListMCPToolsRestAPIResponseObject(MCPTool):
        """
        Object returned by the /tools/list REST API route.
        """

        mcp_info: Optional[MCPInfo] = None
        model_config = ConfigDict(arbitrary_types_allowed=True)

    def _normalize_resource_contents(contents: list) -> List[ReadResourceContents]:
        """Normalize ResourceContents to ReadResourceContents, preserving meta (MCP 1.26.0+)."""
        normalized: List[ReadResourceContents] = []
        for content in contents:
            meta = getattr(content, "meta", None)
            if meta is None and hasattr(content, "model_dump"):
                d = content.model_dump()
                meta = d.get("meta")
                if meta is None:
                    meta = d.get("_meta")
            if isinstance(content, TextResourceContents):
                normalized.append(
                    ReadResourceContents(
                        content=content.text,
                        mime_type=content.mimeType,
                        meta=meta,
                    )
                )
            elif isinstance(content, BlobResourceContents):
                normalized.append(
                    ReadResourceContents(
                        content=content.blob,
                        mime_type=content.mimeType,
                        meta=meta,
                    )
                )
        return normalized

    def _gateway_create_initialization_options(
        self,
        notification_options: Optional[NotificationOptions] = None,
        experimental_capabilities: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> InitializationOptions:
        opts = Server.create_initialization_options(
            self,
            notification_options=notification_options,
            experimental_capabilities=experimental_capabilities or {},
        )
        updates: Dict[str, Any] = {}
        merged = _mcp_gateway_initialize_instructions.get()
        if merged is not None:
            updates["instructions"] = merged
        scoped_server_name = _mcp_gateway_server_name.get()
        if scoped_server_name is not None:
            updates["server_name"] = scoped_server_name
        return opts.model_copy(update=updates) if updates else opts

    ########################################################
    ############ Initialize the MCP Server #################
    ########################################################
    server: Server = Server(
        name=LITELLM_MCP_SERVER_NAME,
        version=LITELLM_MCP_SERVER_VERSION,
    )
    server.create_initialization_options = types.MethodType(  # type: ignore[method-assign]
        _gateway_create_initialization_options, server
    )
    sse: SseServerTransport = SseServerTransport("/mcp/sse/messages")

    # Create session managers
    session_manager_stateless = StreamableHTTPSessionManager(
        app=server,
        event_store=None,
        json_response=False,  # enables SSE streaming
        stateless=True,
    )

    session_manager_stateful = StreamableHTTPSessionManager(
        app=server,
        event_store=None,  # TODO: Add EventStore for reconnection/event replay if needed
        json_response=False,  # enables SSE streaming
        stateless=False,
    )
    _stateful_session_auth_contexts: Dict[str, MCPAuthenticatedUser] = {}
    _stateful_session_auth_context_last_seen: Dict[str, float] = {}
    # Maps session_id -> owner identifier (hashed API key/token) so we can
    # reject requests that supply a session_id created by a different caller.
    # Without this, a leaked mcp-session-id could be driven (or terminated)
    # by any other authenticated proxy user.
    _stateful_session_owners: Dict[str, str] = {}
    # Per-session lock that serializes ``handle_request`` for the same
    # mcp-session-id. The stored ``MCPAuthenticatedUser`` is mutated in place
    # by ``_update_auth_context`` each request; without this lock, two
    # concurrent requests on the same session would clobber each other's
    # auth headers / mcp_servers / oauth state while in-flight callbacks are
    # still reading the shared object.
    _stateful_session_locks: Dict[str, asyncio.Lock] = {}
    _stateful_session_active_request_counts: Dict[str, int] = {}

    def _remove_stateful_session_tracking(session_id: str) -> None:
        _stateful_session_auth_contexts.pop(session_id, None)
        _stateful_session_auth_context_last_seen.pop(session_id, None)
        _stateful_session_owners.pop(session_id, None)
        _stateful_session_locks.pop(session_id, None)
        _stateful_session_active_request_counts.pop(session_id, None)

    # Keep this alias so existing references to session_manager still work
    session_manager = session_manager_stateless

    # Create SSE session manager
    sse_session_manager = StreamableHTTPSessionManager(
        app=server,
        event_store=None,
        json_response=False,  # Use SSE responses for this endpoint
        stateless=True,
    )

    # Context managers for proper lifecycle management
    _session_manager_cm = None
    _session_manager_stateful_cm = None
    _sse_session_manager_cm = None
    _stateful_auth_context_cleanup_task: Optional[asyncio.Task] = None

    async def _purge_expired_stateful_session_auth_contexts(
        now: Optional[float] = None,
    ) -> None:
        """Terminate expired stateful sessions and drop their auth contexts."""
        now = time.monotonic() if now is None else now
        server_instances = getattr(session_manager_stateful, "_server_instances", {})
        expired_session_ids = []
        for session_id, last_seen in _stateful_session_auth_context_last_seen.items():
            if _stateful_session_active_request_counts.get(session_id, 0) > 0:
                continue
            if now - last_seen >= _STATEFUL_SESSION_IDLE_TIMEOUT_SECONDS or session_id not in server_instances:
                expired_session_ids.append(session_id)

        for session_id in expired_session_ids:
            # Re-check the active-request count immediately before tearing
            # the session down. ``await transport.terminate()`` yields to
            # the event loop, so a request that started after the first
            # collection pass could otherwise observe its transport being
            # ripped out from under it mid-flight.
            if _stateful_session_active_request_counts.get(session_id, 0) > 0:
                continue
            # Pop transport + terminate BEFORE removing owner/auth tracking.
            # Reversing the order avoids a window where ``_stateful_session_owners``
            # is empty but ``server_instances`` still serves the session — a
            # concurrent request in that window would observe ``expected_owner
            # is None`` and bypass the owner-binding check.
            transport = server_instances.pop(session_id, None)
            if transport is not None:
                await transport.terminate()
            _remove_stateful_session_tracking(session_id)

        for session_id in list(_stateful_session_auth_context_last_seen):
            if session_id not in _stateful_session_auth_contexts:
                _remove_stateful_session_tracking(session_id)

    async def _enforce_stateful_session_cap_for_owner(owner: str) -> bool:
        """
        Bound the number of concurrent stateful sessions a single caller holds
        before routing a new ``initialize`` to the stateful manager.

        Evicts the caller's *own* oldest idle sessions (no in-flight requests)
        to make room, so a busy-but-legitimate client keeps its newest sessions
        and other callers are never affected. Returns ``True`` if the new
        session may proceed, or ``False`` when the caller is already at the cap
        with every session in flight (the new ``initialize`` should be rejected).
        """
        server_instances = getattr(session_manager_stateful, "_server_instances", {})

        def _owned_live_session_ids() -> List[str]:
            return [
                session_id
                for session_id, session_owner in _stateful_session_owners.items()
                if session_owner == owner and session_id in server_instances
            ]

        owned = _owned_live_session_ids()
        if len(owned) < _MAX_STATEFUL_SESSIONS_PER_OWNER:
            return True

        for session_id in sorted(
            owned,
            key=lambda sid: _stateful_session_auth_context_last_seen.get(sid, 0.0),
        ):
            if len(_owned_live_session_ids()) < _MAX_STATEFUL_SESSIONS_PER_OWNER:
                break
            if _stateful_session_active_request_counts.get(session_id, 0) > 0:
                continue
            transport = server_instances.pop(session_id, None)
            if transport is not None:
                await transport.terminate()
            _remove_stateful_session_tracking(session_id)

        return len(_owned_live_session_ids()) < _MAX_STATEFUL_SESSIONS_PER_OWNER

    async def _cleanup_expired_stateful_session_auth_contexts() -> None:
        while True:
            await asyncio.sleep(_STATEFUL_SESSION_IDLE_TIMEOUT_SECONDS)
            try:
                await _purge_expired_stateful_session_auth_contexts()
            except Exception as e:
                verbose_logger.exception(f"Error cleaning up expired MCP stateful sessions: {e}")

    async def initialize_session_managers():
        """Initialize the session managers. Can be called from main app lifespan."""
        global \
            _SESSION_MANAGERS_INITIALIZED, \
            _session_manager_cm, \
            _session_manager_stateful_cm, \
            _sse_session_manager_cm, \
            _stateful_auth_context_cleanup_task

        # Use async lock to prevent concurrent initialization
        async with _INITIALIZATION_LOCK:
            if _SESSION_MANAGERS_INITIALIZED:
                return

            verbose_logger.info("Initializing MCP session managers...")

            # Start the session managers with context managers
            _session_manager_cm = session_manager_stateless.run()
            _session_manager_stateful_cm = session_manager_stateful.run()
            _sse_session_manager_cm = sse_session_manager.run()

            # Enter the context managers
            await _session_manager_cm.__aenter__()
            await _session_manager_stateful_cm.__aenter__()
            await _sse_session_manager_cm.__aenter__()
            _stateful_auth_context_cleanup_task = asyncio.create_task(_cleanup_expired_stateful_session_auth_contexts())

            _SESSION_MANAGERS_INITIALIZED = True
            verbose_logger.info("MCP Server started with StreamableHTTP and SSE session managers!")

    async def shutdown_session_managers():
        """Shutdown the session managers."""
        global \
            _SESSION_MANAGERS_INITIALIZED, \
            _session_manager_cm, \
            _session_manager_stateful_cm, \
            _sse_session_manager_cm, \
            _stateful_auth_context_cleanup_task

        if _SESSION_MANAGERS_INITIALIZED:
            verbose_logger.info("Shutting down MCP session managers...")

            try:
                if _stateful_auth_context_cleanup_task:
                    _stateful_auth_context_cleanup_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await _stateful_auth_context_cleanup_task
                if _session_manager_cm:
                    await _session_manager_cm.__aexit__(None, None, None)
                if _session_manager_stateful_cm:
                    await _session_manager_stateful_cm.__aexit__(None, None, None)
                if _sse_session_manager_cm:
                    await _sse_session_manager_cm.__aexit__(None, None, None)
            except Exception as e:
                verbose_logger.exception(f"Error during session manager shutdown: {e}")

            _session_manager_cm = None
            _session_manager_stateful_cm = None
            _sse_session_manager_cm = None
            _stateful_auth_context_cleanup_task = None
            _SESSION_MANAGERS_INITIALIZED = False

    @contextlib.asynccontextmanager
    async def lifespan(app) -> AsyncIterator[None]:
        """Application lifespan context manager."""
        await initialize_session_managers()
        try:
            yield
        finally:
            await shutdown_session_managers()

    ########################################################
    ############### MCP Server Routes #######################
    ########################################################

    @server.list_tools()
    async def handle_list_tools() -> "ListToolsResult | List[Tool]":
        """
        List all available tools, with each server's listing outcome attached to the result's
        ``_meta`` (SERVER_OUTCOMES_META_KEY) so a broken upstream is distinguishable from a healthy
        server with no tools. Returning a ListToolsResult (rather than a bare list) makes the MCP SDK
        pass the result through unwrapped, which is what lets the ``_meta`` survive to the client.
        Also captures the active session for propagation to callbacks.
        """
        from mcp.server.lowlevel.server import request_ctx

        req_ctx = request_ctx.get(None)
        _session_reset_token = None
        if req_ctx:
            _session_reset_token = active_mcp_session_var.set(req_ctx.session)
        _trace_token = None

        try:
            _trace_token = _otel_set_mcp_trace_carrier(_mcp_meta_trace_carrier(req_ctx))
            # Get user authentication from context variable
            (
                user_api_key_auth,
                mcp_auth_header,
                mcp_servers,
                mcp_server_auth_headers,
                oauth2_headers,
                raw_headers,
                _client_ip,
            ) = await get_or_extract_auth_context()
            verbose_logger.debug(f"MCP list_tools - User API Key Auth from context: {user_api_key_auth}")
            verbose_logger.debug(f"MCP list_tools - MCP servers from context: {mcp_servers}")
            verbose_logger.debug(
                f"MCP list_tools - MCP server auth headers: {list(mcp_server_auth_headers.keys()) if mcp_server_auth_headers else None}"
            )
            if getattr(
                getattr(user_api_key_auth, "object_permission", None),
                "mcp_tool_search_enabled",
                False,
            ):
                from mcp.types import Tool

                from litellm.proxy._experimental.mcp_server.tool_search import (
                    get_virtual_tool_definitions,
                )

                return [Tool(**d) for d in get_virtual_tool_definitions()]

            # Get mcp_servers from context variable
            verbose_logger.debug("MCP list_tools - Calling _list_mcp_tools")
            listing = await _list_mcp_tools(
                user_api_key_auth=user_api_key_auth,
                mcp_auth_header=mcp_auth_header,
                mcp_servers=mcp_servers,
                mcp_server_auth_headers=mcp_server_auth_headers,
                oauth2_headers=oauth2_headers,
                raw_headers=raw_headers,
                log_list_tools_to_spendlogs=True,
                list_tools_log_source="mcp_protocol",
            )
            verbose_logger.info(f"MCP list_tools - Successfully returned {len(listing.tools)} tools")
            if not listing.outcomes:
                return listing.tools
            outcome_meta = {
                SERVER_OUTCOMES_META_KEY: {
                    key: outcome_wire_value(outcome) for key, outcome in listing.outcomes.items()
                }
            }
            return ListToolsResult.model_validate({"tools": listing.tools, "_meta": outcome_meta})
        except Exception as e:
            verbose_logger.exception(f"Error in list_tools endpoint: {str(e)}")
            # Return empty list instead of failing completely
            # This prevents the HTTP stream from failing and allows the client to get a response
            return []
        finally:
            _otel_reset_mcp_trace_carrier(_trace_token)
            if _session_reset_token is not None:
                active_mcp_session_var.reset(_session_reset_token)

    def _capture_host_progress_callback(host_server) -> Optional[Callable]:
        """Return a progress-forwarding callback bound to the host MCP session.

        Returns ``None`` when the host did not supply a progress token.
        """
        try:
            host_ctx = host_server.request_context
        except Exception as e:
            verbose_logger.warning(f"Could not capture host progress context: {e}")
            return None

        if not (host_ctx and hasattr(host_ctx, "meta") and host_ctx.meta):
            return None
        host_token = getattr(host_ctx.meta, "progressToken", None)
        if host_token is None or not (hasattr(host_ctx, "session") and host_ctx.session):
            return None
        host_session = host_ctx.session

        async def forward_progress(progress: float, total: Optional[float]):
            """Forward progress notifications from external MCP to Host"""
            try:
                await host_session.send_progress_notification(
                    progress_token=host_token,
                    progress=progress,
                    total=total,
                )
                verbose_logger.debug(f"Forwarded progress {progress}/{total} to Host")
            except Exception as e:
                verbose_logger.error(f"Failed to forward progress to Host: {e}")

        verbose_logger.debug(f"Host progressToken captured: {str(host_token)[:8]}...")
        return forward_progress

    async def _build_virtual_call_logging_obj(
        name: str,
        arguments: dict[str, Any],
        user_api_key_auth: UserAPIKeyAuth,
    ) -> Optional[LiteLLMLoggingObj]:
        """Run the pre-call pipeline (guardrails + logging setup) for a virtual
        mcp_tool_call so the SSE path spend-logs like the REST path."""
        from fastapi import Request

        from litellm.proxy.common_request_processing import (
            ProxyBaseLLMRequestProcessing,
        )
        from litellm.proxy.proxy_server import (
            general_settings,
            proxy_config,
            proxy_logging_obj,
        )

        request = Request(
            scope={
                "type": "http",
                "method": "POST",
                "path": "/mcp/tools/call",
                "headers": [(b"content-type", b"application/json")],
            }
        )
        _, virtual_logging_obj = await ProxyBaseLLMRequestProcessing(
            data={"name": name, "arguments": arguments}
        ).common_processing_pre_call_logic(
            request=request,
            user_api_key_dict=user_api_key_auth,
            proxy_config=proxy_config,
            route_type=CallTypes.call_mcp_tool.value,
            proxy_logging_obj=proxy_logging_obj,
            general_settings=general_settings,
        )
        return virtual_logging_obj

    async def _dispatch_virtual_mcp_tool(
        name: str,
        arguments: Optional[dict[str, Any]],
        user_api_key_auth: Optional[UserAPIKeyAuth],
        client_ip: Optional[str],
        mcp_servers: Optional[list[str]] = None,
        mcp_auth_header: Optional[str] = None,
        mcp_server_auth_headers: Optional[dict[str, dict[str, str]]] = None,
        oauth2_headers: Optional[dict[str, str]] = None,
        raw_headers: Optional[dict[str, str]] = None,
    ) -> Optional[CallToolResult]:
        """Handle the mcp_tool_search / mcp_tool_call virtual tools.

        Returns a CallToolResult when ``name`` is a virtual tool, else ``None`` so
        the caller falls through to normal tool routing.
        """
        from litellm.proxy._experimental.mcp_server.tool_search import (
            MCP_TOOL_CALL_TOOL_NAME,
            MCP_TOOL_SEARCH_TOOL_NAME,
            coerce_top_k,
            handle_mcp_tool_call,
            handle_mcp_tool_search,
        )

        if name not in (MCP_TOOL_SEARCH_TOOL_NAME, MCP_TOOL_CALL_TOOL_NAME):
            return None

        if not getattr(
            getattr(user_api_key_auth, "object_permission", None),
            "mcp_tool_search_enabled",
            False,
        ):
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Tool {name} requires mcp_tool_search_enabled on the key",
                    )
                ],
                isError=True,
            )

        args = arguments or {}
        if name == MCP_TOOL_SEARCH_TOOL_NAME:
            return await handle_mcp_tool_search(
                query=args.get("query", ""),
                top_k=coerce_top_k(args.get("top_k", 5)),
                user_api_key_dict=user_api_key_auth,
                client_ip=client_ip,
                mcp_servers=mcp_servers,
                mcp_auth_header=mcp_auth_header,
                mcp_server_auth_headers=mcp_server_auth_headers,
                oauth2_headers=oauth2_headers,
                raw_headers=raw_headers,
            )

        assert user_api_key_auth is not None  # guaranteed by the flag check above
        virtual_logging_obj = await _build_virtual_call_logging_obj(
            name=name, arguments=args, user_api_key_auth=user_api_key_auth
        )
        return await handle_mcp_tool_call(
            tool_name=args.get("tool_name", ""),
            arguments=args.get("arguments") or {},
            user_api_key_dict=user_api_key_auth,
            client_ip=client_ip,
            mcp_servers=mcp_servers,
            mcp_auth_header=mcp_auth_header,
            mcp_server_auth_headers=mcp_server_auth_headers,
            oauth2_headers=oauth2_headers,
            raw_headers=raw_headers,
            litellm_logging_obj=virtual_logging_obj,
        )

    @server.call_tool()
    async def mcp_server_tool_call(name: str, arguments: Dict[str, Any] | None) -> CallToolResult:
        """
        Call a specific tool with the provided arguments
        Args:
            name (str): Name of the tool to call
            arguments (Dict[str, Any] | None): Arguments to pass to the tool
        Returns:
            List[Union[MCPTextContent, MCPImageContent, MCPEmbeddedResource]]: Tool execution results
        Raises:
            HTTPException: If tool not found or arguments missing
        """
        from fastapi import Request
        from mcp.server.lowlevel.server import request_ctx
        from mcp.types import CallToolResult

        from litellm.exceptions import BlockedPiiEntityError, GuardrailRaisedException
        from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request
        from litellm.proxy.proxy_server import proxy_config

        req_ctx = request_ctx.get(None)
        _session_reset_token = None
        if req_ctx:
            _session_reset_token = active_mcp_session_var.set(req_ctx.session)
        _trace_token = None

        try:
            _trace_token = _otel_set_mcp_trace_carrier(_mcp_meta_trace_carrier(req_ctx))
            # Validate arguments
            (
                user_api_key_auth,
                mcp_auth_header,
                mcp_servers,
                mcp_server_auth_headers,
                oauth2_headers,
                raw_headers,
                _client_ip,
            ) = await get_or_extract_auth_context()
            verbose_logger.debug(
                f"MCP mcp_server_tool_call - user_api_key_auth={user_api_key_auth}, user_role={getattr(user_api_key_auth, 'user_role', 'N/A')}"
            )

            verbose_logger.debug(f"MCP mcp_server_tool_call - User API Key Auth from context: {user_api_key_auth}")

            try:
                # Inside this try so virtual-tool errors convert to isError
                # CallToolResult instead of raising out of the protocol handler.
                virtual_tool_result = await _dispatch_virtual_mcp_tool(
                    name=name,
                    arguments=arguments,
                    user_api_key_auth=user_api_key_auth,
                    client_ip=_client_ip,
                    mcp_servers=mcp_servers,
                    mcp_auth_header=mcp_auth_header,
                    mcp_server_auth_headers=mcp_server_auth_headers,
                    oauth2_headers=oauth2_headers,
                    raw_headers=raw_headers,
                )
                if virtual_tool_result is not None:
                    return virtual_tool_result

                host_progress_callback = _capture_host_progress_callback(server)
                # Create a body date for logging
                body_data = {"name": name, "arguments": arguments}
                # Set trace/session id from raw_headers so spend logs and logging_obj stay consistent (same as A2A)
                chain_id = get_chain_id_from_headers(raw_headers)
                if chain_id:
                    body_data["litellm_trace_id"] = chain_id
                    body_data["litellm_session_id"] = chain_id

                request = Request(
                    scope={
                        "type": "http",
                        "method": "POST",
                        "path": "/mcp/tools/call",
                        "headers": [(b"content-type", b"application/json")],
                    }
                )
                if user_api_key_auth is not None:
                    data = await add_litellm_data_to_request(
                        data=body_data,
                        request=request,
                        user_api_key_dict=user_api_key_auth,
                        proxy_config=proxy_config,
                    )
                else:
                    data = body_data

                response = await call_mcp_tool(
                    user_api_key_auth=user_api_key_auth,
                    mcp_auth_header=mcp_auth_header,
                    mcp_servers=mcp_servers,
                    mcp_server_auth_headers=mcp_server_auth_headers,
                    oauth2_headers=oauth2_headers,
                    raw_headers=raw_headers,
                    host_progress_callback=host_progress_callback,
                    **data,  # for logging
                )
            except MCPMissingUserEnvVarsError as e:
                verbose_logger.info(
                    "MCP mcp_server_tool_call missing per-user env vars: server_id=%s missing=%s",
                    e.server_id,
                    e.missing,
                )
                return CallToolResult(
                    content=[TextContent(text=str(e), type="text")],
                    isError=True,
                )
            except BlockedPiiEntityError as e:
                verbose_logger.error(f"BlockedPiiEntityError in MCP tool call: {str(e)}")
                return CallToolResult(
                    content=[
                        TextContent(
                            text=f"Error: Blocked PII entity detected - {str(e)}",
                            type="text",
                        )
                    ],
                    isError=True,
                )
            except GuardrailRaisedException as e:
                verbose_logger.error(f"GuardrailRaisedException in MCP tool call: {str(e)}")
                return CallToolResult(
                    content=[TextContent(text=f"Error: Guardrail violation - {str(e)}", type="text")],
                    isError=True,
                )
            except HTTPException as e:
                verbose_logger.error(f"HTTPException in MCP tool call: {str(e)}")
                return CallToolResult(
                    content=[TextContent(text=f"Error: {str(e.detail)}", type="text")],
                    isError=True,
                )
            except MCPUpstreamAuthError as e:
                # The MCP session manager serializes handler exceptions as JSON-RPC errors, so a
                # mid-session tool call cannot emit a raw 401 + WWW-Authenticate the way the REST
                # call path and the connect-time preemptive check do. Return an explicit isError
                # naming the upstream status (at info level, not a traceback) so the client still
                # learns it must re-authenticate upstream and expected pass-through 401s don't spam.
                verbose_logger.info(f"Upstream auth failure calling MCP tool: HTTP {e.status_code}")
                return CallToolResult(
                    content=[
                        TextContent(
                            text=f"Error: upstream authentication required (HTTP {e.status_code})",
                            type="text",
                        )
                    ],
                    isError=True,
                )
            except Exception as e:
                verbose_logger.exception(f"MCP mcp_server_tool_call - error: {e}")
                return CallToolResult(
                    content=[TextContent(text=f"Error: {str(e)}", type="text")],
                    isError=True,
                )

            return response
        finally:
            _otel_reset_mcp_trace_carrier(_trace_token)
            if _session_reset_token is not None:
                active_mcp_session_var.reset(_session_reset_token)

    @server.list_prompts()
    async def list_prompts() -> List[Prompt]:
        """
        List all available prompts
        """
        from mcp.server.lowlevel.server import request_ctx

        req_ctx = request_ctx.get(None)
        _session_reset_token = None
        if req_ctx:
            _session_reset_token = active_mcp_session_var.set(req_ctx.session)

        try:
            # Get user authentication from context variable
            (
                user_api_key_auth,
                mcp_auth_header,
                mcp_servers,
                mcp_server_auth_headers,
                oauth2_headers,
                raw_headers,
                _client_ip,
            ) = await get_or_extract_auth_context()
            verbose_logger.debug(f"MCP list_prompts - User API Key Auth from context: {user_api_key_auth}")
            verbose_logger.debug(f"MCP list_prompts - MCP servers from context: {mcp_servers}")
            verbose_logger.debug(
                f"MCP list_prompts - MCP server auth headers: {list(mcp_server_auth_headers.keys()) if mcp_server_auth_headers else None}"
            )
            # Get mcp_servers from context variable
            verbose_logger.debug("MCP list_prompts - Calling _list_prompts")
            prompts = await _list_mcp_prompts(
                user_api_key_auth=user_api_key_auth,
                mcp_auth_header=mcp_auth_header,
                mcp_servers=mcp_servers,
                mcp_server_auth_headers=mcp_server_auth_headers,
                oauth2_headers=oauth2_headers,
                raw_headers=raw_headers,
            )
            verbose_logger.info(f"MCP list_prompts - Successfully returned {len(prompts)} prompts")
            return prompts
        except Exception as e:
            verbose_logger.exception(f"Error in list_prompts endpoint: {str(e)}")
            # Return empty list instead of failing completely
            # This prevents the HTTP stream from failing and allows the client to get a response
            return []
        finally:
            if _session_reset_token is not None:
                active_mcp_session_var.reset(_session_reset_token)

    @server.get_prompt()
    async def get_prompt(name: str, arguments: Optional[Dict[str, str]]) -> GetPromptResult:
        """
        Get a specific prompt with the provided arguments

        Args:
            name (str): Name of the prompt to get
            arguments (Dict[str, Any] | None): Arguments to pass to the prompt

        Returns:
            GetPromptResult: Getting prompt execution results
        """

        # Validate arguments
        from mcp.server.lowlevel.server import request_ctx

        req_ctx = request_ctx.get(None)
        _session_reset_token = None
        if req_ctx:
            _session_reset_token = active_mcp_session_var.set(req_ctx.session)

        try:
            (
                user_api_key_auth,
                mcp_auth_header,
                mcp_servers,
                mcp_server_auth_headers,
                oauth2_headers,
                raw_headers,
                _client_ip,
            ) = await get_or_extract_auth_context()

            verbose_logger.debug(f"MCP mcp_server_tool_call - User API Key Auth from context: {user_api_key_auth}")
            return await mcp_get_prompt(
                name=name,
                arguments=arguments,
                user_api_key_auth=user_api_key_auth,
                mcp_auth_header=mcp_auth_header,
                mcp_servers=mcp_servers,
                mcp_server_auth_headers=mcp_server_auth_headers,
                oauth2_headers=oauth2_headers,
                raw_headers=raw_headers,
            )
        finally:
            if _session_reset_token is not None:
                active_mcp_session_var.reset(_session_reset_token)

    @server.list_resources()
    async def list_resources() -> List[Resource]:
        """List all available resources."""
        from mcp.server.lowlevel.server import request_ctx

        req_ctx = request_ctx.get(None)
        _session_reset_token = None
        if req_ctx:
            _session_reset_token = active_mcp_session_var.set(req_ctx.session)

        try:
            (
                user_api_key_auth,
                mcp_auth_header,
                mcp_servers,
                mcp_server_auth_headers,
                oauth2_headers,
                raw_headers,
                _client_ip,
            ) = await get_or_extract_auth_context()
            verbose_logger.debug(f"MCP list_resources - User API Key Auth from context: {user_api_key_auth}")
            verbose_logger.debug(f"MCP list_resources - MCP servers from context: {mcp_servers}")
            verbose_logger.debug(
                f"MCP list_resources - MCP server auth headers: {list(mcp_server_auth_headers.keys()) if mcp_server_auth_headers else None}"
            )

            resources = await _list_mcp_resources(
                user_api_key_auth=user_api_key_auth,
                mcp_auth_header=mcp_auth_header,
                mcp_servers=mcp_servers,
                mcp_server_auth_headers=mcp_server_auth_headers,
                oauth2_headers=oauth2_headers,
                raw_headers=raw_headers,
            )
            verbose_logger.info(f"MCP list_resources - Successfully returned {len(resources)} resources")
            return resources
        except Exception as e:
            verbose_logger.exception(f"Error in list_resources endpoint: {str(e)}")
            return []
        finally:
            if _session_reset_token is not None:
                active_mcp_session_var.reset(_session_reset_token)

    @server.list_resource_templates()
    async def list_resource_templates() -> List[ResourceTemplate]:
        """List all available resource templates."""
        from mcp.server.lowlevel.server import request_ctx

        req_ctx = request_ctx.get(None)
        _session_reset_token = None
        if req_ctx:
            _session_reset_token = active_mcp_session_var.set(req_ctx.session)

        try:
            (
                user_api_key_auth,
                mcp_auth_header,
                mcp_servers,
                mcp_server_auth_headers,
                oauth2_headers,
                raw_headers,
                _client_ip,
            ) = await get_or_extract_auth_context()
            verbose_logger.debug(f"MCP list_resource_templates - User API Key Auth from context: {user_api_key_auth}")
            verbose_logger.debug(f"MCP list_resource_templates - MCP servers from context: {mcp_servers}")
            verbose_logger.debug(
                f"MCP list_resource_templates - MCP server auth headers: {list(mcp_server_auth_headers.keys()) if mcp_server_auth_headers else None}"
            )

            resource_templates = await _list_mcp_resource_templates(
                user_api_key_auth=user_api_key_auth,
                mcp_auth_header=mcp_auth_header,
                mcp_servers=mcp_servers,
                mcp_server_auth_headers=mcp_server_auth_headers,
                oauth2_headers=oauth2_headers,
                raw_headers=raw_headers,
            )
            verbose_logger.info(
                f"MCP list_resource_templates - Successfully returned {len(resource_templates)} resource templates"
            )
            return resource_templates
        except Exception as e:
            verbose_logger.exception(f"Error in list_resource_templates endpoint: {str(e)}")
            return []
        finally:
            if _session_reset_token is not None:
                active_mcp_session_var.reset(_session_reset_token)

    @server.read_resource()
    async def read_resource(url: AnyUrl) -> list[ReadResourceContents]:
        from mcp.server.lowlevel.server import request_ctx

        req_ctx = request_ctx.get(None)
        _session_reset_token = None
        if req_ctx:
            _session_reset_token = active_mcp_session_var.set(req_ctx.session)

        try:
            (
                user_api_key_auth,
                mcp_auth_header,
                mcp_servers,
                mcp_server_auth_headers,
                oauth2_headers,
                raw_headers,
                _client_ip,
            ) = await get_or_extract_auth_context()

            read_resource_result = await mcp_read_resource(
                url=url,
                user_api_key_auth=user_api_key_auth,
                mcp_auth_header=mcp_auth_header,
                mcp_servers=mcp_servers,
                mcp_server_auth_headers=mcp_server_auth_headers,
                oauth2_headers=oauth2_headers,
                raw_headers=raw_headers,
            )

            return _normalize_resource_contents(read_resource_result.contents)
        finally:
            if _session_reset_token is not None:
                active_mcp_session_var.reset(_session_reset_token)

    ########################################################
    ############ End of MCP Server Routes ##################
    ########################################################

    ########################################################
    ############ Helper Functions ##########################
    ########################################################

    async def _get_allowed_mcp_servers_from_mcp_server_names(
        mcp_servers: Optional[List[str]],
        allowed_mcp_servers: List[MCPServer],
    ) -> List[MCPServer]:
        """
        Get the filtered MCP servers from the MCP server names.

        Fails closed when ``mcp_servers`` is explicitly provided (path- or
        header-derived) but none of the names resolve to a server alias or
        access group the caller can access. The previous behavior returned
        the full ``allowed_mcp_servers`` set, which silently widened scope
        when a client targeted ``/mcp/<unknown>/`` and made URL/header
        namespacing appear to work when it did not.
        """

        filtered_server: dict[str, MCPServer] = {}
        # Filter servers based on mcp_servers parameter if provided
        if mcp_servers is not None:
            for server_or_group in mcp_servers:
                server_name_matched = False

                for server in allowed_mcp_servers:
                    if server:
                        match_list = [s.lower() for s in iter_known_server_prefixes(server) if s]

                        if server_or_group.lower() in match_list:
                            filtered_server[server.server_id] = server
                            server_name_matched = True
                            break

                if not server_name_matched:
                    try:
                        access_group_server_ids = await MCPRequestHandler._get_mcp_servers_from_access_groups(
                            [server_or_group]
                        )
                        # Only include servers that the user has access to
                        for server_id in access_group_server_ids:
                            for server in allowed_mcp_servers:
                                if server_id == server.server_id:
                                    filtered_server[server.server_id] = server
                    except Exception as e:
                        verbose_logger.debug(f"Could not resolve '{server_or_group}' as access group: {e}")

        if filtered_server:
            return list(filtered_server.values())

        if mcp_servers is not None:
            # Caller asked for a specific scope but nothing resolved. Fail
            # closed so URL/header namespacing cannot silently fall back to
            # the caller's full allowed-server set.
            verbose_logger.debug(
                "MCP scope filter resolved to no servers for requested names %s; returning empty list (fail-closed).",
                mcp_servers,
            )
            return []

        return allowed_mcp_servers

    def _tool_name_matches(tool_name: str, filter_list: List[str]) -> bool:
        """
        Check if a tool name matches any name in the filter list.

        Checks both the full tool name and unprefixed version (without server prefix).
        This allows users to configure simple tool names regardless of prefixing.
        Comparison is case-insensitive to handle OpenAPI operationIds that may be in camelCase.

        Args:
            tool_name: The tool name to check (may be prefixed like "server-tool_name")
            filter_list: List of tool names to match against

        Returns:
            True if the tool name (prefixed or unprefixed) is in the filter list
        """
        from litellm.proxy._experimental.mcp_server.utils import (
            split_server_prefix_from_name,
        )

        # Normalize filter list to lowercase for case-insensitive comparison
        filter_list_lower = [f.lower() for f in filter_list]

        if tool_name.lower() in filter_list_lower:
            return True

        # Check if the unprefixed name is in the list (case-insensitive)
        unprefixed_name, _ = split_server_prefix_from_name(tool_name)
        return unprefixed_name.lower() in filter_list_lower

    def filter_tools_by_allowed_tools(
        tools: List[MCPTool],
        mcp_server: MCPServer,
    ) -> List[MCPTool]:
        """
        Filter tools by allowed/disallowed tools configuration.

        If allowed_tools is set, only tools in that list are returned.
        If disallowed_tools is set, tools in that list are excluded.
        Tool names are matched with and without server prefixes for flexibility.

        Args:
            tools: List of tools to filter
            mcp_server: Server configuration with allowed_tools/disallowed_tools

        Returns:
            Filtered list of tools
        """
        from litellm.proxy._experimental.mcp_server.utils import (
            server_applies_tool_allowlist,
        )

        tools_to_return = tools

        # Filter by allowed_tools (whitelist)
        if server_applies_tool_allowlist(mcp_server):
            if not mcp_server.allowed_tools:
                return []
            tools_to_return = [tool for tool in tools if _tool_name_matches(tool.name, mcp_server.allowed_tools)]

        # Filter by disallowed_tools (blacklist)
        if mcp_server.disallowed_tools:
            tools_to_return = [
                tool for tool in tools_to_return if not _tool_name_matches(tool.name, mcp_server.disallowed_tools)
            ]

        return tools_to_return

    def apply_tool_overrides(
        tools: List[MCPTool],
        mcp_server: MCPServer,
    ) -> List[MCPTool]:
        """Apply admin-configured display name/description overrides to tools.

        Overrides are keyed by the unprefixed tool name, same convention as
        allowed_tools configuration.
        """
        display_name_map = mcp_server.tool_name_to_display_name or {}
        description_map = mcp_server.tool_name_to_description or {}
        if not display_name_map and not description_map:
            return tools

        for tool in tools:
            unprefixed, _ = split_server_prefix_from_name(tool.name)
            lookup_key = unprefixed or tool.name
            if lookup_key in display_name_map:
                tool.name = display_name_map[lookup_key]
            if lookup_key in description_map:
                tool.description = description_map[lookup_key]
        return tools

    def _get_client_ip_from_context() -> Optional[str]:
        """
        Extract client_ip from auth context.
        Returns None if context not set (caller should handle this as "no IP filtering").
        """
        try:
            auth_user = auth_context_var.get()
            if auth_user and isinstance(auth_user, MCPAuthenticatedUser):
                return auth_user.client_ip
        except Exception:
            pass
        return None

    async def _get_allowed_mcp_servers(
        user_api_key_auth: Optional[UserAPIKeyAuth],
        mcp_servers: Optional[List[str]],
        client_ip: Optional[str] = None,
    ) -> List[MCPServer]:
        """Return allowed MCP servers for a request after applying filters.

        Args:
            user_api_key_auth: The authenticated user's API key info.
            mcp_servers: Optional list of server names to filter to.
            client_ip: Client IP for IP-based access control. If None, falls back to
                      auth context. Pass explicitly from request handlers for safety.
        Note: If client_ip is None and auth context is not set, IP filtering is skipped.
              This is intentional for internal callers but may indicate a bug if called
              from a request handler without proper context setup.
        """
        # Use explicit client_ip if provided, otherwise try auth context
        if client_ip is None:
            client_ip = _get_client_ip_from_context()
            if client_ip is None:
                verbose_logger.debug(
                    "MCP _get_allowed_mcp_servers called without client_ip and no auth context. "
                    "IP filtering will be skipped. This is expected for internal calls."
                )

        allowed_mcp_server_ids = await global_mcp_server_manager.get_allowed_mcp_servers(user_api_key_auth)
        (
            allowed_mcp_server_ids,
            _ip_blocked,
        ) = global_mcp_server_manager.filter_server_ids_by_ip_with_info(allowed_mcp_server_ids, client_ip)
        verbose_logger.debug(
            "MCP IP filter: client_ip=%s, allowed_server_ids=%s",
            client_ip,
            allowed_mcp_server_ids,
        )
        if _ip_blocked > 0:
            verbose_logger.debug(
                "MCP IP filtering: %d server(s) are not accessible from client IP %s "
                "because they are restricted to internal networks. "
                "No tools from those servers will be returned. "
                "To expose a server externally, set 'available_on_public_internet: true' "
                "in its configuration.",
                _ip_blocked,
                client_ip,
            )
        allowed_mcp_servers: List[MCPServer] = []
        for allowed_mcp_server_id in allowed_mcp_server_ids:
            mcp_server = global_mcp_server_manager.get_mcp_server_by_id(allowed_mcp_server_id)
            if mcp_server is not None:
                # Apply the request-time oauth2_flow backstop for legacy null rows.
                mcp_server = MCPServerManager.resolve_oauth2_flow_for_request(mcp_server)
                allowed_mcp_servers.append(mcp_server)

        if mcp_servers is not None:
            allowed_mcp_servers = await _get_allowed_mcp_servers_from_mcp_server_names(
                mcp_servers=mcp_servers,
                allowed_mcp_servers=allowed_mcp_servers,
            )

        return allowed_mcp_servers

    def _client_has_per_server_auth_header(
        server: MCPServer,
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]],
    ) -> bool:
        """True if the request carries a per-server ``x-mcp-{alias}-authorization``
        header for this server. This is the multi-server binding: it names one
        upstream, so it is unambiguously the caller's upstream token regardless of
        auth mode (never the LiteLLM admission credential).

        Resolves through the same ``lookup_mcp_server_auth_in_headers`` egress uses, so
        the connect gate and egress agree on which per-server header names match: a
        dashboard client sends ``x-mcp-{sanitize_mcp_alias_for_header(alias)}-authorization``,
        and matching only the raw alias here would 401 a token egress would forward.
        """
        if not mcp_server_auth_headers:
            return False
        from litellm.proxy._experimental.mcp_server.utils import (
            lookup_mcp_server_auth_in_headers,
        )

        server_headers = lookup_mcp_server_auth_in_headers(
            mcp_server_auth_headers, alias=server.alias, server_name=server.server_name
        )
        if isinstance(server_headers, str):
            return bool(server_headers.strip())
        if isinstance(server_headers, dict):
            return any(isinstance(hk, str) and hk.lower() == "authorization" for hk in server_headers)
        return False

    def _client_has_passthrough_authorization(
        server: MCPServer,
        oauth2_headers: Optional[Dict[str, str]],
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]],
    ) -> bool:
        """True if the incoming request already carries an ``Authorization``
        header the gateway will forward to this pass-through server.

        The client may supply the bearer as either the top-level
        ``Authorization`` header (surfaced via ``oauth2_headers``) or a
        per-server ``x-mcp-auth-<alias>`` style header (surfaced via
        ``mcp_server_auth_headers``). Either form skips the pre-emptive 401.
        """
        if oauth2_headers:
            for k in oauth2_headers.keys():
                if k.lower() == "authorization":
                    return True
        return _client_has_per_server_auth_header(server, mcp_server_auth_headers)

    async def _get_user_oauth_extra_headers_from_db(
        server: MCPServer,
        user_api_key_auth: Optional[UserAPIKeyAuth],
        prefetched_creds: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, str]]:
        """Stored OAuth2 token for (user, server) as an ``Authorization: Bearer`` header, or None.

        Thin wrapper over ``resolve_user_oauth_access_token`` (Redis cache, else DB + refresh);
        ``prefetched_creds`` skips the per-server Redis/DB lookups for the batch path.
        """
        if server.auth_type != MCPAuth.oauth2 or user_api_key_auth is None:
            return None
        from litellm.proxy._experimental.mcp_server.db import (  # noqa: PLC0415
            resolve_user_oauth_access_token,
        )

        token = await resolve_user_oauth_access_token(
            getattr(user_api_key_auth, "user_id", None), server, prefetched_creds
        )
        return {"Authorization": f"Bearer {token}"} if token else None

    async def _prefetch_oauth_creds_for_user(
        user_api_key_auth: Optional[UserAPIKeyAuth],
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch all OAuth2 credentials for the user in one DB query.

        Returns a dict keyed by server_id to avoid N+1 queries in asyncio.gather loops.
        """
        user_id = getattr(user_api_key_auth, "user_id", None) if user_api_key_auth else None
        if not user_id:
            return {}
        try:
            from litellm.proxy._experimental.mcp_server.db import (  # noqa: PLC0415
                list_user_oauth_credentials,
            )
            from litellm.proxy.utils import get_prisma_client_or_throw  # noqa: PLC0415

            prisma_client = get_prisma_client_or_throw(
                "Database not connected. Connect a database to use OAuth2 MCP tools."
            )
            creds = await list_user_oauth_credentials(prisma_client, user_id)
            return {c["server_id"]: c for c in creds if "server_id" in c}
        except Exception as e:
            verbose_logger.warning(f"_prefetch_oauth_creds_for_user: failed to prefetch for user={user_id}: {e}")
            return {}

    def _prepare_mcp_server_headers(
        server: MCPServer,
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]],
        mcp_auth_header: Optional[str],
        oauth2_headers: Optional[Dict[str, str]],
        raw_headers: Optional[Dict[str, str]],
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
        scope_servers: Optional[list[MCPServer]] = None,
    ) -> Tuple[Optional[Union[Dict[str, str], str]], Optional[Dict[str, str]]]:
        """Build auth and extra headers for a server.

        ``scope_servers`` is the full server list a fan-out handler iterates. Passing it lets the
        client-forwarded token modes withhold the caller's request-wide ``Authorization`` when
        another server in the scope would also receive it (``_caller_authorization_fans_out``);
        explicitly-addressed operations leave it None. Per-server ``x-mcp-{alias}-authorization``
        headers are unaffected — they bind one token to one server and are the multi-server shape.
        """
        server_auth_header: Optional[Union[Dict[str, str], str]] = None
        if mcp_server_auth_headers:
            from litellm.proxy._experimental.mcp_server.utils import (
                lookup_mcp_server_auth_in_headers,
            )

            server_auth_header = lookup_mcp_server_auth_in_headers(
                mcp_server_auth_headers,
                alias=server.alias,
                server_name=server.server_name,
            )

        extra_headers: Optional[Dict[str, str]] = None
        is_client_forwarded_mode = server.is_true_passthrough or server.is_oauth_delegate
        # In a multi-server listing scope the request-wide Authorization can only carry one token,
        # so it is withheld from a client-forwarded server when another server in scope also consumes
        # it (RFC 9700 cross-resource replay); such scopes must bind per-server via
        # x-mcp-{alias}-authorization. The decision is computed once so BOTH the forwarding branch and
        # the extra_headers copy loop below honor it — otherwise a server that lists Authorization in
        # extra_headers would re-copy the withheld bearer from raw_headers and replay it anyway.
        withhold_forwarded_authorization = is_client_forwarded_mode and _caller_authorization_fans_out(
            server, scope_servers
        )
        if server.auth_type == MCPAuth.oauth2:
            # For OAuth2 M2M servers, upstream Authorization must come from
            # client_credentials token fetch, never from caller headers.
            if server.has_client_credentials:
                extra_headers = None
            else:
                # Copy to avoid mutating the original dict (important for parallel fetching)
                extra_headers = oauth2_headers.copy() if oauth2_headers else None
                # Migrated authorization_code: the v2 resolver injects the stored per-user
                # token, so drop the caller-forwarded Authorization (apply-if-absent would
                # otherwise let it shadow the resolved token). Delegate keeps it. Centralized
                # via _should_strip_caller_authorization to match _call_regular_mcp_tool.
                if extra_headers and _should_strip_caller_authorization(
                    mcp_server=server,
                    raw_headers=raw_headers,
                    user_api_key_auth=user_api_key_auth,
                ):
                    extra_headers = _without_authorization(extra_headers)
        elif is_client_forwarded_mode:
            if not withhold_forwarded_authorization:
                extra_headers = _client_forwarded_authorization_headers(
                    mcp_server=server,
                    oauth2_headers=oauth2_headers,
                    raw_headers=raw_headers,
                    user_api_key_auth=user_api_key_auth,
                )

        if server.extra_headers and raw_headers:
            if extra_headers is None:
                extra_headers = {}

            normalized_raw_headers = {str(k).lower(): v for k, v in raw_headers.items() if isinstance(k, str)}

            # Centralized strip decision shared with
            # ``MCPServerManager._call_regular_mcp_tool`` so the two
            # code paths cannot drift on this security-sensitive choice.
            # See ``_should_strip_caller_authorization`` for the rules.
            strip_caller_authorization = _should_strip_caller_authorization(
                mcp_server=server,
                raw_headers=raw_headers,
                user_api_key_auth=user_api_key_auth,
            )

            for header in server.extra_headers:
                if not isinstance(header, str):
                    continue
                if header.lower() == "authorization" and (
                    strip_caller_authorization or withhold_forwarded_authorization
                ):
                    continue
                header_value = normalized_raw_headers.get(header.lower())
                if header_value is None:
                    continue
                extra_headers[header] = header_value

        # Reset to None if no headers were actually added
        if extra_headers is not None and len(extra_headers) == 0:
            extra_headers = None

        if server_auth_header is None:
            server_auth_header = mcp_auth_header

        return server_auth_header, extra_headers

    def _merge_gateway_initialize_instructions(
        allowed_mcp_servers: List[MCPServer],
    ) -> Optional[str]:
        """YAML/DB override, else upstream text (prefetch on init, or list_tools / health_check / call_tool cache)."""
        if not allowed_mcp_servers:
            return None

        texts: List[Tuple[str, str]] = []
        for server in allowed_mcp_servers:
            label = server.alias or server.server_name or server.name or server.server_id or "mcp"
            if server.instructions and server.instructions.strip():
                texts.append((label, server.instructions.strip()))
                continue
            if server.spec_path:
                continue
            cached = global_mcp_server_manager._upstream_initialize_instructions_by_server_id.get(server.server_id)
            if cached and cached.strip():
                texts.append((label, cached.strip()))

        if not texts:
            return None
        if len(texts) == 1:
            return texts[0][1]
        return "\n\n---\n\n".join(f"[{lbl}]\n{txt}" for lbl, txt in texts)

    @contextlib.asynccontextmanager
    async def _gateway_initialize_instructions_request_scope(
        user_api_key_auth: Optional[UserAPIKeyAuth],
        mcp_servers: Optional[List[str]],
        client_ip: Optional[str],
        scoped_server_endpoint: bool = False,
    ) -> AsyncIterator[None]:
        allowed = await _get_allowed_mcp_servers(
            user_api_key_auth=user_api_key_auth,
            mcp_servers=mcp_servers,
            client_ip=client_ip,
        )
        if allowed:
            # return_exceptions=True: a per-server probe failure (incl. CancelledError
            # bubbled from anyio task group teardown on connection refused) must not
            # cancel sibling probes or 500 the gateway initialize request.
            await asyncio.gather(
                *[
                    global_mcp_server_manager._ensure_upstream_initialize_instructions_cached(s)
                    for s in allowed
                    if s is not None
                ],
                return_exceptions=True,
            )
        merged = _merge_gateway_initialize_instructions(allowed_mcp_servers=allowed)
        scoped_server_name = None
        if scoped_server_endpoint and len(allowed) == 1:
            scoped_server = allowed[0]
            scoped_server_name = (
                scoped_server.alias or scoped_server.server_name or scoped_server.name or scoped_server.server_id
            )
        instructions_token = _mcp_gateway_initialize_instructions.set(merged)
        server_name_token = _mcp_gateway_server_name.set(scoped_server_name)
        try:
            yield
        finally:
            _mcp_gateway_initialize_instructions.reset(instructions_token)
            _mcp_gateway_server_name.reset(server_name_token)

    def _aggregate_server_key(server: MCPServer) -> str:
        return str(
            getattr(server, "server_name", None)
            or getattr(server, "alias", None)
            or getattr(server, "name", None)
            or "unknown"
        )

    async def _get_tools_from_mcp_servers(
        user_api_key_auth: Optional[UserAPIKeyAuth],
        mcp_auth_header: Optional[str],
        mcp_servers: Optional[List[str]],
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]] = None,
        oauth2_headers: Optional[Dict[str, str]] = None,
        raw_headers: Optional[Dict[str, str]] = None,
        log_list_tools_to_spendlogs: bool = False,
        list_tools_log_source: Optional[str] = None,
        litellm_trace_id: Optional[str] = None,
        request_tags: Optional[list[str]] = None,
        client_ip: Optional[str] = None,
    ) -> AggregateToolListing:
        """
        Helper method to fetch tools from MCP servers based on server filtering criteria.

        Args:
            user_api_key_auth: User authentication info for access control
            mcp_auth_header: Optional auth header for MCP server (deprecated)
            mcp_servers: Optional list of server names/aliases to filter by
            mcp_server_auth_headers: Optional dict of server-specific auth headers
            oauth2_headers: Optional dict of oauth2 headers

        Returns:
            AggregateToolListing: Combined tools from filtered servers plus each server's
            classified listing outcome
        """
        if not MCP_AVAILABLE:
            return AggregateToolListing(tools=[], outcomes={})

        list_tools_start_time = datetime.now()
        litellm_logging_obj: Optional[LiteLLMLoggingObj] = None
        list_tools_request_data: Dict[str, Any] = {}

        if log_list_tools_to_spendlogs:
            # This is intentionally minimal: only async_success_handler / post_call_failure_hook
            rules_obj = Rules()
            list_tools_call_id = str(uuid.uuid4())
            # Derive trace_id from raw_headers when not explicitly passed (same as A2A / MCP call_tool)
            effective_litellm_trace_id = litellm_trace_id or get_chain_id_from_headers(raw_headers)
            spend_logs_metadata: Dict[str, Any] = {
                "mcp_operation": "list_tools",
            }
            if isinstance(list_tools_log_source, str):
                spend_logs_metadata["source"] = list_tools_log_source
            if isinstance(mcp_servers, list):
                spend_logs_metadata["requested_mcp_servers"] = mcp_servers

            list_tools_request_data = {
                "model": "MCP: list_tools",
                "call_type": CallTypes.list_mcp_tools.value,
                "litellm_call_id": list_tools_call_id,
                "litellm_trace_id": effective_litellm_trace_id,
                "metadata": {
                    "spend_logs_metadata": spend_logs_metadata,
                    **({"tags": request_tags} if request_tags else {}),
                },
                # Provide a small input payload for standard logging
                "input": [
                    {
                        "role": "system",
                        "content": {
                            "mcp_operation": "list_tools",
                            "requested_mcp_servers": mcp_servers,
                        },
                    }
                ],
            }

            # Attach user identifiers using the standard helper
            if user_api_key_auth is not None:
                LiteLLMProxyRequestSetup.add_user_api_key_auth_to_request_metadata(
                    data=list_tools_request_data,
                    user_api_key_dict=user_api_key_auth,
                    _metadata_variable_name="metadata",
                )

                user_identifier = getattr(user_api_key_auth, "end_user_id", None) or getattr(
                    user_api_key_auth, "user_id", None
                )
                if user_identifier:
                    list_tools_request_data["user"] = user_identifier

            try:
                litellm_logging_obj, _ = function_setup(
                    original_function="list_mcp_tools",
                    rules_obj=rules_obj,
                    start_time=list_tools_start_time,
                    **list_tools_request_data,
                )
                if litellm_logging_obj:
                    litellm_logging_obj.call_type = CallTypes.list_mcp_tools.value
                    litellm_logging_obj.model = "MCP: list_tools"
            except Exception as logging_error:
                verbose_logger.debug("Failed to initialize logging for MCP list_tools: %s", logging_error)
                litellm_logging_obj = None

        try:
            allowed_mcp_servers = await _get_allowed_mcp_servers(
                user_api_key_auth=user_api_key_auth,
                mcp_servers=mcp_servers,
                client_ip=client_ip,
            )

            # Pre-fetch OAuth credentials only when at least one server uses OAuth2,
            # to avoid an unnecessary DB round-trip on requests with no OAuth2 MCP servers.
            _has_oauth2_server = any(getattr(s, "auth_type", None) == MCPAuth.oauth2 for s in allowed_mcp_servers)
            _prefetched_oauth_creds = (
                await _prefetch_oauth_creds_for_user(user_api_key_auth) if _has_oauth2_server else {}
            )

            async def _fetch_and_filter_server_tools(
                server: MCPServer,
            ) -> "tuple[List[MCPTool], ServerOutcome]":
                """Fetch and filter tools from a single server, classifying any failure into that
                server's outcome so the aggregate can keep serving the healthy subset without a
                broken server masquerading as an empty one."""
                if server is None:
                    return [], ServerListOk(tool_count=0)

                server_auth_header, extra_headers = _prepare_mcp_server_headers(
                    server=server,
                    mcp_server_auth_headers=mcp_server_auth_headers,
                    mcp_auth_header=mcp_auth_header,
                    oauth2_headers=oauth2_headers,
                    raw_headers=raw_headers,
                    user_api_key_auth=user_api_key_auth,
                    scope_servers=allowed_mcp_servers,
                )

                # Prefer server-stored per-user OAuth when configured, so a stale
                # Authorization header from the MCP client cannot override Redis/DB
                # (same issue as call_tool in mcp_server_manager: VS Code caches tokens).
                from litellm.proxy._experimental.mcp_server.outbound_credentials.adapter import (  # noqa: PLC0415
                    to_server_spec,
                )

                # A server migrated to the v2 resolver gets its token from the resolver at connect
                # time; building it here would double-resolve and be shadowed by the v2 graft. The
                # preemptive 401 already challenged a missing token, so one exists for the connect.
                migrated_to_v2 = to_server_spec(server) is not None
                if (
                    not migrated_to_v2
                    and server.auth_type == MCPAuth.oauth2
                    and getattr(server, "needs_user_oauth_token", False)
                    and user_api_key_auth is not None
                ):
                    db_headers = await _get_user_oauth_extra_headers_from_db(
                        server,
                        user_api_key_auth,
                        prefetched_creds=_prefetched_oauth_creds,
                    )
                    if db_headers:
                        extra_headers = db_headers

                # If still no OAuth2 token, fall back to pre-fetched creds (non-stale-client path)
                elif not migrated_to_v2 and extra_headers is None and server.auth_type == MCPAuth.oauth2:
                    extra_headers = await _get_user_oauth_extra_headers_from_db(
                        server,
                        user_api_key_auth,
                        prefetched_creds=_prefetched_oauth_creds,
                    )

                try:
                    tools = await global_mcp_server_manager._get_tools_from_server(
                        server=server,
                        mcp_auth_header=server_auth_header,
                        extra_headers=extra_headers,
                        add_prefix=True,  # Always add server prefix
                        raw_headers=raw_headers,
                        user_api_key_auth=user_api_key_auth,
                        oauth2_headers=oauth2_headers,
                    )
                    filtered_tools = filter_tools_by_allowed_tools(tools, server)

                    filtered_tools = await filter_tools_by_key_team_permissions(
                        tools=filtered_tools,
                        server_id=server.server_id,
                        user_api_key_auth=user_api_key_auth,
                    )

                    # Apply display-name/description overrides last so that
                    # permission filtering always works against original names.
                    filtered_tools = apply_tool_overrides(filtered_tools, server)

                    verbose_logger.debug(
                        f"Successfully fetched {len(tools)} tools from server {server.name}, {len(filtered_tools)} after filtering"
                    )
                    return filtered_tools, ServerListOk(tool_count=len(filtered_tools))
                except MCPUpstreamAuthError as e:
                    # Absorb so one unauthenticated server does not empty every other server's
                    # tools. Surfacing the upstream 401 to the client as a re-auth challenge is
                    # intentionally not done here: raising from this list handler cannot produce a
                    # 401 + WWW-Authenticate (the MCP session manager serializes it as a JSON-RPC
                    # error). Single-server routes surface it via the request-scope preemptive
                    # check in _raise_preemptive_401_for_unauthenticated_servers instead.
                    verbose_logger.debug(f"MCP list_tools: omitting {server.name}; it needs upstream auth")
                    return [], classify_list_exception(e)
                except Exception as e:
                    verbose_logger.exception(f"Error getting tools from server {server.name}: {str(e)}")
                    return [], classify_list_exception(e)

            # Fetch tools from all servers in parallel
            tasks = [_fetch_and_filter_server_tools(server) for server in allowed_mcp_servers]
            results = await asyncio.gather(*tasks)

            # Flatten results into single list
            all_tools: List[MCPTool] = [tool for tools, _ in results for tool in tools]
            server_outcomes: Dict[str, ServerOutcome] = {
                _aggregate_server_key(server): outcome
                for server, (_, outcome) in zip(allowed_mcp_servers, results)
                if server is not None
            }

            # If logging is enabled, enrich spend_logs_metadata with counts
            if litellm_logging_obj:
                per_server_tool_counts: Dict[str, int] = {
                    _aggregate_server_key(server): len(server_tools)
                    for server, (server_tools, _) in zip(allowed_mcp_servers, results)
                    if server is not None
                }

                metadata_dict = litellm_logging_obj.model_call_details.get("metadata")
                if isinstance(metadata_dict, dict):
                    spend_meta = metadata_dict.get("spend_logs_metadata")
                    if not isinstance(spend_meta, dict):
                        spend_meta = {}
                        metadata_dict["spend_logs_metadata"] = spend_meta
                    spend_meta["allowed_server_count"] = len(allowed_mcp_servers)
                    spend_meta["tool_count_total"] = len(all_tools)
                    spend_meta["per_server_tool_counts"] = per_server_tool_counts
                    spend_meta["per_server_list_outcomes"] = {
                        key: outcome_wire_value(outcome) for key, outcome in server_outcomes.items()
                    }

                end_time = datetime.now()
                try:
                    await litellm_logging_obj.async_success_handler(
                        result=[
                            tool.model_dump(mode="json") if isinstance(tool, MCPTool) else tool for tool in all_tools
                        ],
                        start_time=list_tools_start_time,
                        end_time=end_time,
                    )
                except Exception as log_exc:
                    # list_tools responses must not be dropped due to non-blocking
                    # observability/serialization failures.
                    verbose_logger.warning(
                        "MCP list_tools success logging failed (continuing): %s",
                        log_exc,
                    )

            verbose_logger.info(f"Successfully fetched {len(all_tools)} tools total from all MCP servers")

            return AggregateToolListing(tools=all_tools, outcomes=server_outcomes)
        except Exception as e:
            # Only fire failure hook if logging was requested for this list-tools execution
            if log_list_tools_to_spendlogs and user_api_key_auth is not None:
                try:
                    from litellm.proxy.proxy_server import proxy_logging_obj

                    if proxy_logging_obj:
                        traceback_str = traceback.format_exc(limit=MAXIMUM_TRACEBACK_LINES_TO_LOG)
                        await proxy_logging_obj.post_call_failure_hook(
                            request_data=list_tools_request_data or {},
                            original_exception=e,
                            user_api_key_dict=user_api_key_auth,
                            route="/mcp/list_tools",
                            traceback_str=traceback_str,
                        )
                except Exception:
                    verbose_logger.debug("Failed to log MCP list_tools failure via post_call_failure_hook")
            raise

    async def _get_prompts_from_mcp_servers(
        user_api_key_auth: Optional[UserAPIKeyAuth],
        mcp_auth_header: Optional[str],
        mcp_servers: Optional[List[str]],
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]] = None,
        oauth2_headers: Optional[Dict[str, str]] = None,
        raw_headers: Optional[Dict[str, str]] = None,
    ) -> List[Prompt]:
        """
        Helper method to fetch prompt from MCP servers based on server filtering criteria.

        Args:
            user_api_key_auth: User authentication info for access control
            mcp_auth_header: Optional auth header for MCP server (deprecated)
            mcp_servers: Optional list of server names/aliases to filter by
            mcp_server_auth_headers: Optional dict of server-specific auth headers
            oauth2_headers: Optional dict of oauth2 headers

        Returns:
            List[Prompt]: Combined list of prompts from filtered servers
        """
        if not MCP_AVAILABLE:
            return []

        allowed_mcp_servers = await _get_allowed_mcp_servers(
            user_api_key_auth=user_api_key_auth,
            mcp_servers=mcp_servers,
        )

        # Get prompts from each allowed server
        all_prompts = []
        for server in allowed_mcp_servers:
            if server is None:
                continue

            server_auth_header, extra_headers = _prepare_mcp_server_headers(
                server=server,
                mcp_server_auth_headers=mcp_server_auth_headers,
                mcp_auth_header=mcp_auth_header,
                oauth2_headers=oauth2_headers,
                raw_headers=raw_headers,
                user_api_key_auth=user_api_key_auth,
                scope_servers=allowed_mcp_servers,
            )

            try:
                prompts = await global_mcp_server_manager.get_prompts_from_server(
                    server=server,
                    mcp_auth_header=server_auth_header,
                    extra_headers=extra_headers,
                    add_prefix=True,  # Always add server prefix
                    raw_headers=raw_headers,
                )

                all_prompts.extend(prompts)

                verbose_logger.debug(f"Successfully fetched {len(prompts)} prompts from server {server.name}")
            except Exception as e:
                verbose_logger.exception(f"Error getting prompts from server {server.name}: {str(e)}")
                # Continue with other servers instead of failing completely

        verbose_logger.info(f"Successfully fetched {len(all_prompts)} prompts total from all MCP servers")

        return all_prompts

    async def _get_resources_from_mcp_servers(
        user_api_key_auth: Optional[UserAPIKeyAuth],
        mcp_auth_header: Optional[str],
        mcp_servers: Optional[List[str]],
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]] = None,
        oauth2_headers: Optional[Dict[str, str]] = None,
        raw_headers: Optional[Dict[str, str]] = None,
    ) -> List[Resource]:
        """Fetch resources from allowed MCP servers."""

        if not MCP_AVAILABLE:
            return []

        allowed_mcp_servers = await _get_allowed_mcp_servers(
            user_api_key_auth=user_api_key_auth,
            mcp_servers=mcp_servers,
        )

        all_resources: List[Resource] = []
        for server in allowed_mcp_servers:
            if server is None:
                continue

            server_auth_header, extra_headers = _prepare_mcp_server_headers(
                server=server,
                mcp_server_auth_headers=mcp_server_auth_headers,
                mcp_auth_header=mcp_auth_header,
                oauth2_headers=oauth2_headers,
                raw_headers=raw_headers,
                user_api_key_auth=user_api_key_auth,
                scope_servers=allowed_mcp_servers,
            )

            try:
                resources = await global_mcp_server_manager.get_resources_from_server(
                    server=server,
                    mcp_auth_header=server_auth_header,
                    extra_headers=extra_headers,
                    add_prefix=True,  # Always add server prefix
                    raw_headers=raw_headers,
                )
                all_resources.extend(resources)

                verbose_logger.debug(f"Successfully fetched {len(resources)} resources from server {server.name}")
            except Exception as e:
                verbose_logger.exception(f"Error getting resources from server {server.name}: {str(e)}")

        verbose_logger.info(f"Successfully fetched {len(all_resources)} resources total from all MCP servers")

        return all_resources

    async def _get_resource_templates_from_mcp_servers(
        user_api_key_auth: Optional[UserAPIKeyAuth],
        mcp_auth_header: Optional[str],
        mcp_servers: Optional[List[str]],
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]] = None,
        oauth2_headers: Optional[Dict[str, str]] = None,
        raw_headers: Optional[Dict[str, str]] = None,
    ) -> List[ResourceTemplate]:
        """Fetch resource templates from allowed MCP servers."""

        if not MCP_AVAILABLE:
            return []

        allowed_mcp_servers = await _get_allowed_mcp_servers(
            user_api_key_auth=user_api_key_auth,
            mcp_servers=mcp_servers,
        )

        all_resource_templates: List[ResourceTemplate] = []
        for server in allowed_mcp_servers:
            if server is None:
                continue

            server_auth_header, extra_headers = _prepare_mcp_server_headers(
                server=server,
                mcp_server_auth_headers=mcp_server_auth_headers,
                mcp_auth_header=mcp_auth_header,
                oauth2_headers=oauth2_headers,
                raw_headers=raw_headers,
                user_api_key_auth=user_api_key_auth,
                scope_servers=allowed_mcp_servers,
            )

            try:
                resource_templates = await global_mcp_server_manager.get_resource_templates_from_server(
                    server=server,
                    mcp_auth_header=server_auth_header,
                    extra_headers=extra_headers,
                    add_prefix=True,  # Always add server prefix
                    raw_headers=raw_headers,
                )
                all_resource_templates.extend(resource_templates)
                verbose_logger.debug(
                    "Successfully fetched %s resource templates from server %s",
                    len(resource_templates),
                    server.name,
                )
            except Exception as e:
                verbose_logger.exception(
                    "Error getting resource templates from server %s: %s",
                    server.name,
                    str(e),
                )

        verbose_logger.info(
            "Successfully fetched %s resource templates total from all MCP servers",
            len(all_resource_templates),
        )

        return all_resource_templates

    async def filter_tools_by_key_team_permissions(
        tools: List[MCPTool],
        server_id: str,
        user_api_key_auth: Optional[UserAPIKeyAuth],
    ) -> List[MCPTool]:
        """
        Filter tools based on key/team mcp_tool_permissions.

        Note: Tool names in the DB are stored without server prefixes,
        but tool names from MCP servers are prefixed. We need to strip
        the prefix before comparing.
        """
        # Filter by key/team tool-level permissions
        allowed_tool_names = await MCPRequestHandler.get_allowed_tools_for_server(
            server_id=server_id,
            user_api_key_auth=user_api_key_auth,
        )
        if allowed_tool_names is None:
            return tools

        # Tools arrive prefixed with the server's own prefix; strip exactly that
        # prefix (resolved from the server) rather than the first separator, so a
        # prefix containing the separator still reduces to the stored bare name.
        server = global_mcp_server_manager.get_mcp_server_by_id(server_id)
        return [t for t in tools if strip_known_server_prefix(t.name, server) in allowed_tool_names]

    async def _merge_toolset_permissions(
        user_api_key_auth: Optional[UserAPIKeyAuth],
    ) -> Optional[UserAPIKeyAuth]:
        """
        Resolve mcp_toolsets on the key's object_permission into tool-level permissions
        and merge them (union) into object_permission.mcp_tool_permissions.

        Returns the (possibly mutated copy of) user_api_key_auth.
        """
        if user_api_key_auth is None:
            return None
        op = user_api_key_auth.object_permission
        if op is None:
            return user_api_key_auth
        toolset_ids = getattr(op, "mcp_toolsets", None) or []
        if not toolset_ids:
            return user_api_key_auth

        toolset_perms = await global_mcp_server_manager.resolve_toolset_tool_permissions(toolset_ids=toolset_ids)
        if not toolset_perms:
            return user_api_key_auth

        # Merge toolset_perms into existing mcp_tool_permissions (union)
        existing = dict(op.mcp_tool_permissions or {})
        for server_id, tool_names in toolset_perms.items():
            existing_tools = existing.get(server_id, [])
            merged = list(set(existing_tools) | set(tool_names))
            existing[server_id] = merged

        # Build updated object_permission with merged tool permissions and server IDs.
        # Union the toolset's server IDs into mcp_servers so downstream server-level
        # filtering doesn't silently drop servers that the toolset references but that
        # aren't already in the key's explicit mcp_servers list.
        merged_servers = list(set(op.mcp_servers or []) | set(existing.keys()))
        updated_op = op.model_copy(update={"mcp_servers": merged_servers, "mcp_tool_permissions": existing})
        return user_api_key_auth.model_copy(update={"object_permission": updated_op})

    async def _list_mcp_tools(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
        mcp_auth_header: Optional[str] = None,
        mcp_servers: Optional[List[str]] = None,
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]] = None,
        oauth2_headers: Optional[Dict[str, str]] = None,
        raw_headers: Optional[Dict[str, str]] = None,
        log_list_tools_to_spendlogs: bool = False,
        list_tools_log_source: Optional[str] = None,
        client_ip: Optional[str] = None,
    ) -> AggregateToolListing:
        """
        List all available MCP tools.

        Args:
            user_api_key_auth: User authentication info for access control
            mcp_auth_header: Optional auth header for MCP server (deprecated)
            mcp_servers: Optional list of server names/aliases to filter by
            mcp_server_auth_headers: Optional dict of server-specific auth headers {server_alias: auth_value}
            client_ip: Client IP for IP-based server access control

        Returns:
            AggregateToolListing: Combined tools from all accessible servers plus each server's
            classified listing outcome
        """
        if not MCP_AVAILABLE:
            return AggregateToolListing(tools=[], outcomes={})

        # Resolve toolset permissions and merge into the key's object_permission
        # so that the existing filter_tools_by_key_team_permissions logic picks them up.
        user_api_key_auth = await _merge_toolset_permissions(user_api_key_auth)

        try:
            listing = await _get_tools_from_mcp_servers(
                user_api_key_auth=user_api_key_auth,
                mcp_auth_header=mcp_auth_header,
                mcp_servers=mcp_servers,
                mcp_server_auth_headers=mcp_server_auth_headers,
                oauth2_headers=oauth2_headers,
                raw_headers=raw_headers,
                log_list_tools_to_spendlogs=log_list_tools_to_spendlogs,
                list_tools_log_source=list_tools_log_source,
                client_ip=client_ip,
            )
            verbose_logger.debug(f"Successfully fetched {len(listing.tools)} tools from managed MCP servers")
            return listing
        except Exception as e:
            verbose_logger.exception(f"Error getting tools from managed MCP servers: {str(e)}")
            # Continue with an empty listing instead of failing completely
            return AggregateToolListing(tools=[], outcomes={})

    async def _list_mcp_prompts(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
        mcp_auth_header: Optional[str] = None,
        mcp_servers: Optional[List[str]] = None,
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]] = None,
        oauth2_headers: Optional[Dict[str, str]] = None,
        raw_headers: Optional[Dict[str, str]] = None,
    ) -> List[Prompt]:
        """
        List all available MCP prompts.

        Args:
            user_api_key_auth: User authentication info for access control
            mcp_auth_header: Optional auth header for MCP server (deprecated)
            mcp_servers: Optional list of server names/aliases to filter by
            mcp_server_auth_headers: Optional dict of server-specific auth headers {server_alias: auth_value}

        Returns:
            List[Prompt]: Combined list of tools from all accessible servers
        """
        if not MCP_AVAILABLE:
            return []
        # Get tools from managed MCP servers with error handling
        managed_prompts = []
        try:
            managed_prompts = await _get_prompts_from_mcp_servers(
                user_api_key_auth=user_api_key_auth,
                mcp_auth_header=mcp_auth_header,
                mcp_servers=mcp_servers,
                mcp_server_auth_headers=mcp_server_auth_headers,
                oauth2_headers=oauth2_headers,
                raw_headers=raw_headers,
            )
            verbose_logger.debug(f"Successfully fetched {len(managed_prompts)} prompts from managed MCP servers")
        except Exception as e:
            verbose_logger.exception(f"Error getting tools from managed MCP servers: {str(e)}")
            # Continue with empty managed tools list instead of failing completely

        return managed_prompts

    async def _list_mcp_resources(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
        mcp_auth_header: Optional[str] = None,
        mcp_servers: Optional[List[str]] = None,
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]] = None,
        oauth2_headers: Optional[Dict[str, str]] = None,
        raw_headers: Optional[Dict[str, str]] = None,
    ) -> List[Resource]:
        """List all available MCP resources."""

        if not MCP_AVAILABLE:
            return []

        managed_resources: List[Resource] = []
        try:
            managed_resources = await _get_resources_from_mcp_servers(
                user_api_key_auth=user_api_key_auth,
                mcp_auth_header=mcp_auth_header,
                mcp_servers=mcp_servers,
                mcp_server_auth_headers=mcp_server_auth_headers,
                oauth2_headers=oauth2_headers,
                raw_headers=raw_headers,
            )
            verbose_logger.debug(f"Successfully fetched {len(managed_resources)} resources from managed MCP servers")
        except Exception as e:
            verbose_logger.exception(f"Error getting resources from managed MCP servers: {str(e)}")

        return managed_resources

    async def _list_mcp_resource_templates(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
        mcp_auth_header: Optional[str] = None,
        mcp_servers: Optional[List[str]] = None,
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]] = None,
        oauth2_headers: Optional[Dict[str, str]] = None,
        raw_headers: Optional[Dict[str, str]] = None,
    ) -> List[ResourceTemplate]:
        """List all available MCP resource templates."""

        if not MCP_AVAILABLE:
            return []

        managed_resource_templates: List[ResourceTemplate] = []
        try:
            managed_resource_templates = await _get_resource_templates_from_mcp_servers(
                user_api_key_auth=user_api_key_auth,
                mcp_auth_header=mcp_auth_header,
                mcp_servers=mcp_servers,
                mcp_server_auth_headers=mcp_server_auth_headers,
                oauth2_headers=oauth2_headers,
                raw_headers=raw_headers,
            )
            verbose_logger.debug(
                "Successfully fetched %s resource templates from managed MCP servers",
                len(managed_resource_templates),
            )
        except Exception as e:
            verbose_logger.exception(
                "Error getting resource templates from managed MCP servers: %s",
                str(e),
            )

        return managed_resource_templates

    def _resolve_display_name_to_original(
        name: str,
        allowed_mcp_servers: List[MCPServer],
    ) -> str:
        """Translate a display-name override back to the original prefixed tool name.

        When a client received a customised display name from tools/list (e.g.
        "Get Pet") it will call tools/call with that same string.  We need to
        reverse-map it to the original prefixed name (e.g.
        "petstore_mcp-getPetById") before any routing or permission logic runs.
        """
        for server in allowed_mcp_servers:
            display_map = server.tool_name_to_display_name or {}
            for unprefixed_name, display_name in display_map.items():
                if display_name == name:
                    return add_server_prefix_to_name(unprefixed_name, get_server_prefix(server))
        return name

    async def _get_byok_credential(
        mcp_server: MCPServer,
        user_api_key_auth: Optional[UserAPIKeyAuth],
    ) -> Optional[str]:
        """Retrieve the stored BYOK credential for a user+server pair.

        Uses the shared _byok_cred_cache to avoid a DB round-trip on every
        tool call within the TTL window.
        """
        if not mcp_server.is_byok:
            return None
        user_id = (user_api_key_auth.user_id if user_api_key_auth else None) or ""
        if not user_id:
            return None

        cache_key = (user_id, mcp_server.server_id)
        cached = _byok_cred_cache.get(cache_key)
        if cached is not None:
            credential, ts = cached
            if time.monotonic() - ts < _BYOK_CRED_CACHE_TTL:
                return credential

        from litellm.proxy._experimental.mcp_server.db import get_user_credential
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            return None
        credential = await get_user_credential(
            prisma_client=prisma_client,
            user_id=user_id,
            server_id=mcp_server.server_id,
        )
        _write_byok_cred_cache(user_id, mcp_server.server_id, credential)
        return credential

    async def _check_byok_credential(
        mcp_server: MCPServer,
        user_api_key_auth: Optional[UserAPIKeyAuth],
    ) -> None:
        """
        If the MCP server is BYOK-enabled, verify that the requesting user has a
        stored credential.  When no credential is found, raise an HTTP 401 with a
        WWW-Authenticate header that points the MCP client to our OAuth metadata
        endpoint so it can drive the authorization flow.
        """
        if not mcp_server.is_byok:
            return

        user_id = (user_api_key_auth.user_id if user_api_key_auth else None) or ""
        if not user_id:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "byok_auth_required",
                    "server_id": mcp_server.server_id,
                    "server_name": mcp_server.server_name or mcp_server.name,
                    "message": "User identity is required for BYOK servers",
                },
                headers={"WWW-Authenticate": 'Bearer resource_metadata="/.well-known/oauth-protected-resource"'},
            )

        # Check shared credential cache before hitting the DB.
        cache_key = (user_id, mcp_server.server_id)
        cached = _byok_cred_cache.get(cache_key)
        if cached is not None:
            cached_cred, ts = cached
            if time.monotonic() - ts < _BYOK_CRED_CACHE_TTL:
                if cached_cred is None:
                    raise HTTPException(
                        status_code=401,
                        detail={
                            "error": "byok_auth_required",
                            "server_id": mcp_server.server_id,
                            "server_name": mcp_server.server_name or mcp_server.name,
                            "message": (
                                "No stored credential found for this BYOK server. "
                                "Complete the OAuth authorization flow to provide your API key."
                            ),
                        },
                        headers={
                            "WWW-Authenticate": 'Bearer resource_metadata="/.well-known/oauth-protected-resource"'
                        },
                    )
                return

        from litellm.proxy._experimental.mcp_server.db import get_user_credential
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            # Fail closed on DB unavailability: returning here previously
            # bypassed the ownership check and let any proxy-authenticated
            # caller invoke BYOK tools during outage windows.
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "byok_auth_unavailable",
                    "server_id": mcp_server.server_id,
                    "server_name": mcp_server.server_name or mcp_server.name,
                    "message": "BYOK credential check requires a database connection.",
                },
            )

        credential = await get_user_credential(
            prisma_client=prisma_client,
            user_id=user_id,
            server_id=mcp_server.server_id,
        )
        _write_byok_cred_cache(user_id, mcp_server.server_id, credential)
        if credential is None:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "byok_auth_required",
                    "server_id": mcp_server.server_id,
                    "server_name": mcp_server.server_name or mcp_server.name,
                    "message": (
                        "No stored credential found for this BYOK server. "
                        "Complete the OAuth authorization flow to provide your API key."
                    ),
                },
                headers={"WWW-Authenticate": 'Bearer resource_metadata="/.well-known/oauth-protected-resource"'},
            )

    async def execute_mcp_tool(
        name: str,
        arguments: Dict[str, Any],
        allowed_mcp_servers: List[MCPServer],
        start_time: datetime,
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
        mcp_auth_header: Optional[str] = None,
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]] = None,
        oauth2_headers: Optional[Dict[str, str]] = None,
        raw_headers: Optional[Dict[str, str]] = None,
        host_progress_callback: Optional[Callable] = None,
        **kwargs: Any,
    ) -> CallToolResult:
        """
        Execute MCP tool.

        This function assumes permission checks have already been performed.

        Args:
            name: Tool name (may include server prefix)
            arguments: Tool arguments
            allowed_mcp_servers: Pre-validated list of servers the user can access
            start_time: Start time for logging
            user_api_key_auth: Optional user API key auth for logging
            mcp_auth_header: Optional MCP auth header
            mcp_server_auth_headers: Optional server-specific auth headers
            oauth2_headers: Optional OAuth2 headers
            raw_headers: Optional raw HTTP headers
            **kwargs: Additional arguments (e.g., litellm_logging_obj)

        Returns:
            CallToolResult: Tool execution result
        """
        # Track resolved MCP server for both permission checks and dispatch
        mcp_server: Optional[MCPServer] = None
        requested_server_id: Optional[str] = kwargs.get("requested_server_id")

        # If the client called with a display-name override (e.g. "Get Pet"),
        # translate it back to the original prefixed name before any routing.
        name = _resolve_display_name_to_original(name, allowed_mcp_servers)

        # Remove prefix from tool name for logging and processing
        original_tool_name, server_name = split_server_prefix_from_name(name)

        requested_server: Optional[MCPServer] = None
        if requested_server_id:
            requested_server = next(
                (s for s in allowed_mcp_servers if s.server_id == requested_server_id),
                None,
            )

        name_is_prefixed = False
        if requested_server is not None and MCP_TOOL_PREFIX_SEPARATOR in name:
            all_registry_prefixes: Set[str] = set()
            for registry_server in global_mcp_server_manager.get_registry().values():
                for known_prefix in iter_known_server_prefixes(registry_server):
                    all_registry_prefixes.add(normalize_server_name(known_prefix))
            name_is_prefixed = is_tool_name_prefixed(name, known_server_prefixes=all_registry_prefixes)

        if requested_server is not None and not name_is_prefixed:
            # REST callers may pass server_id with the upstream tool name (no
            # LiteLLM prefix). The first segment is not a registered server
            # prefix, so the whole string is the upstream tool name and may
            # legitimately contain the separator (e.g. "text-to-speech").
            # server_id is authoritative for routing and auth.
            mcp_server = requested_server
            server_name = requested_server.name
            original_tool_name = name
        else:
            # Resolve from tool name (MCP JSON-RPC or prefixed REST tool names).
            mcp_server = global_mcp_server_manager._get_mcp_server_from_tool_name(name)
            if mcp_server is None and requested_server is not None:
                for known_prefix in iter_known_server_prefixes(requested_server):
                    candidate = global_mcp_server_manager._get_mcp_server_from_tool_name(
                        add_server_prefix_to_name(name, known_prefix)
                    )
                    if candidate is not None:
                        mcp_server = candidate
                        break
            if mcp_server is not None:
                server_name = mcp_server.name

            if requested_server is not None:
                if mcp_server is not None and mcp_server.server_id != requested_server.server_id:
                    raise HTTPException(
                        status_code=403,
                        detail={
                            "error": "tool_server_mismatch",
                            "message": (
                                f"Tool '{name}' belongs to MCP server "
                                f"'{mcp_server.name}' but request specified "
                                f"server_id for '{requested_server.name}'."
                            ),
                        },
                    )
                if mcp_server is None:
                    mcp_server = requested_server
                    server_name = requested_server.name

        # Only enforce server-level permissions when we can resolve a server
        if server_name:
            if not MCPRequestHandler.is_tool_allowed(
                allowed_mcp_servers=[server.name for server in allowed_mcp_servers],
                server_name=server_name,
            ):
                raise HTTPException(
                    status_code=403,
                    detail=f"User not allowed to call this tool. Allowed MCP servers: {allowed_mcp_servers}",
                )

        standard_logging_mcp_tool_call: StandardLoggingMCPToolCall = _get_standard_logging_mcp_tool_call(
            name=original_tool_name,  # Use original name for logging
            arguments=arguments,
            server_name=server_name,
            session_id=_mcp_session_id_from_headers(raw_headers),
        )
        litellm_logging_obj: Optional[LiteLLMLoggingObj] = kwargs.get("litellm_logging_obj", None)
        if litellm_logging_obj:
            litellm_logging_obj.model_call_details["mcp_tool_call_metadata"] = standard_logging_mcp_tool_call
            litellm_logging_obj.model = f"MCP: {name}"
            litellm_logging_obj.model_call_details["model"] = f"MCP: {name}"
        # Resolve the MCP server early so BYOK checks and credential injection
        # apply to ALL dispatch paths (local tool registry AND managed MCP server).
        if mcp_server is None:
            mcp_server = global_mcp_server_manager._get_mcp_server_from_tool_name(name)

        if mcp_server:
            standard_logging_mcp_tool_call["mcp_server_cost_info"] = (mcp_server.mcp_info or {}).get(
                "mcp_server_cost_info"
            )
            if litellm_logging_obj:
                litellm_logging_obj.model_call_details["mcp_tool_call_metadata"] = standard_logging_mcp_tool_call

            # BYOK: retrieve the stored per-user credential.  A single DB call
            # both checks existence and fetches the value, avoiding a double query.
            if mcp_server.is_byok and not mcp_auth_header:
                byok_cred = await _get_byok_credential(mcp_server, user_api_key_auth)
                if byok_cred is None:
                    raise HTTPException(
                        status_code=401,
                        detail={
                            "error": "byok_auth_required",
                            "server_id": mcp_server.server_id,
                            "server_name": mcp_server.server_name or mcp_server.name,
                            "message": (
                                "No stored credential found for this BYOK server. "
                                "Complete the OAuth authorization flow to provide your API key."
                            ),
                        },
                        headers={
                            "WWW-Authenticate": 'Bearer resource_metadata="/.well-known/oauth-protected-resource"'
                        },
                    )
                mcp_auth_header = byok_cred
            elif mcp_server.is_byok:
                # External auth header supplied; still enforce user-identity check.
                await _check_byok_credential(mcp_server, user_api_key_auth)

        # Check if tool exists in local registry first (for OpenAPI-based tools)
        # These tools are registered with their prefixed names
        #########################################################
        local_tool = global_mcp_tool_registry.get_tool(name)
        if local_tool:
            # OpenAPI-backed tools used to bypass `pre_call_tool_check` —
            # only the managed path ran allowed/banned-tool checks, key/team
            # tool permissions, and parameter validation. Run the same checks
            # before dispatching to the local registry. Refuse the call if
            # we cannot resolve a server: tools registered via
            # openapi_to_mcp_generator are always tied to a server, so a
            # missing mcp_server here means the tool->server mapping has
            # not finished initializing or the registry entry is orphaned.
            # Skipping the check would re-open the same authorization gap.
            if mcp_server is None:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"MCP server for tool '{name}' is not available; "
                        "refusing to dispatch without authorization checks. "
                        "Retry once the server is registered."
                    ),
                )

            # `pre_call_tool_check` calls into `proxy_logging_obj` for the
            # pre-call guardrail hooks, so source it from the canonical
            # `proxy_server` module the same way `_handle_managed_mcp_tool`
            # does. `kwargs.get("proxy_logging_obj")` is None on the MCP
            # entry path and would crash with AttributeError after the
            # security checks pass.
            from litellm.proxy.proxy_server import proxy_logging_obj

            hook_result = await global_mcp_server_manager.pre_call_tool_check(
                name=original_tool_name,
                arguments=arguments or {},
                server_name=server_name or mcp_server.name,
                user_api_key_auth=user_api_key_auth,
                proxy_logging_obj=proxy_logging_obj,  # type: ignore[arg-type]
                server=mcp_server,
                raw_headers=raw_headers,
            )
            # `pre_call_tool_check` may return guardrail-modified
            # arguments; honor them on the local path too.
            if isinstance(hook_result, dict) and "arguments" in hook_result:
                arguments = hook_result["arguments"]

            verbose_logger.debug(f"Executing local registry tool: {name}")
            # For BYOK servers the credential must be injected via a ContextVar
            # because the tool function has headers baked into its closure.
            # Pre-format the full Authorization header value using the server's
            # configured auth_type so the generator doesn't need to know the prefix.
            auth_header_value: Optional[str] = None
            if mcp_auth_header:
                server_auth_type = getattr(mcp_server, "auth_type", None) if mcp_server else None
                if server_auth_type == MCPAuth.api_key:
                    auth_header_value = f"ApiKey {mcp_auth_header}"
                elif server_auth_type == MCPAuth.basic:
                    auth_header_value = f"Basic {mcp_auth_header}"
                else:
                    auth_header_value = f"Bearer {mcp_auth_header}"

            # Forward named client headers to OpenAPI tool upstream requests.
            # MCPServer.extra_headers lists header names to copy from raw_headers.
            # The strip decision is centralized in _should_strip_caller_authorization so this
            # OpenAPI/local path agrees with the managed paths: M2M and the resolver-owned modes
            # (token_exchange's raw subject token, authorization_code's stored token) must never
            # have the caller's Authorization forwarded verbatim upstream.
            forwarded_headers: Optional[Dict[str, str]] = None
            if mcp_server and mcp_server.extra_headers and raw_headers:
                normalized_raw = {str(k).lower(): v for k, v in raw_headers.items() if isinstance(k, str)}
                skip_caller_authorization = _should_strip_caller_authorization(
                    mcp_server=mcp_server,
                    raw_headers=raw_headers,
                    user_api_key_auth=user_api_key_auth,
                )
                for header_name in mcp_server.extra_headers:
                    if not isinstance(header_name, str):
                        continue
                    if skip_caller_authorization and header_name.lower() == "authorization":
                        continue
                    value = normalized_raw.get(header_name.lower())
                    if value is not None:
                        if forwarded_headers is None:
                            forwarded_headers = {}
                        forwarded_headers[header_name] = value

            _auth_token = _request_auth_header.set(auth_header_value)
            _extra_token = _request_extra_headers.set(forwarded_headers)
            try:
                local_content = await _handle_local_mcp_tool(name, arguments)
            finally:
                _request_auth_header.reset(_auth_token)
                _request_extra_headers.reset(_extra_token)
            response = CallToolResult(content=cast(Any, local_content), isError=False)

        # Try managed MCP server tool (pass the full prefixed name)
        # Primary and recommended way to use external MCP servers
        #########################################################
        elif mcp_server:
            response = await _handle_managed_mcp_tool(
                server_name=server_name,
                name=original_tool_name,  # Pass the full name (potentially prefixed)
                arguments=arguments,
                user_api_key_auth=user_api_key_auth,
                mcp_auth_header=mcp_auth_header,
                mcp_server_auth_headers=mcp_server_auth_headers,
                oauth2_headers=oauth2_headers,
                raw_headers=raw_headers,
                litellm_logging_obj=litellm_logging_obj,
                host_progress_callback=host_progress_callback,
            )

        # Fall back to local tool registry with original name (legacy support)
        #########################################################
        # Deprecated: Local MCP Server Tool
        #########################################################
        else:
            local_content = await _handle_local_mcp_tool(original_tool_name, arguments)
            response = CallToolResult(content=cast(Any, local_content), isError=False)

        return response

    _MCP_CREDENTIAL_REQUEST_FIELDS = frozenset(
        {
            "raw_headers",
            "mcp_auth_header",
            "mcp_server_auth_headers",
            "oauth2_headers",
            "user_api_key_auth",
        }
    )

    async def _fire_mcp_tool_call_logging(
        logging_obj: LiteLLMLoggingObj,
        result: Any,
        start_time: datetime,
        end_time: datetime,
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
        request_data: Optional[Mapping[str, object]] = None,
    ) -> None:
        """Fire post-call logging for an executed MCP tool call.

        A result with ``isError=True`` is logged as a failure (``status="failure"``
        payload, so OTel marks the span ERROR) while the HTTP wire behavior stays
        200 + ``isError: true`` per the MCP spec. The error check runs after
        ``async_post_mcp_tool_call_hook`` because guardrails may flip the result
        to ``isError=True`` in that hook. Raised exceptions never reach here (the
        ``@client`` wrapper and ``call_mcp_tool``'s except path log those), so
        this cannot double-log a failure.

        ``request_data`` may carry credential-bearing fields (the REST path puts
        ``raw_headers``, ``mcp_auth_header``, ``mcp_server_auth_headers``, and
        ``oauth2_headers`` at the top level of its data dict), so those are
        stripped before the dict is handed to ``post_call_failure_hook``
        callbacks.
        """
        logging_obj.post_call(original_response=result)
        await logging_obj.async_post_mcp_tool_call_hook(
            kwargs=logging_obj.model_call_details,
            response_obj=result,
            start_time=start_time,
            end_time=end_time,
        )
        logging_obj.call_type = CallTypes.call_mcp_tool.value
        error_message = extract_mcp_tool_result_error_message(result)
        if error_message is None:
            await logging_obj.async_success_handler(result=result, start_time=start_time, end_time=end_time)
            return

        logging_obj.has_run_logging(event_type="sync_success")
        logging_obj.has_run_logging(event_type="async_success")
        tool_error = MCPToolResultError(error_message)
        logging_obj.failure_handler(tool_error, "", start_time, end_time)
        await logging_obj.async_failure_handler(tool_error, "", start_time, end_time)

        if user_api_key_auth is None:
            return
        from litellm.proxy.proxy_server import proxy_logging_obj

        if proxy_logging_obj:
            sanitized_request_data = {
                key: value for key, value in (request_data or {}).items() if key not in _MCP_CREDENTIAL_REQUEST_FIELDS
            }
            await proxy_logging_obj.post_call_failure_hook(
                request_data=sanitized_request_data,
                original_exception=tool_error,
                user_api_key_dict=user_api_key_auth,
                route="/mcp/call_tool",
            )

    @client
    async def call_mcp_tool(
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
        mcp_auth_header: Optional[str] = None,
        mcp_servers: Optional[List[str]] = None,
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]] = None,
        oauth2_headers: Optional[Dict[str, str]] = None,
        raw_headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> CallToolResult:
        """
        Call a specific tool with the provided arguments (handles prefixed tool names).
        """
        start_time = datetime.now()
        litellm_logging_obj: Optional[LiteLLMLoggingObj] = kwargs.get("litellm_logging_obj", None)

        try:
            if arguments is None:
                raise HTTPException(status_code=400, detail="Request arguments are required")

            ## CHECK IF USER IS ALLOWED TO CALL THIS TOOL
            allowed_mcp_server_ids = await global_mcp_server_manager.get_allowed_mcp_servers(
                user_api_key_auth=user_api_key_auth,
            )

            allowed_mcp_servers: List[MCPServer] = []
            for allowed_mcp_server_id in allowed_mcp_server_ids:
                allowed_server = global_mcp_server_manager.get_mcp_server_by_id(allowed_mcp_server_id)
                if allowed_server is not None:
                    # Same request-time oauth2_flow backstop the listing path applies,
                    # so a null-flow M2M-shape row is treated as M2M on tool calls too.
                    allowed_server = MCPServerManager.resolve_oauth2_flow_for_request(allowed_server)
                    allowed_mcp_servers.append(allowed_server)

            allowed_mcp_servers = await _get_allowed_mcp_servers_from_mcp_server_names(
                mcp_servers=mcp_servers,
                allowed_mcp_servers=allowed_mcp_servers,
            )
            if not allowed_mcp_servers:
                raise HTTPException(
                    status_code=403,
                    detail="User not allowed to call this tool.",
                )

            # Delegate to execute_mcp_tool for execution
            response = await execute_mcp_tool(
                name=name,
                arguments=arguments,
                allowed_mcp_servers=allowed_mcp_servers,
                start_time=start_time,
                user_api_key_auth=user_api_key_auth,
                mcp_auth_header=mcp_auth_header,
                mcp_server_auth_headers=mcp_server_auth_headers,
                oauth2_headers=oauth2_headers,
                raw_headers=raw_headers,
                **kwargs,
            )
        except MCPUpstreamAuthError:
            # A client-forwarded pass-through upstream 401 is an expected caller-must-reauth signal, so
            # re-raise it without post_call_failure_hook, which fires the proxy's llm_exceptions alert.
            # mcp_server_tool_call then downgrades it to an informational isError result for the
            # streamable client. Note: this function is @client-decorated, so the decorator's standard
            # failure logging still records the event (spend log / OTel); only the extra alert sink is
            # skipped here.
            raise
        except Exception as e:
            traceback_str = traceback.format_exc(limit=MAXIMUM_TRACEBACK_LINES_TO_LOG)
            from litellm.proxy.proxy_server import proxy_logging_obj

            if proxy_logging_obj and user_api_key_auth:
                await proxy_logging_obj.post_call_failure_hook(
                    request_data=kwargs,
                    original_exception=e,
                    user_api_key_dict=user_api_key_auth,
                    route="/mcp/call_tool",
                    traceback_str=traceback_str,
                )
            raise

        if litellm_logging_obj:
            await _fire_mcp_tool_call_logging(
                logging_obj=litellm_logging_obj,
                result=response,
                start_time=start_time,
                end_time=datetime.now(),
                user_api_key_auth=user_api_key_auth,
                request_data=kwargs,
            )
        return response

    async def mcp_get_prompt(
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
        mcp_auth_header: Optional[str] = None,
        mcp_servers: Optional[List[str]] = None,
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]] = None,
        oauth2_headers: Optional[Dict[str, str]] = None,
        raw_headers: Optional[Dict[str, str]] = None,
    ) -> GetPromptResult:
        """
        Fetch a specific MCP prompt, handling both prefixed and unprefixed names.
        """
        allowed_mcp_servers = await _get_allowed_mcp_servers(
            user_api_key_auth=user_api_key_auth,
            mcp_servers=mcp_servers,
        )

        if not allowed_mcp_servers:
            raise HTTPException(
                status_code=403,
                detail="User not allowed to get this prompt.",
            )

        # Extract server name from prefixed prompt name
        original_prompt_name, server_name = split_server_prefix_from_name(name)

        server = next((s for s in allowed_mcp_servers if s.name == server_name), None)
        if server is None:
            raise HTTPException(
                status_code=403,
                detail="User not allowed to get this prompt.",
            )

        server_auth_header, extra_headers = _prepare_mcp_server_headers(
            server=server,
            mcp_server_auth_headers=mcp_server_auth_headers,
            mcp_auth_header=mcp_auth_header,
            oauth2_headers=oauth2_headers,
            raw_headers=raw_headers,
            user_api_key_auth=user_api_key_auth,
        )

        return await global_mcp_server_manager.get_prompt_from_server(
            server=server,
            prompt_name=original_prompt_name,
            arguments=arguments,
            mcp_auth_header=server_auth_header,
            extra_headers=extra_headers,
            raw_headers=raw_headers,
        )

    async def mcp_read_resource(
        url: AnyUrl,
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
        mcp_auth_header: Optional[str] = None,
        mcp_servers: Optional[List[str]] = None,
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]] = None,
        oauth2_headers: Optional[Dict[str, str]] = None,
        raw_headers: Optional[Dict[str, str]] = None,
    ) -> ReadResourceResult:
        """Read resource contents from upstream MCP servers."""

        allowed_mcp_servers = await _get_allowed_mcp_servers(
            user_api_key_auth=user_api_key_auth,
            mcp_servers=mcp_servers,
        )

        if not allowed_mcp_servers:
            raise HTTPException(
                status_code=403,
                detail="User not allowed to read this resource.",
            )

        if len(allowed_mcp_servers) != 1:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Multiple MCP servers configured; read_resource currently supports exactly one allowed server."
                ),
            )

        server = allowed_mcp_servers[0]

        server_auth_header, extra_headers = _prepare_mcp_server_headers(
            server=server,
            mcp_server_auth_headers=mcp_server_auth_headers,
            mcp_auth_header=mcp_auth_header,
            oauth2_headers=oauth2_headers,
            raw_headers=raw_headers,
            user_api_key_auth=user_api_key_auth,
        )

        return await global_mcp_server_manager.read_resource_from_server(
            server=server,
            url=url,
            mcp_auth_header=server_auth_header,
            extra_headers=extra_headers,
            raw_headers=raw_headers,
        )

    def _get_standard_logging_mcp_tool_call(
        name: str,
        arguments: Dict[str, Any],
        server_name: Optional[str],
        session_id: Optional[str] = None,
    ) -> StandardLoggingMCPToolCall:
        mcp_server = global_mcp_server_manager._get_mcp_server_from_tool_name(name)
        namespaced_tool_name = f"{server_name}/{name}" if server_name else name
        if mcp_server:
            mcp_info = mcp_server.mcp_info or {}
            return StandardLoggingMCPToolCall(
                name=name,
                arguments=arguments,
                mcp_server_name=mcp_info.get("server_name"),
                mcp_server_logo_url=mcp_info.get("logo_url"),
                namespaced_tool_name=namespaced_tool_name,
                mcp_session_id=session_id,
                mcp_auth_mode=mcp_server.auth_type,
                mcp_server_resource=_redact_mcp_resource_url(mcp_server.url),
            )
        else:
            return StandardLoggingMCPToolCall(
                name=name,
                arguments=arguments,
                namespaced_tool_name=namespaced_tool_name,
                mcp_session_id=session_id,
            )

    async def _handle_managed_mcp_tool(
        server_name: str,
        name: str,
        arguments: Dict[str, Any],
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
        mcp_auth_header: Optional[str] = None,
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]] = None,
        oauth2_headers: Optional[Dict[str, str]] = None,
        raw_headers: Optional[Dict[str, str]] = None,
        litellm_logging_obj: Optional[Any] = None,
        host_progress_callback: Optional[Callable] = None,
    ) -> CallToolResult:
        """Handle tool execution for managed server tools"""
        # Import here to avoid circular import
        from litellm.proxy.proxy_server import proxy_logging_obj

        call_tool_result = await global_mcp_server_manager.call_tool(
            server_name=server_name,
            name=name,
            arguments=arguments,
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=mcp_auth_header,
            mcp_server_auth_headers=mcp_server_auth_headers,
            oauth2_headers=oauth2_headers,
            raw_headers=raw_headers,
            proxy_logging_obj=proxy_logging_obj,
            host_progress_callback=host_progress_callback,
        )
        verbose_logger.debug("CALL TOOL RESULT: %s", call_tool_result)
        return call_tool_result

    async def _handle_local_mcp_tool(
        name: str, arguments: Dict[str, Any]
    ) -> List[Union[TextContent, ImageContent, EmbeddedResource]]:
        """
        Handle tool execution for local registry tools
        Note: Local tools don't use prefixes, so we use the original name
        """
        import inspect

        tool = global_mcp_tool_registry.get_tool(name)
        if not tool:
            raise HTTPException(status_code=404, detail=f"Tool '{name}' not found")

        try:
            # Check if handler is async or sync
            if inspect.iscoroutinefunction(tool.handler):
                result = await tool.handler(**arguments)
            else:
                result = tool.handler(**arguments)
            return [TextContent(text=str(result), type="text")]
        except Exception as e:
            verbose_logger.exception(f"Error executing local tool {name}: {str(e)}")
            return [TextContent(text=f"Error: {str(e)}", type="text")]

    def _get_mcp_servers_in_path(path: str) -> Optional[List[str]]:
        """
        Get the MCP servers from the path
        """
        import re

        mcp_servers_from_path: Optional[List[str]] = None
        segments = [s for s in path.split("/") if s]
        if len(segments) >= 2 and segments[1] == "mcp" and segments[0] != "mcp":
            return [segments[0]]

        # Match /mcp/<servers_and_maybe_path>
        # Where servers can be comma-separated list of server names
        # Server names can contain slashes (e.g., "custom_solutions/user_123")
        mcp_path_match = re.match(r"^/mcp/([^?#]+)(?:\?.*)?(?:#.*)?$", path)
        if mcp_path_match:
            servers_and_path = mcp_path_match.group(1)

            if servers_and_path:
                # Check if it contains commas (comma-separated servers)
                if "," in servers_and_path:
                    # For comma-separated, look for a path at the end
                    # Common patterns: /tools, /chat/completions, etc.
                    path_match = re.search(r"/([^/,]+(?:/[^/,]+)*)$", servers_and_path)
                    if path_match:
                        # Path found at the end, remove it from servers
                        path_part = "/" + path_match.group(1)
                        servers_part = servers_and_path[: -len(path_part)]
                        mcp_servers_from_path = [s.strip() for s in servers_part.split(",") if s.strip()]
                    else:
                        # No path, just comma-separated servers
                        mcp_servers_from_path = [s.strip() for s in servers_and_path.split(",") if s.strip()]
                else:
                    # Single server case - use regex approach for server/path separation
                    # This handles cases like "custom_solutions/user_123/chat/completions"
                    # where we want to extract "custom_solutions/user_123" as the server name
                    single_server_match = re.match(r"^([^/]+(?:/[^/]+)?)(?:/.*)?$", servers_and_path)
                    if single_server_match:
                        server_name = single_server_match.group(1)
                        mcp_servers_from_path = [server_name]
                    else:
                        mcp_servers_from_path = [servers_and_path]
        return mcp_servers_from_path

    async def extract_mcp_auth_context(scope, path):
        """
        Extracts mcp_servers from the path and processes the MCP request for auth context.
        Returns: (user_api_key_auth, mcp_auth_header, mcp_servers, mcp_server_auth_headers)
        """
        mcp_servers_from_path = _get_mcp_servers_in_path(path)
        if mcp_servers_from_path is not None:
            (
                user_api_key_auth,
                mcp_auth_header,
                _,
                mcp_server_auth_headers,
                oauth2_headers,
                raw_headers,
            ) = await MCPRequestHandler.process_mcp_request(scope)
            mcp_servers = mcp_servers_from_path
        else:
            (
                user_api_key_auth,
                mcp_auth_header,
                mcp_servers,
                mcp_server_auth_headers,
                oauth2_headers,
                raw_headers,
            ) = await MCPRequestHandler.process_mcp_request(scope)
        return (
            user_api_key_auth,
            mcp_auth_header,
            mcp_servers,
            mcp_server_auth_headers,
            oauth2_headers,
            raw_headers,
        )

    def _get_session_id_from_scope(scope: Scope) -> Optional[str]:
        """
        Extract mcp-session-id from ASGI scope headers.
        Returns None if not present.
        """
        for header_name, header_value in scope.get("headers", []):
            name = header_name if isinstance(header_name, bytes) else header_name.encode()
            if name.lower() == b"mcp-session-id":
                return header_value.decode() if isinstance(header_value, bytes) else str(header_value)
        return None

    def _owner_fingerprint_for(
        user_api_key_auth: Optional[UserAPIKeyAuth],
        oauth2_headers: Optional[Dict[str, str]] = None,
        client_ip: Optional[str] = None,
    ) -> str:
        """
        Stable, non-reversible identifier for the caller used to bind an
        mcp-session-id to its creator. Hash the resolved credential before
        using it so custom key formats are never stored in cleartext.

        For OAuth2 passthrough (``UserAPIKeyAuth()`` with no key/user_id),
        the caller's identity is the upstream OAuth bearer; hash it so two
        OAuth callers with different tokens don't both fingerprint to
        ``anonymous`` and end up sharing a session.

        When no caller-identifying credentials are available at all
        (e.g. proxy running without master key, or an unauthenticated
        passthrough path), fall back to the client IP so two unrelated
        anonymous callers from different sources do not collapse to a
        single ``anonymous`` owner and end up able to drive each other's
        stateful sessions. Note: when even client IP is unavailable
        (exotic deployments without trusted X-Forwarded-For and direct
        socket info), the fingerprint degrades to the ``anonymous``
        sentinel and cannot meaningfully protect against another
        unauthenticated caller who learns the session id — owner-binding
        is best-effort in that mode.
        """

        def _bytes_for_hash(value: Any) -> Optional[bytes]:
            """Only hash str/bytes secrets; skip mocks and other unexpected types."""
            if value is None:
                return None
            if isinstance(value, (bytes, bytearray)):
                return bytes(value)
            if isinstance(value, str):
                return value.encode("utf-8")
            return None

        if user_api_key_auth is not None:
            key_material = _bytes_for_hash(getattr(user_api_key_auth, "api_key", None))
            if key_material:
                api_key_hash = hashlib.sha256(key_material).hexdigest()
                return f"key:{api_key_hash}"
            uid_material = _bytes_for_hash(getattr(user_api_key_auth, "user_id", None))
            if uid_material:
                user_id_hash = hashlib.sha256(uid_material).hexdigest()
                return f"user:{user_id_hash}"
        if oauth2_headers:
            authz = oauth2_headers.get("Authorization") or oauth2_headers.get("authorization")
            authz_bytes = _bytes_for_hash(authz)
            if authz_bytes:
                return f"oauth:{hashlib.sha256(authz_bytes).hexdigest()}"
        if client_ip and isinstance(client_ip, str):
            return f"ip:{hashlib.sha256(client_ip.encode('utf-8')).hexdigest()}"
        return "anonymous"

    def _is_initialize_request(body: bytes) -> bool:
        """
        Check if the request body is a JSON-RPC initialize method.
        Returns True if method is "initialize", False otherwise or on parse error.
        """
        if not body:
            return False
        try:
            data = json.loads(body)
            return isinstance(data, dict) and data.get("method") == "initialize"
        except (json.JSONDecodeError, TypeError):
            return False

    async def _read_request_body_for_routing(
        receive: Receive,
    ) -> Tuple[List[Message], bytes]:
        """
        Read just enough of the request body to decide whether this is a
        JSON-RPC ``initialize`` call. Returns the consumed ASGI messages so
        the caller can replay them faithfully to the downstream handler, and
        the peeked body bytes (capped at ``_MCP_ROUTING_PEEK_MAX_BYTES``).

        Stops reading from the wire as soon as either (a) we have peeked
        ``_MCP_ROUTING_PEEK_MAX_BYTES`` of body, or (b) the body is complete.
        The remainder of an oversized body is streamed lazily through
        ``wrapped_receive`` in the caller — so an authenticated client cannot
        force the proxy to buffer an arbitrarily large payload just to make a
        routing decision.
        """
        consumed_messages: List[Message] = []
        body_chunks: List[bytes] = []
        peeked_bytes = 0

        while True:
            message = await receive()
            consumed_messages.append(message)

            if message.get("type") != "http.request":
                break

            body = message.get("body", b"") or b""
            if body:
                # Only retain up to the remaining peek budget for sniffing.
                # The full ``message`` is already in memory (delivered by
                # the ASGI server) and must round-trip to the downstream
                # handler via ``consumed_messages``, but ``body_chunks`` is
                # purely for the JSON-RPC method check — there is no reason
                # to copy a large body frame into a second buffer.
                remaining = _MCP_ROUTING_PEEK_MAX_BYTES - peeked_bytes
                if remaining > 0:
                    body_chunks.append(body[:remaining])
                    peeked_bytes += min(len(body), remaining)

            if not message.get("more_body", False):
                break

            if peeked_bytes >= _MCP_ROUTING_PEEK_MAX_BYTES:
                # Stop draining; downstream replay will pull remaining chunks
                # directly from the original `receive` via wrapped_receive.
                break

        return consumed_messages, b"".join(body_chunks)

    async def _handle_stale_mcp_session(
        scope: Scope,
        receive: Receive,
        send: Send,
        mgr: "StreamableHTTPSessionManager",
    ) -> bool:
        """
        Inspect the incoming ``mcp-session-id`` header **before** the
        request reaches the MCP SDK.  If the session is stale (not known
        to this worker), strip the header so the SDK creates a fresh
        stateless session instead of returning a 400.

        Returns:
            True if the request was fully handled (e.g. DELETE on
            non-existent session).  False if the request should continue
            to the session manager.

        Fixes https://github.com/BerriAI/litellm/issues/20992
        """
        _mcp_session_header = b"mcp-session-id"
        _headers = scope.get("headers", [])

        def _normalize_header_name(header_name: Any) -> Optional[bytes]:
            if isinstance(header_name, bytes):
                return header_name.lower()
            if isinstance(header_name, str):
                return header_name.lower().encode("utf-8", errors="replace")
            return None

        _session_id: Optional[str] = None
        for header_name, header_value in _headers:
            if _normalize_header_name(header_name) == _mcp_session_header:
                if isinstance(header_value, bytes):
                    _session_id = header_value.decode("utf-8", errors="replace")
                else:
                    _session_id = str(header_value)
                break

        if _session_id is None:
            return False

        # Check in-memory session tracking
        known_sessions = getattr(mgr, "_server_instances", None)
        # If we cannot inspect known_sessions, let the manager handle it
        if known_sessions is None:
            return False

        # If session exists in this worker's memory, let the manager handle it
        try:
            if _session_id in known_sessions:
                return False
        except Exception:
            verbose_logger.debug(
                "Unable to inspect active MCP sessions for '%s'. Deferring to session manager.",
                _session_id,
            )
            return False

        # --- Session not in this worker's memory ---
        method = scope.get("method", "").upper()

        if method == "DELETE":
            _remove_stateful_session_tracking(_session_id)
            verbose_logger.info(
                "DELETE request for non-existent MCP session '%s'. Returning success (idempotent DELETE).",
                _session_id,
            )
            success_response = JSONResponse(
                status_code=200,
                content={"message": "Session terminated successfully"},
            )
            await success_response(scope, receive, send)
            return True

        # Non-DELETE: strip stale session ID to allow new session creation
        verbose_logger.warning(
            "MCP session ID '%s' not found in this worker's memory. "
            "Stripping stale header to force new session creation.",
            _session_id,
        )
        scope["headers"] = [(k, v) for k, v in _headers if _normalize_header_name(k) != _mcp_session_header]
        return False

    async def _apply_toolset_scope(
        user_api_key_auth: UserAPIKeyAuth,
        toolset_id: str,
    ) -> UserAPIKeyAuth:
        """
        Restrict a key's MCP permissions to a single toolset.

        When a request arrives via /toolset/{name}/mcp we override the key's
        object_permission so that only the toolset's tools are visible.

        Raises HTTPException(403) if the key has an explicit toolset grant list
        that does not include toolset_id (i.e. mcp_toolsets is set but empty,
        or set to a list that omits this toolset).  Admin keys always pass.
        """
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable
        from litellm.proxy.management_endpoints.common_utils import _user_has_admin_view

        # A key scoped to no MCP servers opts out of every MCP path. Enforce it
        # here too, since toolset scoping replaces mcp_servers and would otherwise
        # drop the sentinel. Checked before the admin branch, mirroring
        # get_allowed_mcp_servers.
        original_op = user_api_key_auth.object_permission
        if original_op is not None and SpecialMCPServerNames.no_mcp_servers.value in (original_op.mcp_servers or []):
            raise HTTPException(
                status_code=403,
                detail="API key is scoped to no MCP servers; toolset access is denied.",
            )

        # Access control: non-admin keys must have this toolset in their grant list.
        # Use _user_has_admin_view so that PROXY_ADMIN_VIEW_ONLY is also treated as admin.
        is_admin = _user_has_admin_view(user_api_key_auth)
        if not is_admin:
            op = user_api_key_auth.object_permission
            granted = getattr(op, "mcp_toolsets", None) if op else None
            # granted=None → key has no explicit toolset grants → deny (same semantics as
            # fetch_mcp_toolsets which returns [] for non-admin keys with no grants configured).
            # granted=[] or list without toolset_id → also deny.
            if granted is None or toolset_id not in granted:
                raise HTTPException(
                    status_code=403,
                    detail=f"API key does not have access to toolset '{toolset_id}'.",
                )

        tool_permissions = await global_mcp_server_manager.resolve_toolset_tool_permissions(toolset_ids=[toolset_id])
        server_ids = list(tool_permissions.keys())
        existing_op = user_api_key_auth.object_permission
        if existing_op is not None:
            updated_op = existing_op.model_copy(
                update={
                    "mcp_servers": server_ids,
                    "mcp_tool_permissions": tool_permissions,
                    "mcp_toolsets": [],
                    # mcp_access_groups is preserved: a key's access-group grants
                    # remain valid even when the request is scoped to a single toolset.
                }
            )
        else:
            updated_op = LiteLLM_ObjectPermissionTable(
                object_permission_id="toolset-scope",
                mcp_servers=server_ids,
                mcp_tool_permissions=tool_permissions,
            )
        return user_api_key_auth.model_copy(update={"object_permission": updated_op})

    def _get_passthrough_resource_metadata_url(scope: Scope, server_name: str) -> str:
        request = StarletteRequest(scope)
        base_url = get_request_base_url(request)
        _path = scope.get("_original_path") or scope.get("path", "") or ""

        if _path.startswith(f"/{server_name}/mcp"):
            return f"{base_url}/.well-known/oauth-protected-resource/{server_name}/mcp"
        return f"{base_url}/.well-known/oauth-protected-resource/mcp/{server_name}"

    def _get_passthrough_www_authenticate(
        scope: Scope,
        server_name: str,
        invalid_token: bool = False,
    ) -> str:
        resource_metadata_url = _get_passthrough_resource_metadata_url(
            scope=scope,
            server_name=server_name,
        )
        params = []
        if invalid_token:
            params.append('error="invalid_token"')
        params.append(f'resource_metadata="{resource_metadata_url}"')
        return "Bearer " + ", ".join(params)

    async def _raise_preemptive_401_for_unauthenticated_servers(
        scope: Scope,
        mcp_servers: Optional[List[str]],
        oauth2_headers: Optional[Dict[str, str]],
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]],
        user_api_key_auth: Optional[UserAPIKeyAuth],
        client_ip: Optional[str],
        allowed_server_ids: Optional[Set[str]] = None,
    ) -> None:
        """Fail fast with HTTP 401 for MCP servers that need user auth but
        didn't receive it on this request. Covers both gateway-managed OAuth2
        (points clients at the gateway AS metadata) and pass-through OAuth
        (points clients at the upstream resource-metadata via our well-known).

        ``allowed_server_ids`` may be passed by callers that have already
        narrowed the authorized server set (e.g. toolset scoping); servers
        not in that set are skipped so a client targeting a toolset that
        excludes a passthrough server is not pushed into an OAuth flow for
        a server it will be 403'd on immediately after authentication.
        """
        for server_name in mcp_servers or []:
            server = global_mcp_server_manager.get_mcp_server_by_name(server_name, client_ip=client_ip)
            if server is not None and allowed_server_ids is not None and server.server_id not in allowed_server_ids:
                # Caller's narrowed scope excludes this server — skip the
                # preemptive challenge and let downstream authorization
                # return 403.
                continue
            if server and server.auth_type == MCPAuth.oauth2:
                # The challenge decision is per oauth2 sub-mode, not per header:
                # gateway-managed modes (M2M and interactive authorization_code)
                # never receive a client-supplied upstream token, so a bearer in
                # Authorization is a LiteLLM key (surfaced here as oauth2_headers)
                # and must not suppress the challenge. Only the delegate mode
                # treats a present bearer as the upstream token. The sub-mode is
                # resolved the same way egress resolves it, via
                # effective_oauth2_flow: an unstamped (null oauth2_flow) row with
                # the M2M shape resolves to client_credentials, so the bare
                # has_client_credentials column is never trusted here.
                if MCPServerManager.effective_oauth2_flow(server) == "client_credentials":
                    # M2M: the gateway mints its own token at egress from the
                    # stored client credentials, so there is nothing to challenge.
                    continue

                if getattr(server, "delegate_auth_to_upstream", False) is not True:
                    # Gateway-managed interactive (authorization_code): the only
                    # thing that authorizes egress is a stored per-user token, so
                    # challenge whenever one is absent, regardless of any bearer.
                    # The v2 resolver owns the existence check, so every
                    # authorization_code resolution (egress and this discovery
                    # challenge) runs through it.
                    if await global_mcp_server_manager.has_user_oauth_token(server, user_api_key_auth):
                        continue

                    request = StarletteRequest(scope)
                    base_url = get_request_base_url(request)
                    _path = scope.get("_original_path") or scope.get("path", "") or ""

                    # Pick the well-known AS-metadata form that matches the inbound route
                    # so strict RFC 9728 §3.2 clients can resolve it correctly.
                    if _path.startswith(f"/mcp/{server_name}"):
                        _as_url = f"{base_url}/.well-known/oauth-authorization-server/mcp/{server_name}"
                    else:
                        _as_url = f"{base_url}/.well-known/oauth-authorization-server/{server_name}"
                    authorization_uri = f'Bearer authorization_uri="{_as_url}"'

                    raise HTTPException(
                        status_code=401,
                        detail="Unauthorized",
                        headers={"www-authenticate": authorization_uri},
                    )

                if not oauth2_headers:
                    # Delegate-auth servers run upstream PKCE: a present bearer is
                    # the upstream token, so only challenge when it is absent, with
                    # the proxied resource_metadata (RFC 9728), not the gateway
                    # authorization_uri above which would authorize against the
                    # gateway instead of the upstream IdP.
                    www_authenticate = _get_passthrough_www_authenticate(
                        scope=scope,
                        server_name=server_name,
                    )
                    raise HTTPException(
                        status_code=401,
                        detail="Unauthorized",
                        headers={"www-authenticate": www_authenticate},
                    )
                # Delegate server with a bearer present: it is the upstream token,
                # so admit the session and move to the next target. Every oauth2
                # sub-mode is terminal here (continue or raise) so no oauth2 server
                # reaches the token_exchange / pass-through blocks below.
                continue

            # token_exchange (OBO): the caller supplied no subject token. Challenge at connect
            # (transport level, where WWW-Authenticate survives) with the RFC 9728 resource_metadata
            # so the client discovers the IdP, SSOs, and retries with a subject token, which LiteLLM
            # then exchanges. A tool-call-time 401 would be wrapped into a JSON-RPC error and the
            # header lost, so the discovery flow needs this pre-emptive challenge.
            if server and server.auth_type == MCPAuth.oauth2_token_exchange and not oauth2_headers:
                from litellm.proxy._experimental.mcp_server.outbound_credentials.adapter import (  # noqa: PLC0415
                    raise_token_exchange_challenge,
                )
                from litellm.proxy.utils import get_server_root_path  # noqa: PLC0415

                raise_token_exchange_challenge(server, root_path=get_server_root_path())

            # token_exchange (OBO) with a subject present: run the exchange here at the transport
            # edge, so a rejected subject raises the RFC 9728 challenge (and a gateway fault its
            # public status) instead of the session opening and list_tools masking the failure as
            # an empty tool list. Gated to single-server routes; the multi-server aggregate keeps
            # absorbing per-server auth failures so one bad server cannot 401 the whole connect.
            if (
                server
                and server.auth_type == MCPAuth.oauth2_token_exchange
                and oauth2_headers
                and len(mcp_servers or []) == 1
            ):
                await global_mcp_server_manager.preflight_token_exchange(
                    server=server,
                    oauth2_headers=oauth2_headers,
                    user_api_key_auth=user_api_key_auth,
                )

            # Pass-through OAuth: when the admin has opted a server into
            # forwarding the client's bearer token (is_oauth_passthrough) and
            # the client hasn't supplied one, fail fast with 401 and point
            # them at the gateway's oauth-protected-resource well-known URL.
            # That endpoint proxies the upstream's metadata so the client
            # kicks off OAuth against the real upstream IdP, not the gateway.
            if (
                server
                and server.is_oauth_passthrough
                and not _client_has_passthrough_authorization(server, oauth2_headers, mcp_server_auth_headers)
            ):
                www_authenticate = _get_passthrough_www_authenticate(
                    scope=scope,
                    server_name=server_name,
                )
                raise HTTPException(
                    status_code=401,
                    detail="Unauthorized",
                    headers={"www-authenticate": www_authenticate},
                )

            if (
                server
                and server.is_oauth_delegate
                and len(mcp_servers or []) == 1
                and _get_forwarded_auth_from_scope(scope) is None
                and not _client_has_per_server_auth_header(server, mcp_server_auth_headers)
            ):
                www_authenticate = _get_passthrough_www_authenticate(
                    scope=scope,
                    server_name=server_name,
                )
                raise HTTPException(
                    status_code=401,
                    detail="Unauthorized",
                    headers={"www-authenticate": www_authenticate},
                )

            if (
                server
                and server.is_true_passthrough
                and len(mcp_servers or []) == 1
                and not _scope_has_authorization_header(scope)
                and not _client_has_per_server_auth_header(server, mcp_server_auth_headers)
            ):
                if server.is_dcr_bridge:
                    raise HTTPException(
                        status_code=401,
                        detail="Unauthorized",
                        headers={
                            "www-authenticate": _get_passthrough_www_authenticate(
                                scope=scope,
                                server_name=server_name,
                            )
                        },
                    )
                upstream_status, upstream_www_authenticate = await _probe_upstream_auth(server.url or "", "")
                if upstream_status == 401 and upstream_www_authenticate:
                    raise HTTPException(
                        status_code=401,
                        detail="Unauthorized",
                        headers={"www-authenticate": upstream_www_authenticate},
                    )

    def _get_authorization_header_from_scope(scope: Scope) -> Optional[str]:
        """First ``Authorization`` header value in the ASGI scope, or None."""
        for key, value in scope.get("headers", []):
            if key.lower() == b"authorization":
                return value.decode("latin-1")
        return None

    def _scope_has_authorization_header(scope: Scope) -> bool:
        return _get_authorization_header_from_scope(scope) is not None

    def _get_forwarded_auth_from_scope(scope: Scope) -> Optional[str]:
        """Return the upstream-bound ``Authorization`` header value, or None.

        Only returns the ``Authorization`` header when ``x-litellm-api-key`` is
        also present. In that case ``Authorization`` is unambiguously the
        upstream token the caller wants forwarded to the MCP server. When
        ``x-litellm-api-key`` is absent the ``Authorization`` header may itself
        be the LiteLLM proxy API key (backward-compat path in
        ``MCPRequestHandler.process_mcp_request``), and forwarding it upstream
        would leak the proxy key to a third-party MCP server.
        """
        has_litellm_key_header = any(key.lower() == b"x-litellm-api-key" for key, _ in scope.get("headers", []))
        if not has_litellm_key_header:
            return None
        return _get_authorization_header_from_scope(scope)

    def _is_delegate_upstream_probe_target(server: MCPServer) -> bool:
        """Whether ``server`` is an interactive delegate-auth server whose client-supplied
        token should be preflighted upstream.

        Mirrors the anonymous-delegate gate in ``get_allowed_mcp_servers``: the flow is
        resolved via ``effective_oauth2_flow`` so an unstamped M2M-shape row fails closed
        (its stored client credentials drive egress; the caller's bearer is irrelevant).
        """
        return (
            server.auth_type == MCPAuth.oauth2
            and server.delegate_auth_to_upstream is True
            and MCPServerManager.effective_oauth2_flow(server) != "client_credentials"
        )

    async def _probe_upstream_auth(
        url: str,
        auth_header: str,
        timeout: float = 5.0,
    ) -> tuple[int, Optional[str]]:
        """JSON-RPC initialize-probe the upstream URL to check whether the token is accepted.

        Uses POST so StreamableHTTP MCP servers run the same auth path as a
        real client request. Returns (status_code, www_authenticate).
        Fails-open with (200, None) on network errors so a transient hiccup
        does not block valid requests.

        Uses the public ``AsyncHTTPHandler.post()`` interface and catches
        ``httpx.HTTPStatusError`` separately so the 401/403 we want to surface
        is not swallowed by the broad fail-open ``except Exception`` below.
        """
        client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.MCP,
            params={"timeout": timeout},
        )
        probe_payload = {
            "jsonrpc": "2.0",
            "id": "litellm-mcp-auth-probe",
            "method": "initialize",
            "params": {
                "protocolVersion": MCPSpecVersion.jun_2025.value,
                "capabilities": {},
                "clientInfo": {
                    "name": "litellm-mcp-auth-probe",
                    "version": "1.0.0",
                },
            },
        }
        probe_headers = {
            "Accept": "application/json, text/event-stream",
            **({"Authorization": auth_header} if auth_header else {}),
        }
        try:
            resp = await client.post(
                url=url,
                headers=probe_headers,
                json=probe_payload,
                timeout=timeout,
            )
            return resp.status_code, resp.headers.get("www-authenticate")
        except httpx.HTTPStatusError as exc:
            # AsyncHTTPHandler.post() calls raise_for_status(); a 401/403 from
            # upstream lands here. Return its status so the caller can map it
            # to the appropriate response.
            return exc.response.status_code, exc.response.headers.get("www-authenticate")
        except Exception as exc:
            verbose_logger.debug(f"_probe_upstream_auth: probe to {url} failed ({exc}), allowing request through")
            return 200, None

    async def _check_passthrough_upstream_auth(
        scope: Scope,
        user_api_key_auth: Optional[UserAPIKeyAuth],
        mcp_servers: Optional[List[str]],
        client_ip: Optional[str],
    ) -> None:
        """Probe pass-through and delegate-auth upstream servers in parallel before the MCP session starts.

        Only servers the caller's key is already authorized to reach are probed —
        the list is derived from _get_allowed_mcp_servers so that a user cannot
        trigger an upstream probe against a server their key is not permitted for.

        The MCP SDK commits HTTP 200 headers before invoking handlers, so a 401
        can only be returned before that point. This function raises HTTPException(401)
        with a WWW-Authenticate header if any upstream rejects the client token, or 403
        if the upstream accepts it but forbids the caller.
        Fails-open: network errors are logged and the request is allowed through.

        Delegate-auth servers (``auth_type=oauth2`` + ``delegate_auth_to_upstream``)
        are probed with the caller's bare ``Authorization`` bearer. That bearer is only
        an upstream token (never a LiteLLM key) when admission took the delegate bypass,
        so the delegate target is resolved through ``get_mcp_server_by_name`` -- the same
        resolver admission used -- rather than the wider allowed-server prefix/access-group
        matching. A name that only reaches a delegate server via server_id or an access
        group would have been admitted as a real LiteLLM key, so probing it would leak that
        key upstream; requiring the admission-resolver match closes that gap. Without the
        probe a rejected token is absorbed by the tools/list handler and masked as an empty
        tool list. Gated to single-server routes so one rejected token cannot 401 a
        multi-server aggregate connect, matching the OBO preflight gating; the challenge
        echoes the requested name so aliased routes get the same resource_metadata URL as
        the tokenless preemptive challenge.
        """
        forwarded_auth = _get_forwarded_auth_from_scope(scope)
        requested_single_target = mcp_servers[0] if mcp_servers is not None and len(mcp_servers) == 1 else None
        # The bare Authorization header (no x-litellm-api-key) is a valid upstream token
        # only when admission classified it as one, i.e. the single requested name resolves
        # to a delegate server under admission's own resolver. Resolve it the same way here
        # so a server_id- or access-group-named delegate (which admission would have treated
        # as a LiteLLM key) is never probed with that key.
        delegate_server = (
            global_mcp_server_manager.get_mcp_server_by_name(requested_single_target, client_ip=client_ip)
            if requested_single_target
            else None
        )
        delegate_auth = (
            _get_authorization_header_from_scope(scope)
            if delegate_server is not None and _is_delegate_upstream_probe_target(delegate_server)
            else None
        )
        if not forwarded_auth and not delegate_auth:
            return

        # Use the authorized server set, not the raw user-supplied names, so that
        # a caller cannot force a probe to a server their key is not allowed to use.
        allowed_servers = await _get_allowed_mcp_servers(
            user_api_key_auth=user_api_key_auth,
            mcp_servers=mcp_servers,
            client_ip=client_ip,
        )
        passthrough_targets: Tuple[Tuple[MCPServer, str, str], ...] = (
            tuple(
                (srv, forwarded_auth, srv.name)
                for srv in allowed_servers
                # Restrict to genuine OAuth pass-through servers (auth_type none +
                # Authorization in extra_headers). Gateway-managed OAuth2 servers
                # must not receive the ``resource_metadata=`` challenge emitted
                # below — they require ``authorization_uri=`` pointing at the
                # gateway AS metadata. ``is_oauth_passthrough`` already requires
                # ``auth_type in (None, MCPAuth.none)``, which is mutually
                # exclusive with ``has_client_credentials`` (oauth2 + M2M flow),
                # so M2M servers are implicitly excluded here.
                if srv.is_oauth_passthrough
            )
            if forwarded_auth
            else ()
        )
        # Probe the admission-resolved delegate server only when the caller is actually
        # authorized for it (present in the IP-filtered allowed set), keyed by server_id.
        delegate_targets: Tuple[Tuple[MCPServer, str, str], ...] = (
            tuple(
                (srv, delegate_auth, requested_single_target)
                for srv in allowed_servers
                if delegate_server is not None and srv.server_id == delegate_server.server_id
            )
            if delegate_auth and requested_single_target
            else ()
        )
        probe_targets = passthrough_targets + delegate_targets
        if not probe_targets:
            return

        probe_results = await asyncio.gather(
            *[_probe_upstream_auth(srv.url or "", auth_header) for srv, auth_header, _ in probe_targets]
        )
        for (srv, _, challenge_server_name), (probe_status, _) in zip(probe_targets, probe_results):
            if probe_status == 401:
                # Token is missing or expired: keep pass-through clients on the
                # protected-resource discovery flow so they re-authorize against
                # the upstream IdP metadata proxied by LiteLLM.
                www_authenticate = _get_passthrough_www_authenticate(
                    scope=scope,
                    server_name=challenge_server_name,
                    invalid_token=True,
                )
                raise HTTPException(
                    status_code=401,
                    detail="Unauthorized",
                    headers={"www-authenticate": www_authenticate},
                )
            if probe_status == 403:
                # Token is valid but the caller lacks permission — do not hint
                # at re-authorization (RFC 9110: a fresh token with the same
                # scopes would just hit 403 again and loop indefinitely).
                raise HTTPException(
                    status_code=403,
                    detail="Forbidden",
                )

    async def handle_streamable_http_mcp(scope: Scope, receive: Receive, send: Send) -> None:
        """Handle MCP requests through StreamableHTTP."""
        try:
            path = scope.get("path", "")
            (
                user_api_key_auth,
                mcp_auth_header,
                mcp_servers,
                mcp_server_auth_headers,
                oauth2_headers,
                raw_headers,
            ) = await extract_mcp_auth_context(scope, path)
            scoped_server_endpoint = len(_get_mcp_servers_in_path(path) or []) == 1

            # Extract client IP for MCP access control
            _client_ip = IPAddressUtils.get_mcp_client_ip(StarletteRequest(scope))

            verbose_logger.debug(f"MCP request mcp_servers (header/path): {mcp_servers}")
            verbose_logger.debug(
                f"MCP server auth headers: {list(mcp_server_auth_headers.keys()) if mcp_server_auth_headers else None}"
            )

            # Strip any client-supplied x-mcp-toolset-id to prevent forgery.
            scope["headers"] = [(k, v) for k, v in scope.get("headers", []) if k.lower() != b"x-mcp-toolset-id"]

            # Apply toolset scope if set server-side via ContextVar (set by
            # /toolset/{name}/mcp and /{name}/mcp route handlers in proxy_server.py).
            active_toolset_id = _mcp_active_toolset_id.get()
            toolset_allowed_server_ids: Optional[Set[str]] = None
            if active_toolset_id and user_api_key_auth is not None:
                user_api_key_auth = await _apply_toolset_scope(user_api_key_auth, active_toolset_id)
                op = user_api_key_auth.object_permission
                toolset_allowed_server_ids = set(op.mcp_servers or []) if op else set()

            # https://datatracker.ietf.org/doc/html/rfc9728#name-www-authenticate-response
            # Must run after toolset scoping so the challenge set is derived
            # from the fully-authorized server set: a passthrough server that
            # the active toolset excludes should not trigger an OAuth flow
            # for a server the caller will be 403'd on after authentication.
            await _raise_preemptive_401_for_unauthenticated_servers(
                scope=scope,
                mcp_servers=mcp_servers,
                oauth2_headers=oauth2_headers,
                mcp_server_auth_headers=mcp_server_auth_headers,
                user_api_key_auth=user_api_key_auth,
                client_ip=_client_ip,
                allowed_server_ids=toolset_allowed_server_ids,
            )

            # Pre-flight auth check for pass-through servers.  Must run after
            # toolset scoping so the probe list is derived from the fully-authorized
            # server set, not the raw user-supplied names.
            await _check_passthrough_upstream_auth(scope, user_api_key_auth, mcp_servers, _client_ip)

            # Inject masked debug headers when client sends x-litellm-mcp-debug: true
            _debug_headers = MCPDebug.maybe_build_debug_headers(
                raw_headers=raw_headers,
                scope=dict(scope),
                mcp_servers=mcp_servers,
                mcp_auth_header=mcp_auth_header,
                mcp_server_auth_headers=mcp_server_auth_headers,
                oauth2_headers=oauth2_headers,
                client_ip=_client_ip,
            )
            if _debug_headers:
                send = MCPDebug.wrap_send_with_debug_headers(send, _debug_headers)

            # Ensure session managers are initialized
            if not _SESSION_MANAGERS_INITIALIZED:
                await initialize_session_managers()
                # Give it a moment to start up
                await asyncio.sleep(0.1)

            # Route based on mcp-session-id and request method:
            # - Has session ID → stateful (Claude Code, Cursor, VSCode)
            # - No session ID + initialize → stateful (so client gets mcp-session-id)
            # - No session ID + other → stateless (curl, Inspector, Notion)
            session_id = _get_session_id_from_scope(scope)
            is_initialize = False
            consumed_messages: List[Message] = []

            # Owner-binding: a live stateful session may only be driven by the
            # caller that created it. Reject mismatches with 403 so a leaked
            # mcp-session-id cannot be hijacked by another authenticated user.
            #
            # Run before ``_handle_stale_mcp_session`` so a non-owner cannot
            # force-clean another caller's residual tracking entries via a
            # stale DELETE, and before peeking the request body so the 403
            # response sees a pristine ``receive`` channel.
            if session_id:
                expected_owner = _stateful_session_owners.get(session_id)
                request_owner = _owner_fingerprint_for(user_api_key_auth, oauth2_headers, _client_ip)
                if expected_owner is not None and expected_owner != request_owner:
                    verbose_logger.warning(
                        "Rejecting MCP request: session '%s' owner mismatch.",
                        session_id,
                    )
                    forbidden_response = JSONResponse(
                        status_code=403,
                        content={
                            "error": "Forbidden",
                            "details": "mcp-session-id is bound to a different caller.",
                        },
                    )
                    await forbidden_response(scope, receive, send)
                    return

            # Handle stale session IDs before choosing a target manager. Stale
            # non-DELETE requests have their session header stripped and should
            # be routed as no-session requests.
            if session_id:
                handled = await _handle_stale_mcp_session(scope, receive, send, session_manager_stateful)
                if handled:
                    # Request was fully handled (e.g., DELETE on non-existent session)
                    return
                session_id = _get_session_id_from_scope(scope)

            body = b""
            if scope.get("method") == "POST":
                consumed_messages, body = await _read_request_body_for_routing(receive)
                is_initialize = _is_initialize_request(body)

            use_stateful = bool(session_id or is_initialize)
            target_manager = session_manager_stateful if use_stateful else session_manager_stateless

            verbose_logger.debug(
                f"MCP routing to {'stateful' if use_stateful else 'stateless'} manager"
                + (f" (session={session_id[:8]}...)" if session_id else "")
                + (" (initialize)" if is_initialize else "")
            )

            # A new `initialize` (no session id) is about to create a stateful
            # session. Cap how many a single caller can hold so an authenticated
            # client cannot spam `initialize` and exhaust memory.
            if is_initialize and not session_id:
                request_owner = _owner_fingerprint_for(user_api_key_auth, oauth2_headers, _client_ip)
                if not await _enforce_stateful_session_cap_for_owner(request_owner):
                    verbose_logger.warning(
                        "Rejecting MCP initialize: caller already holds the maximum number of active stateful sessions."
                    )
                    too_many_response = JSONResponse(
                        status_code=429,
                        content={
                            "error": "Too Many Requests",
                            "details": "Too many active MCP sessions for this caller.",
                        },
                    )
                    await too_many_response(scope, receive, send)
                    return

            # Replay body messages if we consumed them for peeking
            original_receive = receive
            if consumed_messages:

                async def wrapped_receive():
                    if consumed_messages:
                        return consumed_messages.pop(0)
                    return await original_receive()

                receive = wrapped_receive

            # Serialize requests on the same stateful session so concurrent
            # callers don't clobber each other's auth context mid-flight.
            #
            # Skip the lock for streaming GETs (SSE channels held open for the
            # life of the session): holding a per-session lock for a long-lived
            # stream would block every subsequent POST on the same session.
            # POST/DELETE are the methods that actually mutate the shared
            # auth context, so serializing those is sufficient for the
            # clobbering race between concurrent JSON-RPC calls.
            #
            # Also skip the lock for JSON-RPC *responses* (POSTs that carry
            # a ``result`` or ``error`` but no ``method``). These are replies
            # to server-initiated requests such as ``elicitation/create`` or
            # ``sampling/createMessage``. The in-flight tool-call POST that
            # triggered the server request already holds the session lock, so
            # trying to acquire it again for the response POST would deadlock.
            is_jsonrpc_response = False
            request_method = (scope.get("method") or "").upper()
            if body and request_method == "POST":
                try:
                    _peeked = json.loads(body)
                    if (
                        isinstance(_peeked, dict)
                        and _peeked.get("jsonrpc") == "2.0"
                        and "id" in _peeked
                        and "method" not in _peeked
                        and ("result" in _peeked or "error" in _peeked)
                    ):
                        is_jsonrpc_response = True
                        verbose_logger.debug(
                            "MCP: detected JSON-RPC response POST (id=%s), skipping session lock to avoid deadlock",
                            _peeked.get("id"),
                        )
                except (json.JSONDecodeError, TypeError):
                    # Peek cap truncated the body, so it can't be fully parsed.
                    # Scan the top-level keys (depth-aware) instead of a flat
                    # substring search: a response's result payload may nest a
                    # "method" field, and misreading that would acquire the lock
                    # and deadlock the in-flight tool call awaiting this
                    # response. A false skip is harmless; a false acquire is not.
                    _body_str = body.decode("utf-8", errors="replace")
                    if (
                        '"jsonrpc"' in _body_str
                        and ('"result"' in _body_str or '"error"' in _body_str)
                        and not _jsonrpc_text_has_top_level_method(_body_str)
                    ):
                        is_jsonrpc_response = True
                        verbose_logger.debug(
                            "MCP: detected truncated JSON-RPC response POST via "
                            "top-level key scan, skipping session lock to avoid deadlock"
                        )

            session_lock: Optional[asyncio.Lock] = None
            if use_stateful and session_id and request_method in ("POST", "DELETE") and not is_jsonrpc_response:
                session_lock = _stateful_session_locks.setdefault(session_id, asyncio.Lock())

            active_request_session_ids: List[str] = []

            def _increment_active_request_session(session_id_to_track: str) -> None:
                if session_id_to_track in active_request_session_ids:
                    return
                active_request_session_ids.append(session_id_to_track)
                _stateful_session_active_request_counts[session_id_to_track] = (
                    _stateful_session_active_request_counts.get(session_id_to_track, 0) + 1
                )

            if use_stateful and session_id:
                _increment_active_request_session(session_id)

            def _track_initialized_stateful_session(
                initialized_session_id: str,
            ) -> None:
                _increment_active_request_session(initialized_session_id)

            async def _dispatch() -> None:
                auth_user = _set_or_update_auth_context(
                    user_api_key_auth=user_api_key_auth,
                    mcp_auth_header=mcp_auth_header,
                    mcp_servers=mcp_servers,
                    mcp_server_auth_headers=mcp_server_auth_headers,
                    oauth2_headers=oauth2_headers,
                    raw_headers=raw_headers,
                    client_ip=_client_ip,
                    session_id=session_id if use_stateful else None,
                    touch_last_seen=(scope.get("method") or "").upper() != "DELETE",
                    copy_existing_session_auth_context=is_initialize,
                )
                local_send = send
                if use_stateful and is_initialize:
                    local_send = _wrap_send_with_stateful_session_auth_context(
                        local_send,
                        auth_user,
                        _owner_fingerprint_for(user_api_key_auth, oauth2_headers, _client_ip),
                        _track_initialized_stateful_session,
                    )

                async with _gateway_initialize_instructions_request_scope(
                    user_api_key_auth,
                    mcp_servers,
                    _client_ip,
                    scoped_server_endpoint=scoped_server_endpoint,
                ):
                    await target_manager.handle_request(scope, receive, local_send)
                    if use_stateful and session_id and scope.get("method") == "DELETE":
                        _remove_stateful_session_tracking(session_id)

            try:
                if session_lock is not None:
                    async with session_lock:
                        await _dispatch()
                else:
                    await _dispatch()
            finally:
                for active_request_session_id in active_request_session_ids:
                    active_request_count = _stateful_session_active_request_counts.get(active_request_session_id, 0) - 1
                    if active_request_count > 0:
                        _stateful_session_active_request_counts[active_request_session_id] = active_request_count
                    else:
                        _stateful_session_active_request_counts.pop(active_request_session_id, None)

                    if scope.get("method") != "DELETE" and active_request_session_id in _stateful_session_auth_contexts:
                        _stateful_session_auth_context_last_seen[active_request_session_id] = time.monotonic()

                    # Periodic cleanup iterates _stateful_session_auth_context_last_seen,
                    # so locks for untracked sessions must be dropped here.
                    if active_request_count <= 0 and active_request_session_id not in _stateful_session_auth_contexts:
                        _stateful_session_locks.pop(active_request_session_id, None)
        except MCPUpstreamAuthError as e:
            # Upstream delegated auth returned 401; surface it to the client so
            # standards-compliant MCP clients trigger the upstream OAuth flow.
            raise e.to_http_exception(
                base_url=get_request_base_url(StarletteRequest(scope)),
                request_path=scope.get("_original_path") or scope.get("path"),
            )
        except HTTPException:
            # Re-raise HTTP exceptions to preserve status codes and details
            raise
        except ProxyException as e:
            # Auth failures from user_api_key_auth arrive as ProxyException, not
            # HTTPException. Preserve the real status (e.g. 401 + WWW-Authenticate)
            # so OAuth clients can re-authenticate instead of receiving a generic
            # 500 that surfaces as a cancelled tool call.
            raise _proxy_exception_to_http_exception(e)
        except Exception as e:
            verbose_logger.exception(f"Error handling MCP request: {e}")
            # Try to send a graceful error response for non-HTTP exceptions
            try:
                from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR

                error_response = JSONResponse(
                    status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                    content={"error": "MCP request failed", "details": str(e)},
                )
                await error_response(scope, receive, send)
            except Exception as response_error:
                verbose_logger.exception(f"Failed to send error response: {response_error}")
                # If we can't send a proper response, re-raise the original error
                raise e

    async def handle_sse_mcp(scope: Scope, receive: Receive, send: Send) -> None:
        """Handle MCP requests through SSE."""
        try:
            path = scope.get("path", "")
            (
                user_api_key_auth,
                mcp_auth_header,
                mcp_servers,
                mcp_server_auth_headers,
                oauth2_headers,
                raw_headers,
            ) = await extract_mcp_auth_context(scope, path)
            scoped_server_endpoint = len(_get_mcp_servers_in_path(path) or []) == 1

            # Extract client IP for MCP access control
            _sse_client_ip = IPAddressUtils.get_mcp_client_ip(StarletteRequest(scope))

            verbose_logger.debug(f"MCP request mcp_servers (header/path): {mcp_servers}")
            verbose_logger.debug(
                f"MCP server auth headers: {list(mcp_server_auth_headers.keys()) if mcp_server_auth_headers else None}"
            )

            # Strip any client-supplied x-mcp-toolset-id to prevent forgery.
            scope["headers"] = [(k, v) for k, v in scope.get("headers", []) if k.lower() != b"x-mcp-toolset-id"]

            # Apply toolset scope if set server-side via ContextVar so the
            # downstream probe list matches the fully-authorized server set
            # (mirrors the streamable HTTP handler).
            active_toolset_id = _mcp_active_toolset_id.get()
            toolset_allowed_server_ids: Optional[Set[str]] = None
            if active_toolset_id and user_api_key_auth is not None:
                user_api_key_auth = await _apply_toolset_scope(user_api_key_auth, active_toolset_id)
                op = user_api_key_auth.object_permission
                toolset_allowed_server_ids = set(op.mcp_servers or []) if op else set()

            # https://datatracker.ietf.org/doc/html/rfc9728#name-www-authenticate-response
            # Must run after toolset scoping so the challenge set is derived
            # from the fully-authorized server set: a passthrough server that
            # the active toolset excludes should not trigger an OAuth flow
            # for a server the caller will be 403'd on after authentication.
            await _raise_preemptive_401_for_unauthenticated_servers(
                scope=scope,
                mcp_servers=mcp_servers,
                oauth2_headers=oauth2_headers,
                mcp_server_auth_headers=mcp_server_auth_headers,
                user_api_key_auth=user_api_key_auth,
                client_ip=_sse_client_ip,
                allowed_server_ids=toolset_allowed_server_ids,
            )

            # Pre-flight auth check for pass-through servers: surface upstream
            # 401/403 as a proper challenge before the SSE session commits 200
            # headers, so clients can refresh their OAuth token instead of
            # being stuck with a silently empty tool list. Must run after
            # toolset scoping so the probe list is derived from the fully-
            # authorized server set, not the raw user-supplied names.
            await _check_passthrough_upstream_auth(scope, user_api_key_auth, mcp_servers, _sse_client_ip)
            set_auth_context(
                user_api_key_auth=user_api_key_auth,
                mcp_auth_header=mcp_auth_header,
                mcp_servers=mcp_servers,
                mcp_server_auth_headers=mcp_server_auth_headers,
                oauth2_headers=oauth2_headers,
                raw_headers=raw_headers,
                client_ip=_sse_client_ip,
            )

            if not _SESSION_MANAGERS_INITIALIZED:
                await initialize_session_managers()
                await asyncio.sleep(0.1)

            async with _gateway_initialize_instructions_request_scope(
                user_api_key_auth,
                mcp_servers,
                _sse_client_ip,
                scoped_server_endpoint=scoped_server_endpoint,
            ):
                await sse_session_manager.handle_request(scope, receive, send)
        except MCPUpstreamAuthError as e:
            # Upstream delegated auth returned 401; surface it to the client so
            # standards-compliant MCP clients trigger the upstream OAuth flow.
            raise e.to_http_exception(
                base_url=get_request_base_url(StarletteRequest(scope)),
                request_path=scope.get("_original_path") or scope.get("path"),
            )
        except HTTPException:
            # Re-raise HTTP exceptions to preserve status codes and details
            # (e.g. 401 + WWW-Authenticate challenges from OAuth pass-through).
            raise
        except ProxyException as e:
            # Auth failures from user_api_key_auth arrive as ProxyException, not
            # HTTPException. Preserve the real status (e.g. 401 + WWW-Authenticate)
            # so OAuth clients can re-authenticate instead of receiving a generic
            # 500 that surfaces as a cancelled tool call.
            raise _proxy_exception_to_http_exception(e)
        except Exception as e:
            verbose_logger.exception(f"Error handling MCP request: {e}")
            # Try to send a graceful error response for non-HTTP exceptions
            try:
                # Send a proper HTTP error response instead of letting the exception bubble up
                from starlette.responses import JSONResponse
                from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR

                error_response = JSONResponse(
                    status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                    content={"error": "MCP request failed", "details": str(e)},
                )
                await error_response(scope, receive, send)
            except Exception as response_error:
                verbose_logger.exception(f"Failed to send error response: {response_error}")
                # If we can't send a proper response, re-raise the original error
                raise e

    app = FastAPI(
        title=LITELLM_MCP_SERVER_NAME,
        description=LITELLM_MCP_SERVER_DESCRIPTION,
        version=LITELLM_MCP_SERVER_VERSION,
        lifespan=lifespan,
    )

    # Routes
    @app.get(
        "/enabled",
        description="Returns if the MCP server is enabled",
    )
    def get_mcp_server_enabled() -> Dict[str, bool]:
        """
        Returns if the MCP server is enabled
        """
        return {"enabled": MCP_AVAILABLE}

    # Mount the MCP handlers
    app.mount("/", handle_streamable_http_mcp)
    app.mount("/mcp", handle_streamable_http_mcp)
    app.mount("/{mcp_server_name}/mcp", handle_streamable_http_mcp)
    app.mount("/sse", handle_sse_mcp)
    app.add_middleware(AuthContextMiddleware)

    ########################################################
    ############ Auth Context Functions ####################
    ########################################################

    def _update_auth_context(
        auth_user: MCPAuthenticatedUser,
        user_api_key_auth: Optional[UserAPIKeyAuth],
        mcp_auth_header: Optional[str] = None,
        mcp_servers: Optional[List[str]] = None,
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]] = None,
        oauth2_headers: Optional[Dict[str, str]] = None,
        raw_headers: Optional[Dict[str, str]] = None,
        client_ip: Optional[str] = None,
    ) -> None:
        auth_user.user_api_key_auth = user_api_key_auth
        auth_user.mcp_auth_header = mcp_auth_header
        auth_user.mcp_servers = mcp_servers
        auth_user.mcp_server_auth_headers = mcp_server_auth_headers or {}
        auth_user.oauth2_headers = oauth2_headers
        auth_user.raw_headers = raw_headers
        auth_user.client_ip = client_ip

    def set_auth_context(
        user_api_key_auth: Optional[UserAPIKeyAuth],
        mcp_auth_header: Optional[str] = None,
        mcp_servers: Optional[List[str]] = None,
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]] = None,
        oauth2_headers: Optional[Dict[str, str]] = None,
        raw_headers: Optional[Dict[str, str]] = None,
        client_ip: Optional[str] = None,
    ) -> MCPAuthenticatedUser:
        """
        Set the UserAPIKeyAuth in the auth context variable.

        Args:
            user_api_key_auth: UserAPIKeyAuth object
            mcp_auth_header: MCP auth header to be passed to the MCP server (deprecated)
            mcp_servers: Optional list of server names and access groups to filter by
            mcp_server_auth_headers: Optional dict of server-specific auth headers {server_alias: auth_value}
            client_ip: Client IP address for MCP access control
        """
        auth_user = MCPAuthenticatedUser(
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=mcp_auth_header,
            mcp_servers=mcp_servers,
            mcp_server_auth_headers=mcp_server_auth_headers,
            oauth2_headers=oauth2_headers,
            raw_headers=raw_headers,
            client_ip=client_ip,
        )
        auth_context_var.set(auth_user)
        return auth_user

    def _set_or_update_auth_context(
        user_api_key_auth: Optional[UserAPIKeyAuth],
        mcp_auth_header: Optional[str] = None,
        mcp_servers: Optional[List[str]] = None,
        mcp_server_auth_headers: Optional[Dict[str, Dict[str, str]]] = None,
        oauth2_headers: Optional[Dict[str, str]] = None,
        raw_headers: Optional[Dict[str, str]] = None,
        client_ip: Optional[str] = None,
        session_id: Optional[str] = None,
        touch_last_seen: bool = True,
        copy_existing_session_auth_context: bool = False,
    ) -> MCPAuthenticatedUser:
        auth_user = _stateful_session_auth_contexts.get(session_id) if session_id else None
        if auth_user is not None and session_id is not None:
            if touch_last_seen:
                _stateful_session_auth_context_last_seen[session_id] = time.monotonic()
            if copy_existing_session_auth_context:
                return set_auth_context(
                    user_api_key_auth=user_api_key_auth,
                    mcp_auth_header=mcp_auth_header,
                    mcp_servers=mcp_servers,
                    mcp_server_auth_headers=mcp_server_auth_headers,
                    oauth2_headers=oauth2_headers,
                    raw_headers=raw_headers,
                    client_ip=client_ip,
                )
            _update_auth_context(
                auth_user=auth_user,
                user_api_key_auth=user_api_key_auth,
                mcp_auth_header=mcp_auth_header,
                mcp_servers=mcp_servers,
                mcp_server_auth_headers=mcp_server_auth_headers,
                oauth2_headers=oauth2_headers,
                raw_headers=raw_headers,
                client_ip=client_ip,
            )
            auth_context_var.set(auth_user)
            return auth_user
        return set_auth_context(
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=mcp_auth_header,
            mcp_servers=mcp_servers,
            mcp_server_auth_headers=mcp_server_auth_headers,
            oauth2_headers=oauth2_headers,
            raw_headers=raw_headers,
            client_ip=client_ip,
        )

    def _wrap_send_with_stateful_session_auth_context(
        send: Send,
        auth_user: MCPAuthenticatedUser,
        owner_fingerprint: str,
        on_session_registered: Optional[Callable[[str], None]] = None,
    ) -> Send:
        async def wrapped_send(message: Message) -> None:
            if message.get("type") == "http.response.start":
                for key, value in message.get("headers", []):
                    header_name = key if isinstance(key, bytes) else str(key).encode()
                    if header_name.lower() == b"mcp-session-id":
                        session_id = value.decode() if isinstance(value, bytes) else str(value)
                        if on_session_registered is not None:
                            on_session_registered(session_id)
                        auth_context_var.set(auth_user)
                        _stateful_session_auth_contexts[session_id] = auth_user
                        _stateful_session_auth_context_last_seen[session_id] = time.monotonic()
                        _stateful_session_owners[session_id] = owner_fingerprint
                        break
            await send(message)

        return wrapped_send

    def get_auth_context() -> Tuple[
        Optional[UserAPIKeyAuth],
        Optional[str],
        Optional[List[str]],
        Optional[Dict[str, Dict[str, str]]],
        Optional[Dict[str, str]],
        Optional[Dict[str, str]],
        Optional[str],
    ]:
        """
        Get the UserAPIKeyAuth from the auth context variable.

        Returns:
            Tuple containing: UserAPIKeyAuth, MCP auth header (deprecated),
            MCP servers, server-specific auth headers, OAuth2 headers, raw headers, client IP
        """
        auth_user = auth_context_var.get()
        if auth_user and isinstance(auth_user, MCPAuthenticatedUser):
            return (
                auth_user.user_api_key_auth,
                auth_user.mcp_auth_header,
                auth_user.mcp_servers,
                auth_user.mcp_server_auth_headers,
                auth_user.oauth2_headers,
                auth_user.raw_headers,
                auth_user.client_ip,
            )
        return None, None, None, None, None, None, None

    def _get_current_session():
        try:
            from mcp.server.lowlevel.server import request_ctx

            return request_ctx.get().session
        except (LookupError, ImportError):
            return None

    def _cache_auth_context_lazily():
        session = _get_current_session()
        if session is None:
            return
        try:
            if session in _session_obj_auth_storage:
                return
        except TypeError:
            verbose_logger.debug(
                "_cache_auth_context_lazily: session object is unhashable (type=%s), cannot cache auth context",
                type(session).__name__,
            )
            return

        auth = auth_context_var.get()
        if auth and isinstance(auth, MCPAuthenticatedUser):
            try:
                _session_obj_auth_storage[session] = auth
            except TypeError:
                verbose_logger.debug(
                    "_cache_auth_context_lazily: could not store auth via "
                    "session identity — session object is unhashable"
                )

    def _recover_auth_from_session() -> Optional[MCPAuthenticatedUser]:
        session = _get_current_session()
        if session is None:
            return None

        stored: Optional[MCPAuthenticatedUser] = None
        try:
            stored = _session_obj_auth_storage.get(session)
        except TypeError:
            verbose_logger.debug(
                "_recover_auth_from_session: session object is unhashable "
                "(type=%s), skipping _session_obj_auth_storage lookup",
                type(session).__name__,
            )

        return stored

    async def get_or_extract_auth_context() -> Tuple[
        Optional[UserAPIKeyAuth],
        Optional[str],
        Optional[List[str]],
        Optional[Dict[str, Dict[str, str]]],
        Optional[Dict[str, str]],
        Optional[Dict[str, str]],
        Optional[str],
    ]:
        """
        Get auth context from ContextVar first, then fall back to session
        storage (which survives cross-task boundaries in the MCP SDK).
        """
        (
            user_api_key_auth,
            mcp_auth_header,
            mcp_servers,
            mcp_server_auth_headers,
            oauth2_headers,
            raw_headers,
            _client_ip,
        ) = get_auth_context()

        if user_api_key_auth is not None:
            _cache_auth_context_lazily()
        else:
            stored = _recover_auth_from_session()

            if stored:
                user_api_key_auth = stored.user_api_key_auth
                mcp_auth_header = stored.mcp_auth_header
                mcp_servers = stored.mcp_servers
                mcp_server_auth_headers = stored.mcp_server_auth_headers
                oauth2_headers = stored.oauth2_headers
                raw_headers = stored.raw_headers
                _client_ip = stored.client_ip
        return (
            user_api_key_auth,
            mcp_auth_header,
            mcp_servers,
            mcp_server_auth_headers,
            oauth2_headers,
            raw_headers,
            _client_ip,
        )

    def get_active_mcp_session() -> Optional[_McpServerSession]:
        """Return the active MCP session captured during handler execution."""
        session = active_mcp_session_var.get()
        if session is not None:
            return session
        return _get_current_session()

    def get_active_auth_context() -> Optional[MCPAuthenticatedUser]:
        """Return auth context from ContextVar or session storage."""
        auth = auth_context_var.get()
        if auth and isinstance(auth, MCPAuthenticatedUser):
            return auth

        stored = _recover_auth_from_session()
        if stored is not None:
            return stored
        return None

    ########################################################
    ############ End of Auth Context Functions #############
    ########################################################

else:
    app = FastAPI()
