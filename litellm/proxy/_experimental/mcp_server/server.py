"""
LiteLLM MCP Server Routes
"""

# pyright: reportInvalidTypeForm=false, reportArgumentType=false, reportOptionalCall=false

import asyncio
import contextlib
import contextvars
import hashlib
import json
import os
import time
import types
from collections import Counter
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Final, NoReturn, Protocol

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import ConfigDict, TypeAdapter, ValidationError
from starlette.requests import Request as StarletteRequest
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.types import Message, Receive, Scope, Send

from litellm._logging import verbose_logger
from litellm.constants import (
    MCP_GATEWAY_SESSION_ID_PREFIX_LENGTH,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
    MCPRequestHandler,
    _is_mcp_admitted_user_subject,
)
from litellm.proxy._experimental.mcp_server.caller_sign_in import caller_sign_in_for
from litellm.proxy._experimental.mcp_server.client_allowlist import (
    MCPClientAllowlist,
    check_mcp_client_allowed,
    load_mcp_client_allowlist,
)
from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
    get_request_base_url,
)
from litellm.proxy._experimental.mcp_server.exceptions import (
    MCPUpstreamAuthError,
)
from litellm.proxy._experimental.mcp_server.mcp_context import (
    _mcp_active_toolset_id,
    _mcp_gateway_initialize_instructions,
    _mcp_gateway_server_name,
    _mcp_proxy_mode,  # pyright: ignore[reportPrivateUsage]  # server-owned request mode
    active_mcp_request_ctx_var,
    get_active_mcp_request_ctx,
)
from litellm.proxy._experimental.mcp_server.mcp_debug import (
    MCP_AUTH_DIAGNOSTICS_SCOPE_KEY,
    MCPAuthDiagnostics,
    MCPDebug,
)
from litellm.proxy._experimental.mcp_server.oauth_utils import (
    _redact_mcp_resource_url,
    get_passthrough_www_authenticate,
    get_route_relative_request_path,
    well_known_root_suffix,
)
from litellm.proxy._experimental.mcp_server.ui_session_utils import is_ui_session_credential
from litellm.proxy._experimental.mcp_server.utils import (
    LITELLM_MCP_SERVER_DESCRIPTION,
    LITELLM_MCP_SERVER_NAME,
    LITELLM_MCP_SERVER_VERSION,
)
from litellm.proxy._types import (
    ProxyException,
    SpecialMCPServerNames,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.ip_address_utils import IPAddressUtils
from litellm.types.mcp import (
    MCPAuth,
    MCPGatewaySession,
    MCPGatewaySessionGroupCount,
    MCPGatewaySessionsResponse,
    MCPGatewaySessionsTerminateResponse,
    MCPSpecVersion,
)
from litellm.types.mcp_server.mcp_server_manager import MCPServer

if TYPE_CHECKING:
    from mcp.server.session import ServerSession as _McpServerSession


_STATEFUL_SESSION_IDLE_TIMEOUT_SECONDS: Final = 30 * 60
# Upper bound on concurrent stateful sessions a single caller may hold. Each
# `initialize` creates a session that survives until the idle timeout, so
# without a cap an authenticated client could spam `initialize` and exhaust
# memory. The caller's own oldest idle sessions are evicted to make room; if
# the cap is still hit (every session in flight), the new `initialize` is
# rejected with 429.
_MAX_STATEFUL_SESSIONS_PER_OWNER: Final = 100
# Maximum bytes to peek when sniffing the JSON-RPC method on a POST.
# An `initialize` envelope is a few hundred bytes; capping the peek
# prevents an authenticated client from forcing the proxy to buffer an
# arbitrarily large body just to make a routing decision.
_MCP_ROUTING_PEEK_MAX_BYTES: Final = 4096
# ASGI scope keys carrying OTel request state into a stateful MCP message handler.
_MCP_TRANSPORT_SPAN_SCOPE_KEY: Final = "litellm_otel_transport_span"
_MCP_DESTINATIONS_SCOPE_KEY: Final = "litellm_otel_request_destinations"
_MCP_PROTOCOL_VERSION_HEADER: Final = b"mcp-protocol-version"


def reject_disallowed_mcp_origin(request: StarletteRequest) -> None:
    from litellm.proxy.proxy_server import origins  # noqa: PLC0415  # proxy imports this module during startup

    if "*" not in origins and any(origin not in origins for origin in request.headers.getlist("origin")):
        raise HTTPException(status_code=403, detail="Invalid Origin header")


def unsupported_protocol_version(scope: Scope) -> str | None:
    """Return the unsupported ``MCP-Protocol-Version`` header value, if any.

    SDK 2's ``StreamableHTTPSessionManager`` routes any version outside
    ``HANDSHAKE_PROTOCOL_VERSIONS`` to the modern single-exchange path, which
    bypasses litellm's session/auth model, so the ASGI entry rejects it.
    """
    from litellm.proxy._experimental.mcp_server.capabilities import configured_versions

    headers: Final[Iterable[tuple[bytes, bytes]]] = scope.get("headers") or ()
    values: Final = tuple(
        raw.decode("latin-1").strip() for key, raw in headers if key.lower() == _MCP_PROTOCOL_VERSION_HEADER
    )
    for value in values:
        if value and value not in configured_versions():
            return value
    return None


# Check if MCP is available
# "mcp" requires python 3.10 or higher, but several litellm users use python 3.8
# We're making this conditional import to avoid breaking users who use python 3.8.
# TODO: Make this a util function for litellm client usage
MCP_AVAILABLE: bool = True
try:
    import weakref

    from mcp import ReadResourceResult, Resource
    from mcp.server import Server
    from mcp.server.runner import serve_loop
    from mcp.server.session import ServerSession as _McpServerSession
    from mcp.types import (
        BlobResourceContents,
        DiscoverRequest,
        DiscoverResult,
        GetPromptResult,
        RequestParams,
        ResourceTemplate,
        TextResourceContents,
    )

    # Robust auth lookup keyed by session_object.
    _session_obj_auth_storage: "weakref.WeakKeyDictionary[object, MCPAuthenticatedUser]" = weakref.WeakKeyDictionary()
except ImportError as e:
    verbose_logger.debug("MCP module not found: %s", e)
    MCP_AVAILABLE = False
    # When MCP is not available, we set these to None at module level
    # All code using these types is inside `if MCP_AVAILABLE:` blocks
    # so they will never be accessed at runtime
    BlobResourceContents = None
    GetPromptResult = None
    ReadResourceResult = None
    Resource = None
    ResourceTemplate = None
    Server = None
    TextResourceContents = None

active_mcp_session_var: Final[contextvars.ContextVar["_McpServerSession | None"]] = contextvars.ContextVar(
    "active_mcp_session", default=None
)


# Global variables to track initialization
_SESSION_MANAGERS_INITIALIZED = False
_INITIALIZATION_LOCK: Final = asyncio.Lock()


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
    in_object: Final[list[bool]] = []
    reading_key = False
    expect_key = False
    key_chars: list[str] = []
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


def _mcp_meta_trace_carrier(req_ctx: object) -> dict[str, str] | None:
    """The W3C trace context (``traceparent``/``tracestate``) the MCP client
    propagated in the request's ``params._meta`` (SEP-414), or ``None``.

    When present, the MCP span records this propagated context as a span *link*,
    never the parent — a remote parent would root the span in a trace whose root
    never reaches the gateway's tracing backend. The span itself nests under the
    transport span of the request carrying this specific message, so a
    streamable-HTTP session that multiplexes many messages still does not glue
    every message under the session's first request;
    see ``resolve_mcp_span_context``. The client's W3C Baggage is
    deliberately excluded: it is caller-controlled, and the otel baggage processor
    stamps allowlisted baggage keys (``litellm.team.id``, ``litellm.metadata.*``,
    ...) onto the span, so honoring remote baggage would let a client spoof a
    span's identity attribution.
    """
    meta: Final = getattr(req_ctx, "meta", None)
    extra: Final = meta if isinstance(meta, Mapping) else getattr(meta, "model_extra", None)
    if not isinstance(extra, Mapping):
        return None
    carrier: Final = {key: extra[key] for key in ("traceparent", "tracestate") if isinstance(extra.get(key), str)}
    return carrier or None


def _otel_set_mcp_trace_carrier(carrier: dict[str, str] | None) -> object:
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


def _otel_publish_transport_span_on_scope(scope: Scope) -> None:
    """Record this request's tracing span on its own ASGI scope.

    Resolved on the ASGI request task, where the proxy's server span is anchored,
    and read back by the MCP message handler through ``req_ctx.request`` — the
    ``Request`` the transport attaches to each message. A stateful streamable-HTTP
    session handles every message on the task spawned by its ``initialize`` POST, so
    the handler's own task cannot see later requests' spans.

    The scope, not the shared session auth context: a JSON-RPC *response* POST
    deliberately skips the per-session lock (it can arrive while the tool call that
    awaits it is still in flight), so a field on that shared object would be
    overwritten mid-call and the tool call would attribute itself to the response's
    request. A scope belongs to exactly one request and dies with it, which also
    keeps a finished span from being retained by an idle session.

    The live span, not just its context: a failed tool call stamps ``error.*`` on it,
    which needs a span still open for writes. Lazily imported so opentelemetry stays
    an optional dependency; a no-op when otel_v2 is unavailable or no request span is
    anchored."""
    try:
        from litellm.integrations.otel.plumbing.context import (
            request_root_span,
        )

        span: Final = request_root_span()
    except ImportError:
        return
    if span is not None:
        scope[_MCP_TRANSPORT_SPAN_SCOPE_KEY] = span


def _otel_value_from_message_scope(req_ctx: object, key: str) -> object:
    request: Final = getattr(req_ctx, "request", None)
    scope: Final = getattr(request, "scope", None)
    if not isinstance(scope, Mapping):
        return None
    return scope.get(key)


def _otel_transport_span_from_message(req_ctx: object) -> object:
    """The tracing span of the HTTP request that carried this MCP message."""
    return _otel_value_from_message_scope(req_ctx, _MCP_TRANSPORT_SPAN_SCOPE_KEY)


def _otel_set_mcp_transport_span(span: object) -> object:
    """Publish the current message's transport span, which the otel_v2 MCP span
    attaches to and a failed tool call stamps its error on. Returns a reset token,
    or ``None`` when otel_v2 is unavailable."""
    if span is None:
        return None
    try:
        from litellm.integrations.otel.plumbing.context import (
            set_mcp_message_transport_span,
        )

        return set_mcp_message_transport_span(span)
    except ImportError:
        return None


def _otel_reset_mcp_transport_span(token: object) -> None:
    """Paired with ``_otel_set_mcp_transport_span``."""
    if token is None:
        return
    try:
        from litellm.integrations.otel.plumbing.context import (
            reset_mcp_message_transport_span,
        )

        reset_mcp_message_transport_span(token)
    except ImportError:
        return


def _otel_publish_request_destinations_on_scope(scope: Scope) -> None:
    try:
        from litellm.integrations.otel.plumbing.context import request_destinations

        scope[_MCP_DESTINATIONS_SCOPE_KEY] = request_destinations()
    except ImportError:
        return


def _otel_set_mcp_request_destinations(req_ctx: object) -> object:
    destinations: Final = _otel_value_from_message_scope(req_ctx, _MCP_DESTINATIONS_SCOPE_KEY)
    if not isinstance(destinations, tuple):
        return None
    try:
        from litellm.integrations.otel.model.destination import OtelDestination
        from litellm.integrations.otel.plumbing.context import set_request_destinations

        destination_adapter: Final[TypeAdapter[tuple[OtelDestination, ...]]] = TypeAdapter(
            tuple[OtelDestination, ...],
            config=ConfigDict(revalidate_instances="always"),
        )
        validated_destinations: Final = destination_adapter.validate_python(destinations, strict=True)
        return set_request_destinations(validated_destinations)
    except (ImportError, ValidationError):
        return None


def _otel_reset_mcp_request_destinations(token: object) -> None:
    if token is None:
        return
    try:
        from litellm.integrations.otel.plumbing.context import reset_request_destinations

        reset_request_destinations(token)
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
    __all__ = (
        "_MCP_CREDENTIAL_REQUEST_FIELDS",
        "BlobResourceContents",
        "ListMCPToolsRestAPIResponseObject",
        "ResourceTemplate",
        "TextResourceContents",
        "_McpDeniedDetail",
        "_aggregate_server_key",
        "_build_virtual_call_logging_obj",
        "_check_byok_credential",
        "_client_has_passthrough_authorization",
        "_client_has_per_server_auth_header",
        "_dispatch_virtual_mcp_tool",
        "_fire_mcp_tool_call_logging",
        "_get_allowed_mcp_servers",
        "_get_allowed_mcp_servers_from_mcp_server_names",
        "_get_byok_credential",
        "_get_prompts_from_mcp_servers",
        "_get_resource_templates_from_mcp_servers",
        "_get_resources_from_mcp_servers",
        "_get_standard_logging_mcp_tool_call",
        "_get_tools_from_mcp_servers",
        "_get_user_oauth_extra_headers_from_db",
        "_handle_local_mcp_tool",
        "_handle_managed_mcp_tool",
        "_http_detail_message",
        "_invalidate_byok_cred_cache",
        "_list_mcp_prompts",
        "_list_mcp_resource_templates",
        "_list_mcp_resources",
        "_list_mcp_tools",
        "_list_tools_before_first_call",
        "_mcp_session_id_from_headers",
        "_merge_gateway_initialize_instructions",
        "_prefetch_oauth_creds_for_user",
        "_prepare_mcp_server_headers",
        "_raise_if_initialize_grants_no_mcp_servers",
        "_redact_mcp_resource_url",
        "_resolve_display_name_to_original",
        "_run_post_mcp_call_guardrails",
        "_server_answers_to",
        "_tool_name_matches",
        "apply_tool_overrides",
        "call_mcp_tool",
        "execute_mcp_tool",
        "filter_tools_by_allowed_tools",
        "filter_tools_by_key_team_permissions",
        "fire_mcp_tool_call_failure_logging",
        "global_mcp_server_manager",
        "mcp_get_prompt",
        "mcp_read_resource",
        "raise_denied_scoped_mcp_access",
    )
    from mcp.server import Server

    # Import auth context variables and middleware
    from mcp.server.auth.middleware.auth_context import (
        AuthContextMiddleware,
        auth_context_var,
    )
    from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
    from mcp.server.auth.provider import AccessToken
    from mcp.server.context import ServerRequestContext
    from mcp.server.lowlevel.server import NotificationOptions
    from mcp.server.models import InitializationOptions
    from mcp.shared.exceptions import MCPError
    from mcp.types import (
        CallToolRequest,
        GetPromptRequest,
        ListPromptsRequest,
        ListResourcesRequest,
        ListResourceTemplatesRequest,
        ListToolsRequest,
        ReadResourceRequest,
    )

    from litellm.proxy._experimental.mcp_server import operations
    from litellm.proxy._experimental.mcp_server.contracts import OperationContext
    from litellm.proxy._experimental.mcp_server.operations import (
        _invalidate_byok_cred_cache,
        _mcp_session_id_from_headers,
    )
    from litellm.proxy._experimental.mcp_server.result_conversion import wire_compat_for

    try:
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    except ImportError:
        StreamableHTTPSessionManager = None
    from mcp.types import (
        INVALID_REQUEST,
        CallToolRequestParams,
        CallToolResult,
        GetPromptRequestParams,
        Implementation,
        InitializeRequest,
        InputRequiredResult,
        ListPromptsResult,
        ListResourcesResult,
        ListResourceTemplatesResult,
        ListToolsResult,
        PaginatedRequestParams,
        ReadResourceRequestParams,
    )

    from litellm.proxy._experimental.mcp_server.auth.litellm_auth_handler import (
        MCPAuthenticatedUser,
    )
    from litellm.proxy._experimental.mcp_server.capabilities import GatewayVersionPolicy, configured_versions
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        MCPServerManager,
        global_mcp_server_manager,
    )

    ######################################################
    ############ MCP Tools List REST API Response Object #
    # Defined here because we don't want to add `mcp` as a
    # required dependency for `litellm` pip package
    ######################################################
    from litellm.proxy._experimental.mcp_server.operations import (
        ListMCPToolsRestAPIResponseObject,
    )
    from litellm.proxy._experimental.mcp_server.sse_transport import SseServerTransport

    def _gateway_create_initialization_options(
        self,
        notification_options: NotificationOptions | None = None,
        experimental_capabilities: dict[str, dict[str, object]] | None = None,
        extensions: dict[str, dict[str, object]] | None = None,
    ) -> InitializationOptions:
        base_options: Final = Server.create_initialization_options(
            self,
            notification_options=notification_options,
            experimental_capabilities=experimental_capabilities or {},
            extensions=extensions,
        )
        opts: Final = (
            base_options.model_copy(
                update={  # mutable-ok: Pydantic update payload
                    "capabilities": base_options.capabilities.model_copy(
                        update={"prompts": None, "resources": None}  # mutable-ok: Pydantic update payload
                    )
                }
            )
            if _mcp_proxy_mode.get()
            else base_options
        )
        updates: Final[dict[str, str]] = {}
        merged: Final = _mcp_gateway_initialize_instructions.get()
        if merged is not None:
            updates["instructions"] = merged
        scoped_server_name: Final = _mcp_gateway_server_name.get()
        if scoped_server_name is not None:
            updates["server_name"] = scoped_server_name
        return opts.model_copy(update=updates) if updates else opts

    ########################################################
    ############ Initialize the MCP Server #################
    ########################################################
    server: Final[Server] = Server(
        name=LITELLM_MCP_SERVER_NAME,
        version=LITELLM_MCP_SERVER_VERSION,
    )
    server.middleware.append(GatewayVersionPolicy())
    server.create_initialization_options = types.MethodType(_gateway_create_initialization_options, server)
    sse: Final[SseServerTransport] = SseServerTransport("/sse/messages")

    # Create session managers
    session_manager_stateless: Final = StreamableHTTPSessionManager(
        app=server,
        event_store=None,
        json_response=False,  # enables SSE streaming
        stateless=True,
    )

    session_manager_stateful: Final = StreamableHTTPSessionManager(
        app=server,
        event_store=None,  # TODO: Add EventStore for reconnection/event replay if needed
        json_response=False,  # enables SSE streaming
        stateless=False,
    )
    _stateful_session_auth_contexts: Final[dict[str, MCPAuthenticatedUser]] = {}
    _stateful_session_auth_context_last_seen: Final[dict[str, float]] = {}
    # Maps session_id -> owner identifier (hashed API key/token) so we can
    # reject requests that supply a session_id created by a different caller.
    # Without this, a leaked mcp-session-id could be driven (or terminated)
    # by any other authenticated proxy user.
    _stateful_session_owners: Final[dict[str, str]] = {}
    # Per-session lock that serializes ``handle_request`` for the same
    # mcp-session-id. The stored ``MCPAuthenticatedUser`` is mutated in place
    # by ``_update_auth_context`` each request; without this lock, two
    # concurrent requests on the same session would clobber each other's
    # auth headers / mcp_servers / oauth state while in-flight callbacks are
    # still reading the shared object.
    _stateful_session_locks: Final[dict[str, asyncio.Lock]] = {}
    _stateful_session_active_request_counts: Final[dict[str, int]] = {}
    _stateful_session_client_info: Final[dict[str, Implementation]] = {}  # mutable-ok: cleared on session teardown
    _admin_terminated_session_ids: Final[dict[str, float]] = {}  # mutable-ok: admin-closed id -> last replay

    class _TerminableTransport(Protocol):
        async def terminate(self) -> None: ...

    class _TransportRegistry(Protocol):
        def __contains__(self, session_id: object, /) -> bool: ...

        def pop(self, session_id: str, default: None, /) -> "_TerminableTransport | None": ...

    def _stateful_server_instances() -> _TransportRegistry:
        return getattr(session_manager_stateful, "_server_instances", {})

    def _remove_stateful_session_tracking(session_id: str) -> None:
        _stateful_session_auth_contexts.pop(session_id, None)
        _stateful_session_auth_context_last_seen.pop(session_id, None)
        _stateful_session_owners.pop(session_id, None)
        _stateful_session_locks.pop(session_id, None)
        _stateful_session_active_request_counts.pop(session_id, None)
        _stateful_session_client_info.pop(session_id, None)

    # Keep this alias so existing references to session_manager still work
    session_manager: Final = session_manager_stateless

    # Context managers for proper lifecycle management
    _session_manager_cm = None
    _session_manager_stateful_cm = None
    _stateful_auth_context_cleanup_task: asyncio.Task | None = None

    async def _purge_expired_stateful_session_auth_contexts(
        now: float | None = None,
    ) -> None:
        """Terminate expired stateful sessions and drop their auth contexts."""
        now = time.monotonic() if now is None else now
        server_instances: Final = _stateful_server_instances()
        expired_session_ids: Final[list[str]] = []
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
        _forget_expired_admin_terminated_session_ids(now)

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
        server_instances: Final = _stateful_server_instances()

        def _owned_live_session_ids() -> list[str]:
            return [
                session_id
                for session_id, session_owner in _stateful_session_owners.items()
                if session_owner == owner and session_id in server_instances
            ]

        owned: Final = _owned_live_session_ids()
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
                verbose_logger.exception("Error cleaning up expired MCP stateful sessions: %s", e)

    async def initialize_session_managers():
        """Initialize the session managers. Can be called from main app lifespan."""
        global \
            _SESSION_MANAGERS_INITIALIZED, \
            _session_manager_cm, \
            _session_manager_stateful_cm, \
            _stateful_auth_context_cleanup_task

        # Use async lock to prevent concurrent initialization
        async with _INITIALIZATION_LOCK:
            if _SESSION_MANAGERS_INITIALIZED:
                return

            verbose_logger.info("Initializing MCP session managers...")

            # Start the session managers with context managers
            _session_manager_cm = session_manager_stateless.run()
            _session_manager_stateful_cm = session_manager_stateful.run()

            # Enter the context managers
            await _session_manager_cm.__aenter__()
            await _session_manager_stateful_cm.__aenter__()
            _stateful_auth_context_cleanup_task = asyncio.create_task(_cleanup_expired_stateful_session_auth_contexts())

            _SESSION_MANAGERS_INITIALIZED = True
            verbose_logger.info("MCP Server started with StreamableHTTP and SSE session managers!")

    async def shutdown_session_managers():
        """Shutdown the session managers."""
        global \
            _SESSION_MANAGERS_INITIALIZED, \
            _session_manager_cm, \
            _session_manager_stateful_cm, \
            _stateful_auth_context_cleanup_task

        if _SESSION_MANAGERS_INITIALIZED:
            verbose_logger.info("Shutting down MCP session managers...")

            try:
                if _stateful_auth_context_cleanup_task:
                    _stateful_auth_context_cleanup_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await _stateful_auth_context_cleanup_task
                if _session_manager_stateful_cm:
                    await _session_manager_stateful_cm.__aexit__(None, None, None)
                if _session_manager_cm:
                    await _session_manager_cm.__aexit__(None, None, None)
            except Exception as e:
                verbose_logger.exception("Error during session manager shutdown: %s", e)

            _session_manager_cm = None
            _session_manager_stateful_cm = None
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

    @contextlib.asynccontextmanager
    async def _legacy_operation_context(ctx: ServerRequestContext, *, trace: bool) -> AsyncGenerator[OperationContext]:
        with contextlib.ExitStack() as cleanup:
            cleanup.callback(active_mcp_request_ctx_var.reset, active_mcp_request_ctx_var.set(ctx))
            cleanup.callback(active_mcp_session_var.reset, active_mcp_session_var.set(ctx.session))
            if trace:
                cleanup.callback(
                    _otel_reset_mcp_trace_carrier, _otel_set_mcp_trace_carrier(_mcp_meta_trace_carrier(ctx))
                )
                cleanup.callback(
                    _otel_reset_mcp_transport_span, _otel_set_mcp_transport_span(_otel_transport_span_from_message(ctx))
                )
                cleanup.callback(_otel_reset_mcp_request_destinations, _otel_set_mcp_request_destinations(ctx))
            (
                auth,
                token,
                servers,
                server_headers,
                oauth_headers,
                headers,
                client_ip,
            ) = await get_or_extract_auth_context()
            yield operations.prepare_context(
                auth,
                token,
                servers,
                server_headers,
                oauth_headers,
                headers,
                client_ip,
                _mcp_proxy_mode.get(),
                wire_compat_for(ctx.protocol_version),
                ctx.protocol_version,
            )

    async def handle_list_tools(ctx: ServerRequestContext, params: PaginatedRequestParams) -> ListToolsResult:
        try:
            async with _legacy_operation_context(ctx, trace=True) as context:
                return await operations.GatewayOperations(_capture_host_progress_callback(ctx)).execute(
                    ListToolsRequest(params=params), context
                )
        except MCPError:
            raise
        except HTTPException as exc:
            raise MCPError(code=INVALID_REQUEST, message=operations._http_detail_message(exc.detail)) from exc
        except Exception as exc:  # noqa: BLE001  # preserve native listing fallback for ingress failures
            verbose_logger.exception("Error in list_tools endpoint: %s", exc)
            return ListToolsResult(tools=[])

    def _capture_host_progress_callback(ctx: ServerRequestContext) -> Callable | None:
        """Return a progress-forwarding callback bound to the host MCP session.

        Returns ``None`` when the host did not supply a progress token.
        """
        host_ctx: Final = ctx

        if not (host_ctx and hasattr(host_ctx, "meta") and host_ctx.meta):
            return None
        host_token: Final = host_ctx.meta.get("progress_token")
        if host_token is None or not (hasattr(host_ctx, "session") and host_ctx.session):
            return None
        host_session: Final = host_ctx.session

        async def forward_progress(progress: float, total: float | None):
            """Forward progress notifications from external MCP to Host"""
            try:
                await host_session.send_progress_notification(
                    progress_token=host_token,
                    progress=progress,
                    total=total,
                )
                verbose_logger.debug("Forwarded progress %s/%s to Host", progress, total)
            except Exception as e:
                verbose_logger.error("Failed to forward progress to Host: %s", e)

        verbose_logger.debug("Host progressToken captured: %s...", str(host_token)[:8])
        return forward_progress

    def _reject_mcp_proxy_operation() -> NoReturn:
        from mcp.shared.exceptions import MCPError
        from mcp.types import METHOD_NOT_FOUND

        raise MCPError(code=METHOD_NOT_FOUND, message="Operation unavailable on /mcp/proxy")

    from litellm.proxy._experimental.mcp_server.operations import (
        _build_virtual_call_logging_obj,
        _dispatch_virtual_mcp_tool,
    )

    async def mcp_server_tool_call(
        ctx: ServerRequestContext, params: CallToolRequestParams
    ) -> CallToolResult | InputRequiredResult:
        async with _legacy_operation_context(ctx, trace=True) as context:
            return await operations.GatewayOperations(_capture_host_progress_callback(ctx)).execute(
                CallToolRequest(params=params), context
            )

    async def list_prompts(ctx: ServerRequestContext, params: PaginatedRequestParams) -> ListPromptsResult:
        if _mcp_proxy_mode.get():
            _reject_mcp_proxy_operation()
        try:
            async with _legacy_operation_context(ctx, trace=False) as context:
                return await operations.GatewayOperations(_capture_host_progress_callback(ctx)).execute(
                    ListPromptsRequest(params=params), context
                )
        except Exception as exc:  # noqa: BLE001  # preserve native listing fallback for ingress failures
            verbose_logger.exception("Error in list_prompts endpoint: %s", exc)
            return ListPromptsResult(prompts=[])

    async def get_prompt(ctx: ServerRequestContext, params: GetPromptRequestParams) -> GetPromptResult:
        if _mcp_proxy_mode.get():
            _reject_mcp_proxy_operation()
        async with _legacy_operation_context(ctx, trace=False) as context:
            return await operations.GatewayOperations(_capture_host_progress_callback(ctx)).execute(
                GetPromptRequest(params=params), context
            )

    async def list_resources(ctx: ServerRequestContext, params: PaginatedRequestParams) -> ListResourcesResult:
        if _mcp_proxy_mode.get():
            _reject_mcp_proxy_operation()
        try:
            async with _legacy_operation_context(ctx, trace=False) as context:
                return await operations.GatewayOperations(_capture_host_progress_callback(ctx)).execute(
                    ListResourcesRequest(params=params), context
                )
        except Exception as exc:  # noqa: BLE001  # preserve native listing fallback for ingress failures
            verbose_logger.exception("Error in list_resources endpoint: %s", exc)
            return ListResourcesResult(resources=[])

    async def list_resource_templates(
        ctx: ServerRequestContext, params: PaginatedRequestParams
    ) -> ListResourceTemplatesResult:
        if _mcp_proxy_mode.get():
            _reject_mcp_proxy_operation()
        try:
            async with _legacy_operation_context(ctx, trace=False) as context:
                return await operations.GatewayOperations(_capture_host_progress_callback(ctx)).execute(
                    ListResourceTemplatesRequest(params=params), context
                )
        except Exception as exc:  # noqa: BLE001  # preserve native listing fallback for ingress failures
            verbose_logger.exception("Error in list_resource_templates endpoint: %s", exc)
            return ListResourceTemplatesResult(resource_templates=[])

    async def read_resource(ctx: ServerRequestContext, params: ReadResourceRequestParams) -> ReadResourceResult:
        if _mcp_proxy_mode.get():
            _reject_mcp_proxy_operation()
        async with _legacy_operation_context(ctx, trace=False) as context:
            return await operations.GatewayOperations(_capture_host_progress_callback(ctx)).execute(
                ReadResourceRequest(params=params), context
            )

    async def discover(ctx: ServerRequestContext, params: RequestParams) -> DiscoverResult:
        async with _legacy_operation_context(ctx, trace=False) as context:
            return await operations.GatewayOperations().execute(DiscoverRequest(params=params), context)

    server.add_request_handler("server/discover", RequestParams, discover)
    server.add_request_handler("tools/list", PaginatedRequestParams, handle_list_tools)
    server.add_request_handler("tools/call", CallToolRequestParams, mcp_server_tool_call)
    server.add_request_handler("prompts/list", PaginatedRequestParams, list_prompts)
    server.add_request_handler("prompts/get", GetPromptRequestParams, get_prompt)
    server.add_request_handler("resources/list", PaginatedRequestParams, list_resources)
    server.add_request_handler("resources/templates/list", PaginatedRequestParams, list_resource_templates)
    server.add_request_handler("resources/read", ReadResourceRequestParams, read_resource)

    ########################################################
    ############ End of MCP Server Routes ##################
    ########################################################

    ########################################################
    ############ Helper Functions ##########################
    ########################################################

    from litellm.proxy._experimental.mcp_server.operations import (
        _client_has_passthrough_authorization,
        _client_has_per_server_auth_header,
        _get_allowed_mcp_servers,
        _get_allowed_mcp_servers_from_mcp_server_names,
        _get_user_oauth_extra_headers_from_db,
        _http_detail_message,
        _McpDeniedDetail,
        _merge_gateway_initialize_instructions,
        _prefetch_oauth_creds_for_user,
        _prepare_mcp_server_headers,
        _raise_if_initialize_grants_no_mcp_servers,
        _server_answers_to,
        _tool_name_matches,
        apply_tool_overrides,
        filter_tools_by_allowed_tools,
        raise_denied_scoped_mcp_access,
    )

    @contextlib.asynccontextmanager
    async def _gateway_initialize_instructions_request_scope(
        user_api_key_auth: UserAPIKeyAuth | None,
        mcp_servers: list[str] | None,
        client_ip: str | None,
        scoped_server_endpoint: bool = False,
        is_initialize: bool = False,
    ) -> AsyncIterator[None]:
        allowed: Final = await operations._get_allowed_mcp_servers(
            user_api_key_auth=user_api_key_auth,
            mcp_servers=mcp_servers,
            client_ip=client_ip,
        )
        if is_initialize:
            await operations._raise_if_initialize_grants_no_mcp_servers(
                allowed, user_api_key_auth, mcp_servers, client_ip
            )
        if allowed:
            # return_exceptions=True: a per-server probe failure (incl. CancelledError
            # bubbled from anyio task group teardown on connection refused) must not
            # cancel sibling probes or 500 the gateway initialize request.
            await asyncio.gather(
                *[
                    operations.global_mcp_server_manager._ensure_upstream_initialize_instructions_cached(s)
                    for s in allowed
                    if s is not None
                ],
                return_exceptions=True,
            )
        merged: Final = operations._merge_gateway_initialize_instructions(allowed_mcp_servers=allowed)
        scoped_server_name = None
        if scoped_server_endpoint and len(allowed) == 1:
            scoped_server: Final = allowed[0]
            scoped_server_name = (
                scoped_server.alias or scoped_server.server_name or scoped_server.name or scoped_server.server_id
            )
        instructions_token: Final = _mcp_gateway_initialize_instructions.set(merged)
        server_name_token: Final = _mcp_gateway_server_name.set(scoped_server_name)
        try:
            yield
        finally:
            _mcp_gateway_initialize_instructions.reset(instructions_token)
            _mcp_gateway_server_name.reset(server_name_token)

    from litellm.proxy._experimental.mcp_server.operations import (
        _MCP_CREDENTIAL_REQUEST_FIELDS,
        _aggregate_server_key,
        _check_byok_credential,
        _fire_mcp_tool_call_logging,
        _get_byok_credential,
        _get_prompts_from_mcp_servers,
        _get_resource_templates_from_mcp_servers,
        _get_resources_from_mcp_servers,
        _get_standard_logging_mcp_tool_call,
        _get_tools_from_mcp_servers,
        _handle_local_mcp_tool,
        _handle_managed_mcp_tool,
        _list_mcp_prompts,
        _list_mcp_resource_templates,
        _list_mcp_resources,
        _list_mcp_tools,
        _list_tools_before_first_call,
        _resolve_display_name_to_original,
        _run_post_mcp_call_guardrails,
        call_mcp_tool,
        execute_mcp_tool,
        filter_tools_by_key_team_permissions,
        fire_mcp_tool_call_failure_logging,
        mcp_get_prompt,
        mcp_read_resource,
    )

    def _get_mcp_servers_in_path(path: str) -> list[str] | None:
        """
        Get the MCP servers from the path
        """
        import re

        if path.rstrip("/") in ("/mcp/sse", "/mcp/sse/messages"):
            return None
        mcp_servers_from_path: list[str] | None = None
        segments: Final = [s for s in path.split("/") if s]
        if len(segments) >= 2 and segments[1] == "mcp" and segments[0] != "mcp":
            return [segments[0]]

        # Match /mcp/<servers_and_maybe_path>
        # Where servers can be comma-separated list of server names
        # Server names can contain slashes (e.g., "custom_solutions/user_123")
        mcp_path_match: Final = re.match(r"^/mcp/([^?#]+)(?:\?.*)?(?:#.*)?$", path)
        if mcp_path_match:
            servers_and_path: Final = mcp_path_match.group(1)

            if servers_and_path:
                # Check if it contains commas (comma-separated servers)
                if "," in servers_and_path:
                    # For comma-separated, look for a path at the end
                    # Common patterns: /tools, /chat/completions, etc.
                    path_match: Final = re.search(r"/([^/,]+(?:/[^/,]+)*)$", servers_and_path)
                    if path_match:
                        # Path found at the end, remove it from servers
                        path_part: Final = "/" + path_match.group(1)
                        servers_part: Final = servers_and_path[: -len(path_part)]
                        mcp_servers_from_path = [s.strip() for s in servers_part.split(",") if s.strip()]
                    else:
                        # No path, just comma-separated servers
                        mcp_servers_from_path = [s.strip() for s in servers_and_path.split(",") if s.strip()]
                else:
                    # Single server case - use regex approach for server/path separation
                    # This handles cases like "custom_solutions/user_123/chat/completions"
                    # where we want to extract "custom_solutions/user_123" as the server name
                    single_server_match: Final = re.match(r"^([^/]+(?:/[^/]+)?)(?:/.*)?$", servers_and_path)
                    if single_server_match:
                        server_name: Final = single_server_match.group(1)
                        mcp_servers_from_path = [server_name]
                    else:
                        mcp_servers_from_path = [servers_and_path]
        return mcp_servers_from_path

    def _load_mcp_client_allowlist() -> MCPClientAllowlist | None:
        from litellm.proxy.proxy_server import general_settings

        return load_mcp_client_allowlist(general_settings)

    def reject_disallowed_mcp_client(headers: Mapping[str, str], user_api_key_auth: UserAPIKeyAuth | None) -> None:
        """Gate every MCP tool surface on ``mcp_allowed_clients``; the dashboard's own session is not a client app."""
        if user_api_key_auth is not None and is_ui_session_credential(user_api_key_auth):
            return
        rejection: Final = check_mcp_client_allowed(
            allowlist=_load_mcp_client_allowlist(),
            jwt_claims=user_api_key_auth.jwt_claims if user_api_key_auth is not None else None,
            headers=headers,
        )
        if rejection is None:
            return
        verbose_logger.warning("Rejected MCP request from a disallowed client application: %s", rejection.details)
        raise HTTPException(status_code=403, detail=rejection.response_body)

    async def extract_mcp_auth_context(scope, path):
        """
        Extracts mcp_servers from the path and processes the MCP request for auth context.
        Returns: (user_api_key_auth, mcp_auth_header, mcp_servers, mcp_server_auth_headers)
        """
        mcp_servers_from_path: Final = _get_mcp_servers_in_path(path)
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

    def _get_session_id_from_scope(scope: Scope) -> str | None:
        """
        Extract mcp-session-id from ASGI scope headers.
        Returns None if not present.
        """
        scope_headers: Final[Sequence[tuple[bytes | str, bytes | str]]] = scope.get("headers", [])
        for header_name, header_value in scope_headers:
            name = header_name if isinstance(header_name, bytes) else header_name.encode()
            if name.lower() == b"mcp-session-id":
                return header_value.decode() if isinstance(header_value, bytes) else str(header_value)
        return None

    def _owner_fingerprint_for(
        user_api_key_auth: UserAPIKeyAuth | None,
        oauth2_headers: dict[str, str] | None = None,
        client_ip: str | None = None,
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

        def _bytes_for_hash(value: object) -> bytes | None:
            """Only hash str/bytes secrets; skip mocks and other unexpected types."""
            if value is None:
                return None
            if isinstance(value, (bytes, bytearray)):
                return bytes(value)
            if isinstance(value, str):
                return value.encode("utf-8")
            return None

        if user_api_key_auth is not None:
            key_material: Final = _bytes_for_hash(getattr(user_api_key_auth, "api_key", None))
            if key_material:
                api_key_hash: Final = hashlib.sha256(key_material).hexdigest()
                return f"key:{api_key_hash}"
            uid_material: Final = _bytes_for_hash(getattr(user_api_key_auth, "user_id", None))
            if uid_material:
                user_id_hash: Final = hashlib.sha256(uid_material).hexdigest()
                return f"user:{user_id_hash}"
        if oauth2_headers:
            authz: Final = oauth2_headers.get("Authorization") or oauth2_headers.get("authorization")
            authz_bytes: Final = _bytes_for_hash(authz)
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
            data: Final = json.loads(body)
            return isinstance(data, dict) and data.get("method") == "initialize"
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
            return False

    def _extract_initialize_client_info(body: bytes) -> Implementation | None:
        try:
            return InitializeRequest.model_validate_json(body, by_name=False).params.client_info
        except ValidationError:
            return None

    def _group_session_counts(
        sessions: Sequence[MCPGatewaySession],
        label_for: Callable[[MCPGatewaySession], str | None],
    ) -> tuple[MCPGatewaySessionGroupCount, ...]:
        counts: Final = types.MappingProxyType(Counter(label_for(session) for session in sessions))
        return tuple(
            sorted(
                (MCPGatewaySessionGroupCount(label=label, count=count) for label, count in counts.items()),
                key=lambda group: (-group.count, group.label is None, group.label or ""),
            )
        )

    def _gateway_session_for(session_id: str, auth_user: MCPAuthenticatedUser, now: float) -> MCPGatewaySession:
        client_info: Final = _stateful_session_client_info.get(session_id)
        key_auth: Final = auth_user.user_api_key_auth
        return MCPGatewaySession(
            session_id_prefix=session_id[:MCP_GATEWAY_SESSION_ID_PREFIX_LENGTH],
            client_name=client_info.name if client_info is not None else None,
            client_version=client_info.version if client_info is not None else None,
            user_id=key_auth.user_id if key_auth is not None else None,
            user_email=key_auth.user_email if key_auth is not None else None,
            key_alias=key_auth.key_alias if key_auth is not None else None,
            team_id=key_auth.team_id if key_auth is not None else None,
            team_alias=key_auth.team_alias if key_auth is not None else None,
            client_ip=auth_user.client_ip,
            idle_seconds=max(0.0, now - _stateful_session_auth_context_last_seen.get(session_id, now)),
            in_flight_requests=_stateful_session_active_request_counts.get(session_id, 0),
        )

    def get_mcp_gateway_sessions_report(now: float | None = None) -> MCPGatewaySessionsResponse:
        """Live stateful Streamable HTTP sessions held by this worker process.

        Only sessions whose transport is still registered with the stateful
        session manager are reported; SSE and stateless requests hold no
        session and are never counted.
        """
        report_time: Final = time.monotonic() if now is None else now
        live_session_ids: Final = frozenset(_stateful_server_instances())
        sessions: Final = tuple(
            _gateway_session_for(session_id, auth_user, report_time)
            for session_id, auth_user in tuple(_stateful_session_auth_contexts.items())
            if session_id in live_session_ids
        )
        return MCPGatewaySessionsResponse(
            worker_pid=os.getpid(),
            total_sessions=len(sessions),
            by_client=_group_session_counts(sessions, lambda session: session.client_name),
            by_user=_group_session_counts(sessions, lambda session: session.user_id),
            sessions=sessions,
        )

    def _session_matches_admin_selector(
        session_id: str,
        auth_user: MCPAuthenticatedUser,
        session_id_prefix: str | None,
        user_id: str | None,
    ) -> bool:
        if session_id_prefix is not None and not session_id.startswith(session_id_prefix):
            return False
        if user_id is None:
            return True
        key_auth: Final = auth_user.user_api_key_auth
        return key_auth is not None and key_auth.user_id == user_id

    def _forget_expired_admin_terminated_session_ids(now: float) -> None:
        for session_id in [
            session_id
            for session_id, last_replayed in _admin_terminated_session_ids.items()
            if now - last_replayed >= _STATEFUL_SESSION_IDLE_TIMEOUT_SECONDS
        ]:
            del _admin_terminated_session_ids[session_id]

    def _is_admin_terminated_session_id(session_id: str, now: float) -> bool:
        last_replayed: Final = _admin_terminated_session_ids.get(session_id)
        if last_replayed is None:
            return False
        if now - last_replayed >= _STATEFUL_SESSION_IDLE_TIMEOUT_SECONDS:
            del _admin_terminated_session_ids[session_id]
            return False
        _admin_terminated_session_ids[session_id] = now
        return True

    async def terminate_mcp_gateway_sessions(
        *,
        session_id_prefix: str | None = None,
        user_id: str | None = None,
    ) -> MCPGatewaySessionsTerminateResponse:
        """Force-close every live stateful session on this worker matching the selector.

        The transport is terminated (open streams close), all per-session
        tracking is dropped, and the id is remembered so a client that keeps
        sending it receives 404 and has to ``initialize`` again, which re-runs
        admission. Only sessions held by this worker process are affected.
        """
        now: Final = time.monotonic()
        _forget_expired_admin_terminated_session_ids(now)
        server_instances: Final = _stateful_server_instances()
        targets: Final = tuple(
            (session_id, auth_user)
            for session_id, auth_user in tuple(_stateful_session_auth_contexts.items())
            if session_id in server_instances
            and _session_matches_admin_selector(session_id, auth_user, session_id_prefix, user_id)
        )
        terminated: Final = tuple(_gateway_session_for(session_id, auth_user, now) for session_id, auth_user in targets)
        for session_id, _ in targets:
            _admin_terminated_session_ids[session_id] = now
            transport = server_instances.pop(session_id, None)
            _remove_stateful_session_tracking(session_id)
            if transport is not None:
                await transport.terminate()
            verbose_logger.warning("MCP session '%s' terminated by an administrator.", session_id)
        return MCPGatewaySessionsTerminateResponse(
            worker_pid=os.getpid(),
            terminated_sessions=len(terminated),
            sessions=terminated,
        )

    async def _read_request_body_for_routing(
        receive: Receive,
    ) -> tuple[list[Message], bytes]:
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
        consumed_messages: Final[list[Message]] = []
        body_chunks: Final[list[bytes]] = []
        peeked_bytes = 0

        while True:
            message = await receive()
            consumed_messages.append(message)

            if message.get("type") != "http.request":
                break

            body: bytes = message.get("body", b"") or b""
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
        _mcp_session_header: Final = b"mcp-session-id"
        _headers: Final[Sequence[tuple[bytes | str, bytes | str]]] = scope.get("headers", [])

        def _normalize_header_name(header_name: object) -> bytes | None:
            if isinstance(header_name, bytes):
                return header_name.lower()
            if isinstance(header_name, str):
                return header_name.lower().encode("utf-8", errors="replace")
            return None

        _session_id: str | None = None
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
        known_sessions: Final = getattr(mgr, "_server_instances", None)
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
        method: Final = scope.get("method", "").upper()

        if method == "DELETE":
            _remove_stateful_session_tracking(_session_id)
            verbose_logger.info(
                "DELETE request for non-existent MCP session '%s'. Returning success (idempotent DELETE).",
                _session_id,
            )
            success_response: Final = JSONResponse(
                status_code=200,
                content={"message": "Session terminated successfully"},
            )
            await success_response(scope, receive, send)
            return True

        if _is_admin_terminated_session_id(_session_id, time.monotonic()):
            terminated_response: Final = JSONResponse(
                status_code=404,
                content={  # mutable-ok: JSONResponse content must be a plain dict
                    "error": "Not Found",
                    "details": "mcp-session-id was terminated by an administrator. Send initialize to start a new session.",
                },
            )
            await terminated_response(scope, receive, send)
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
        original_op: Final = user_api_key_auth.object_permission
        if original_op is not None and SpecialMCPServerNames.no_mcp_servers.value in (original_op.mcp_servers or []):
            raise HTTPException(
                status_code=403,
                detail="API key is scoped to no MCP servers; toolset access is denied.",
            )

        # Access control: non-admin keys must have this toolset in their grant list.
        # Use _user_has_admin_view so that PROXY_ADMIN_VIEW_ONLY is also treated as admin.
        is_admin: Final = _user_has_admin_view(user_api_key_auth)
        if not is_admin:
            op: Final = user_api_key_auth.object_permission
            granted: Final = getattr(op, "mcp_toolsets", None) if op else None
            # granted=None → key has no explicit toolset grants → deny (same semantics as
            # fetch_mcp_toolsets which returns [] for non-admin keys with no grants configured).
            # granted=[] or list without toolset_id → also deny.
            if granted is None or toolset_id not in granted:
                raise HTTPException(
                    status_code=403,
                    detail=f"API key does not have access to toolset '{toolset_id}'.",
                )

        tool_permissions = await operations.global_mcp_server_manager.resolve_toolset_tool_permissions(
            toolset_ids=[toolset_id]
        )
        server_ids: Final = list(tool_permissions.keys())
        existing_op: Final = user_api_key_auth.object_permission
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
        return user_api_key_auth.model_copy(update={"object_permission": updated_op, "mcp_toolset_id": toolset_id})

    async def _key_granted_single_server(
        server: MCPServer,
        mcp_servers: Sequence[str] | None,
        user_api_key_auth: UserAPIKeyAuth | None,
        client_ip: str | None,
    ) -> bool:
        """Sign-in challenges are issued only on a single-server connect the key's grant admits, so a key
        without access gets the grant's 403 instead of a sign-in it could not use."""
        if len(mcp_servers or []) != 1:
            return False
        allowed: Final = await operations._get_allowed_mcp_servers(
            user_api_key_auth=user_api_key_auth, mcp_servers=mcp_servers, client_ip=client_ip
        )
        return any(granted.server_id == server.server_id for granted in allowed)

    async def _raise_preemptive_401_for_unauthenticated_servers(
        scope: Scope,
        mcp_servers: list[str] | None,
        oauth2_headers: dict[str, str] | None,
        mcp_server_auth_headers: dict[str, dict[str, str]] | None,
        user_api_key_auth: UserAPIKeyAuth | None,
        client_ip: str | None,
        allowed_server_ids: set[str] | None = None,
        raw_headers: Mapping[str, str] | None = None,
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
            server = operations.global_mcp_server_manager.get_mcp_server_answering_to(server_name, client_ip=client_ip)
            if server is not None and allowed_server_ids is not None and server.server_id not in allowed_server_ids:
                # Caller's narrowed scope excludes this server — skip the
                # preemptive challenge and let downstream authorization
                # return 403.
                continue
            if server is not None and server.auth_type == MCPAuth.oauth2 and server.oauth2_flow == "client_credentials":
                # Stamped M2M: the challenge decision below never reads discovered
                # metadata, so deferred-discovery failures must not 503 this loop.
                # Unstamped rows stay on the discover-first path because filling
                # authorization_url/token_url can change their inferred flow.
                continue
            if server is not None:
                server = await operations.global_mcp_server_manager.ensure_oauth_metadata_discovered(server)
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
                    # challenge) runs through it. A keyless admitted subject is
                    # challenged with the per-server resource_metadata (whose
                    # authorization server is the gateway itself, vaulting via the
                    # authorize interlude); the per-server relay advertised below
                    # cannot vault without a litellm key on its token request.
                    if await operations.global_mcp_server_manager.has_user_oauth_token(server, user_api_key_auth):
                        continue

                    if _is_mcp_admitted_user_subject(user_api_key_auth):
                        raise HTTPException(
                            status_code=401,
                            detail="Unauthorized",
                            headers={
                                "www-authenticate": get_passthrough_www_authenticate(
                                    scope=scope,
                                    server_name=server_name,
                                )
                            },
                        )

                    request = StarletteRequest(scope)
                    base_url = get_request_base_url(request)
                    _path = get_route_relative_request_path(scope)

                    # Pick the well-known AS-metadata form that matches the inbound route
                    # so strict RFC 9728 §3.2 clients can resolve it correctly.
                    as_metadata_root = f"{base_url}/.well-known/oauth-authorization-server{well_known_root_suffix()}"
                    if _path.startswith(f"/mcp/{server_name}"):
                        _as_url = f"{as_metadata_root}/mcp/{server_name}"
                    else:
                        _as_url = f"{as_metadata_root}/{server_name}"
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
                    www_authenticate = get_passthrough_www_authenticate(
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

            # Caller sign-in: challenge at connect because a tool-call-time 401 is wrapped into a
            # JSON-RPC error and the WWW-Authenticate header is lost. Non-OBO gates fire only on a
            # single-server connect the key's grant admits.
            sign_in = caller_sign_in_for(server, user_api_key_auth) if server is not None else None
            if (
                server
                and sign_in is not None
                and operations.global_mcp_server_manager._extract_subject_token(  # pyright: ignore[reportPrivateUsage]  # the manager owns the subject/admission filter shared with the preflight
                    oauth2_headers, raw_headers, user_api_key_auth
                )
                is None
                and (
                    server.auth_type == MCPAuth.oauth2_token_exchange
                    or await _key_granted_single_server(server, mcp_servers, user_api_key_auth, client_ip)
                )
            ):
                from litellm.proxy._experimental.mcp_server.outbound_credentials.adapter import (  # noqa: PLC0415  # lazy: adapter pulls MCP subgraph
                    raise_token_exchange_challenge,
                )
                from litellm.proxy.middleware.per_request_root_path_middleware import (  # noqa: PLC0415  # lazy: middleware imports proxy utils
                    get_request_root_path,
                )

                raise_token_exchange_challenge(server, root_path=get_request_root_path())

            # Exchange-backed modes (token_exchange's OBO mint, id_jag's stored-assertion mint): run
            # the exchange here at the transport edge, so a rejected subject raises the RFC 9728
            # challenge and any other failure its public status, instead of the session opening and
            # list_tools masking it as an empty tool list. The manager owns which modes pre-flight
            # and what each mints from. Gated to single-server routes the key may reach; the
            # multi-server aggregate keeps absorbing per-server auth failures so one bad server
            # cannot 401 the whole connect.
            if (
                server
                and len(mcp_servers or []) == 1
                and server.server_id
                in frozenset(
                    allowed.server_id
                    for allowed in await operations._get_allowed_mcp_servers(
                        user_api_key_auth=user_api_key_auth, mcp_servers=mcp_servers, client_ip=client_ip
                    )
                )
            ):
                await operations.global_mcp_server_manager.preflight_token_exchange(
                    server=server,
                    oauth2_headers=oauth2_headers,
                    user_api_key_auth=user_api_key_auth,
                    raw_headers=raw_headers,
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
                and not operations._client_has_passthrough_authorization(
                    server, oauth2_headers, mcp_server_auth_headers
                )
            ):
                www_authenticate = get_passthrough_www_authenticate(
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
                and not operations._client_has_per_server_auth_header(server, mcp_server_auth_headers)
            ):
                www_authenticate = get_passthrough_www_authenticate(
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
                and not operations._client_has_per_server_auth_header(server, mcp_server_auth_headers)
            ):
                if server.is_dcr_bridge:
                    raise HTTPException(
                        status_code=401,
                        detail="Unauthorized",
                        headers={
                            "www-authenticate": get_passthrough_www_authenticate(
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

    def _get_authorization_header_from_scope(scope: Scope) -> str | None:
        """First ``Authorization`` header value in the ASGI scope, or None."""
        scope_headers: Final[Sequence[tuple[bytes, bytes]]] = scope.get("headers", [])
        for key, value in scope_headers:
            if key.lower() == b"authorization":
                return value.decode("latin-1")
        return None

    def _scope_has_authorization_header(scope: Scope) -> bool:
        return _get_authorization_header_from_scope(scope) is not None

    def _get_forwarded_auth_from_scope(scope: Scope) -> str | None:
        """Return the upstream-bound ``Authorization`` header value, or None.

        Only returns the ``Authorization`` header when ``x-litellm-api-key`` is
        also present. In that case ``Authorization`` is unambiguously the
        upstream token the caller wants forwarded to the MCP server. When
        ``x-litellm-api-key`` is absent the ``Authorization`` header may itself
        be the LiteLLM proxy API key (backward-compat path in
        ``MCPRequestHandler.process_mcp_request``), and forwarding it upstream
        would leak the proxy key to a third-party MCP server.
        """
        scope_headers: Final[Sequence[tuple[bytes, bytes]]] = scope.get("headers", [])
        has_litellm_key_header: Final = any(key.lower() == b"x-litellm-api-key" for key, _ in scope_headers)
        if not has_litellm_key_header:
            return None
        return _get_authorization_header_from_scope(scope)

    async def _probe_upstream_auth(
        url: str,
        auth_header: str,
        timeout: float = 5.0,
    ) -> tuple[int, str | None]:
        """JSON-RPC initialize-probe the upstream URL to check whether the token is accepted.

        Uses POST so StreamableHTTP MCP servers run the same auth path as a
        real client request. Returns (status_code, www_authenticate).
        Fails-open with (200, None) on network errors so a transient hiccup
        does not block valid requests.

        Uses the public ``AsyncHTTPHandler.post()`` interface and catches
        ``httpx.HTTPStatusError`` separately so the 401/403 we want to surface
        is not swallowed by the broad fail-open ``except Exception`` below.
        """
        client: Final = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.MCP,
            params={"timeout": timeout},
        )
        probe_payload: Final = {
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
        probe_headers: Final = {
            "Accept": "application/json, text/event-stream",
            **({"Authorization": auth_header} if auth_header else {}),
        }
        try:
            resp: Final = await client.post(
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
            verbose_logger.debug("_probe_upstream_auth: probe to %s failed (%s), allowing request through", url, exc)
            return 200, None

    async def _check_passthrough_upstream_auth(
        scope: Scope,
        user_api_key_auth: UserAPIKeyAuth | None,
        mcp_servers: list[str] | None,
        client_ip: str | None,
    ) -> None:
        """Probe pass-through upstream servers in parallel before the MCP session starts.

        Only servers the caller's key is already authorized to reach are probed —
        the list is derived from _get_allowed_mcp_servers so that a user cannot
        trigger an upstream probe against a server their key is not permitted for.

        The MCP SDK commits HTTP 200 headers before invoking handlers, so a 401
        can only be returned before that point. This function raises HTTPException(401)
        with a WWW-Authenticate header if any upstream rejects the client token, or 403
        if the upstream accepts it but forbids the caller.
        Fails-open: network errors are logged and the request is allowed through.

        """
        forwarded_auth: Final = _get_forwarded_auth_from_scope(scope)
        if not forwarded_auth:
            return

        # Use the authorized server set, not the raw user-supplied names, so that
        # a caller cannot force a probe to a server their key is not allowed to use.
        allowed_servers: Final = await operations._get_allowed_mcp_servers(
            user_api_key_auth=user_api_key_auth,
            mcp_servers=mcp_servers,
            client_ip=client_ip,
        )
        passthrough_targets: Final[tuple[tuple[MCPServer, str, str], ...]] = tuple(
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
        probe_targets: Final = passthrough_targets
        if not probe_targets:
            return

        probe_results: Final = await asyncio.gather(
            *[_probe_upstream_auth(srv.url or "", auth_header) for srv, auth_header, _ in probe_targets]
        )
        for (srv, _, challenge_server_name), (probe_status, _) in zip(probe_targets, probe_results):
            if probe_status == 401:
                # Token is missing or expired: keep pass-through clients on the
                # protected-resource discovery flow so they re-authorize against
                # the upstream IdP metadata proxied by LiteLLM.
                www_authenticate = get_passthrough_www_authenticate(
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
            reject_disallowed_mcp_origin(StarletteRequest(scope))
            bad_version: Final = unsupported_protocol_version(scope)
            if bad_version is not None:
                supported: Final = ", ".join(configured_versions())
                await JSONResponse(
                    status_code=400,
                    content={  # mutable-ok: JSON-RPC error payload
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": INVALID_REQUEST,
                            "message": f"Unsupported MCP-Protocol-Version {bad_version}; supported: {supported}",
                        },
                    },
                )(scope, receive, send)
                return
            path: Final[str] = scope.get("path", "")
            (
                user_api_key_auth,
                mcp_auth_header,
                mcp_servers,
                mcp_server_auth_headers,
                oauth2_headers,
                raw_headers,
            ) = await extract_mcp_auth_context(scope, path)
            reject_disallowed_mcp_client(StarletteRequest(scope).headers, user_api_key_auth)
            scoped_server_endpoint: Final = len(_get_mcp_servers_in_path(path) or []) == 1

            # Extract client IP for MCP access control
            _client_ip: Final = IPAddressUtils.get_mcp_client_ip(StarletteRequest(scope))

            verbose_logger.debug("MCP request mcp_servers (header/path): %s", mcp_servers)
            verbose_logger.debug(
                "MCP server auth headers: %s", list(mcp_server_auth_headers.keys()) if mcp_server_auth_headers else None
            )

            # Strip any client-supplied x-mcp-toolset-id to prevent forgery.
            scope_headers: Final[Sequence[tuple[bytes, bytes]]] = scope.get("headers", [])
            scope["headers"] = [(k, v) for k, v in scope_headers if k.lower() != b"x-mcp-toolset-id"]

            # Apply toolset scope if set server-side via ContextVar (set by
            # /toolset/{name}/mcp and /{name}/mcp route handlers in proxy_server.py).
            active_toolset_id: Final = _mcp_active_toolset_id.get()
            toolset_allowed_server_ids: set[str] | None = None
            if active_toolset_id and user_api_key_auth is not None:
                user_api_key_auth = await _apply_toolset_scope(user_api_key_auth, active_toolset_id)
                op: Final = user_api_key_auth.object_permission
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
                raw_headers=raw_headers,
            )

            # Pre-flight auth check for pass-through servers.  Must run after
            # toolset scoping so the probe list is derived from the fully-authorized
            # server set, not the raw user-supplied names.
            await _check_passthrough_upstream_auth(scope, user_api_key_auth, mcp_servers, _client_ip)

            # Inject masked debug headers when client sends x-litellm-mcp-debug: true
            _debug_headers: Final = MCPDebug.maybe_build_debug_headers(
                raw_headers=raw_headers,
                scope=dict(scope),
                mcp_servers=mcp_servers,
                oauth2_headers=oauth2_headers,
                client_ip=_client_ip,
            )
            diagnostics: Final = MCPAuthDiagnostics() if _debug_headers else None
            if diagnostics is not None:
                scope[MCP_AUTH_DIAGNOSTICS_SCOPE_KEY] = diagnostics
                send = MCPDebug.wrap_send_with_debug_headers(
                    send, _debug_headers, diagnostics.headers, request_method=scope.get("method")
                )

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
            consumed_messages: list[Message] = []

            # Owner-binding: a live stateful session may only be driven by the
            # caller that created it. Reject mismatches with 403 so a leaked
            # mcp-session-id cannot be hijacked by another authenticated user.
            #
            # Run before ``_handle_stale_mcp_session`` so a non-owner cannot
            # force-clean another caller's residual tracking entries via a
            # stale DELETE, and before peeking the request body so the 403
            # response sees a pristine ``receive`` channel.
            if session_id:
                expected_owner: Final = _stateful_session_owners.get(session_id)
                request_owner = _owner_fingerprint_for(user_api_key_auth, oauth2_headers, _client_ip)
                if expected_owner is not None and expected_owner != request_owner:
                    verbose_logger.warning(
                        "Rejecting MCP request: session '%s' owner mismatch.",
                        session_id,
                    )
                    forbidden_response: Final = JSONResponse(
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
                handled: Final = await _handle_stale_mcp_session(scope, receive, send, session_manager_stateful)
                if handled:
                    # Request was fully handled (e.g., DELETE on non-existent session)
                    return
                session_id = _get_session_id_from_scope(scope)

            body = b""
            if scope.get("method") == "POST":
                consumed_messages, body = await _read_request_body_for_routing(receive)
                is_initialize = _is_initialize_request(body)

            use_stateful: Final = bool(session_id or is_initialize)
            target_manager: Final = session_manager_stateful if use_stateful else session_manager_stateless

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
                    too_many_response: Final = JSONResponse(
                        status_code=429,
                        content={
                            "error": "Too Many Requests",
                            "details": "Too many active MCP sessions for this caller.",
                        },
                    )
                    await too_many_response(scope, receive, send)
                    return

            # Replay body messages if we consumed them for peeking
            original_receive: Final = receive
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
            request_method: Final = (scope.get("method") or "").upper()
            if body and request_method == "POST":
                try:
                    _peeked: Final = json.loads(body)
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
                except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
                    # Peek cap truncated the body, so it can't be fully parsed.
                    # Scan the top-level keys (depth-aware) instead of a flat
                    # substring search: a response's result payload may nest a
                    # "method" field, and misreading that would acquire the lock
                    # and deadlock the in-flight tool call awaiting this
                    # response. A false skip is harmless; a false acquire is not.
                    _body_str: Final = body.decode("utf-8", errors="replace")
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

            session_lock: asyncio.Lock | None = None
            if use_stateful and session_id and request_method in ("POST", "DELETE") and not is_jsonrpc_response:
                session_lock = _stateful_session_locks.setdefault(session_id, asyncio.Lock())

            active_request_session_ids: Final[list[str]] = []

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
                _otel_publish_transport_span_on_scope(scope)
                _otel_publish_request_destinations_on_scope(scope)
                auth_user: Final = _set_or_update_auth_context(
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
                        client_info=_extract_initialize_client_info(body),
                    )

                async with _gateway_initialize_instructions_request_scope(
                    user_api_key_auth,
                    mcp_servers,
                    _client_ip,
                    scoped_server_endpoint=scoped_server_endpoint,
                    is_initialize=is_initialize,
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
            verbose_logger.exception("Error handling MCP request: %s", e)
            # Try to send a graceful error response for non-HTTP exceptions
            try:
                from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR

                error_response: Final = JSONResponse(
                    status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                    content={"error": "MCP request failed", "details": str(e)},
                )
                await error_response(scope, receive, send)
            except Exception as response_error:
                verbose_logger.exception("Failed to send error response: %s", response_error)
                # If we can't send a proper response, re-raise the original error
                raise e

    async def handle_sse_mcp(scope: Scope, receive: Receive, send: Send) -> None:
        """Handle MCP requests through SSE."""
        try:
            reject_disallowed_mcp_origin(StarletteRequest(scope))
            bad_version: Final = unsupported_protocol_version(scope)
            if bad_version is not None:
                supported: Final = ", ".join(configured_versions())
                await JSONResponse(
                    status_code=400,
                    content={  # mutable-ok: JSON-RPC error payload
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": INVALID_REQUEST,
                            "message": f"Unsupported MCP-Protocol-Version {bad_version}; supported: {supported}",
                        },
                    },
                )(scope, receive, send)
                return
            from litellm.proxy.auth.auth_utils import get_request_route

            path: Final = get_request_route(StarletteRequest(scope))
            (
                user_api_key_auth,
                mcp_auth_header,
                mcp_servers,
                mcp_server_auth_headers,
                oauth2_headers,
                raw_headers,
            ) = await extract_mcp_auth_context(scope, path)
            reject_disallowed_mcp_client(StarletteRequest(scope).headers, user_api_key_auth)
            scoped_server_endpoint: Final = len(_get_mcp_servers_in_path(path) or []) == 1

            # Extract client IP for MCP access control
            _sse_client_ip: Final = IPAddressUtils.get_mcp_client_ip(StarletteRequest(scope))

            verbose_logger.debug("MCP request mcp_servers (header/path): %s", mcp_servers)
            verbose_logger.debug(
                "MCP server auth headers: %s", list(mcp_server_auth_headers.keys()) if mcp_server_auth_headers else None
            )

            # Strip any client-supplied x-mcp-toolset-id to prevent forgery.
            scope_headers: Final[Sequence[tuple[bytes, bytes]]] = scope.get("headers", [])
            scope["headers"] = [(k, v) for k, v in scope_headers if k.lower() != b"x-mcp-toolset-id"]

            # Apply toolset scope if set server-side via ContextVar so the
            # downstream probe list matches the fully-authorized server set
            # (mirrors the streamable HTTP handler).
            active_toolset_id: Final = _mcp_active_toolset_id.get()
            toolset_allowed_server_ids: set[str] | None = None
            if active_toolset_id and user_api_key_auth is not None:
                user_api_key_auth = await _apply_toolset_scope(user_api_key_auth, active_toolset_id)
                op: Final = user_api_key_auth.object_permission
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
                raw_headers=raw_headers,
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

            owner: Final = _owner_fingerprint_for(user_api_key_auth, oauth2_headers, _sse_client_ip)
            transport_scope: Final[Scope] = {
                **scope,
                "user": AuthenticatedUser(AccessToken(token=owner, client_id=owner, scopes=[])),
            }
            if scope["method"] == "POST":
                await sse.handle_post_message(transport_scope, receive, send)
                return

            async with _gateway_initialize_instructions_request_scope(
                user_api_key_auth,
                mcp_servers,
                _sse_client_ip,
                scoped_server_endpoint=scoped_server_endpoint,
                is_initialize=scope.get("method") == "GET",
            ):
                async with (
                    sse.connect_sse(transport_scope, receive, send) as (read_stream, write_stream),
                    server.lifespan(server) as lifespan_state,
                ):
                    await serve_loop(
                        server,
                        read_stream,
                        write_stream,
                        lifespan_state=lifespan_state,
                        init_options=server.create_initialization_options(),
                    )
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
            verbose_logger.exception("Error handling MCP request: %s", e)
            # Try to send a graceful error response for non-HTTP exceptions
            try:
                # Send a proper HTTP error response instead of letting the exception bubble up
                from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR

                error_response: Final = JSONResponse(
                    status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                    content={"error": "MCP request failed", "details": str(e)},
                )
                await error_response(scope, receive, send)
            except Exception as response_error:
                verbose_logger.exception("Failed to send error response: %s", response_error)
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
    def get_mcp_server_enabled() -> dict[str, bool]:
        """
        Returns if the MCP server is enabled
        """
        return {"enabled": MCP_AVAILABLE}

    class _LegacySseEndpoint:
        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            await handle_sse_mcp(scope, receive, send)

    for sse_path, sse_method in (
        ("/sse", "GET"),
        ("/sse/", "GET"),
        ("/sse/messages", "POST"),
        ("/sse/messages/", "POST"),
    ):
        app.router.routes.append(Route(sse_path, endpoint=_LegacySseEndpoint(), methods=[sse_method]))

    # Mount the MCP handlers
    app.mount("/", handle_streamable_http_mcp)
    app.mount("/mcp", handle_streamable_http_mcp)
    app.mount("/{mcp_server_name}/mcp", handle_streamable_http_mcp)
    app.add_middleware(AuthContextMiddleware)

    ########################################################
    ############ Auth Context Functions ####################
    ########################################################

    def _update_auth_context(
        auth_user: MCPAuthenticatedUser,
        user_api_key_auth: UserAPIKeyAuth | None,
        mcp_auth_header: str | None = None,
        mcp_servers: list[str] | None = None,
        mcp_server_auth_headers: dict[str, dict[str, str]] | None = None,
        oauth2_headers: dict[str, str] | None = None,
        raw_headers: dict[str, str] | None = None,
        client_ip: str | None = None,
    ) -> None:
        auth_user.user_api_key_auth = user_api_key_auth
        auth_user.mcp_auth_header = mcp_auth_header
        auth_user.mcp_servers = mcp_servers
        auth_user.mcp_server_auth_headers = mcp_server_auth_headers or {}
        auth_user.oauth2_headers = oauth2_headers
        auth_user.raw_headers = raw_headers
        auth_user.client_ip = client_ip

    def set_auth_context(
        user_api_key_auth: UserAPIKeyAuth | None,
        mcp_auth_header: str | None = None,
        mcp_servers: list[str] | None = None,
        mcp_server_auth_headers: dict[str, dict[str, str]] | None = None,
        oauth2_headers: dict[str, str] | None = None,
        raw_headers: dict[str, str] | None = None,
        client_ip: str | None = None,
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
        auth_user: Final = MCPAuthenticatedUser(
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
        user_api_key_auth: UserAPIKeyAuth | None,
        mcp_auth_header: str | None = None,
        mcp_servers: list[str] | None = None,
        mcp_server_auth_headers: dict[str, dict[str, str]] | None = None,
        oauth2_headers: dict[str, str] | None = None,
        raw_headers: dict[str, str] | None = None,
        client_ip: str | None = None,
        session_id: str | None = None,
        touch_last_seen: bool = True,
        copy_existing_session_auth_context: bool = False,
    ) -> MCPAuthenticatedUser:
        auth_user: Final = _stateful_session_auth_contexts.get(session_id) if session_id else None
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
        on_session_registered: Callable[[str], None] | None = None,
        client_info: Implementation | None = None,
    ) -> Send:
        async def wrapped_send(message: Message) -> None:
            if message.get("type") == "http.response.start":
                response_headers: Final[Sequence[tuple[bytes | str, bytes | str]]] = message.get("headers", [])
                for key, value in response_headers:
                    header_name = key if isinstance(key, bytes) else str(key).encode()
                    if header_name.lower() == b"mcp-session-id":
                        session_id = value.decode() if isinstance(value, bytes) else str(value)
                        if on_session_registered is not None:
                            on_session_registered(session_id)
                        auth_context_var.set(auth_user)
                        _stateful_session_auth_contexts[session_id] = auth_user
                        _stateful_session_auth_context_last_seen[session_id] = time.monotonic()
                        _stateful_session_owners[session_id] = owner_fingerprint
                        if client_info is not None:
                            _stateful_session_client_info[session_id] = client_info
                        break
            await send(message)

        return wrapped_send

    def get_auth_context() -> tuple[
        UserAPIKeyAuth | None,
        str | None,
        list[str] | None,
        dict[str, dict[str, str]] | None,
        dict[str, str] | None,
        dict[str, str] | None,
        str | None,
    ]:
        """
        Get the UserAPIKeyAuth from the auth context variable.

        Returns:
            Tuple containing: UserAPIKeyAuth, MCP auth header (deprecated),
            MCP servers, server-specific auth headers, OAuth2 headers, raw headers, client IP
        """
        auth_user: Final = auth_context_var.get()
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
        ctx: Final = get_active_mcp_request_ctx()
        return ctx.session if ctx is not None else None

    def _cache_auth_context_lazily():
        session: Final = _get_current_session()
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

        auth: Final = auth_context_var.get()
        if auth and isinstance(auth, MCPAuthenticatedUser):
            try:
                _session_obj_auth_storage[session] = auth
            except TypeError:
                verbose_logger.debug(
                    "_cache_auth_context_lazily: could not store auth via "
                    "session identity — session object is unhashable"
                )

    def _recover_auth_from_session() -> MCPAuthenticatedUser | None:
        session: Final = _get_current_session()
        if session is None:
            return None

        stored: MCPAuthenticatedUser | None = None
        try:
            stored = _session_obj_auth_storage.get(session)
        except TypeError:
            verbose_logger.debug(
                "_recover_auth_from_session: session object is unhashable "
                "(type=%s), skipping _session_obj_auth_storage lookup",
                type(session).__name__,
            )

        return stored

    async def get_or_extract_auth_context() -> tuple[
        UserAPIKeyAuth | None,
        str | None,
        list[str] | None,
        dict[str, dict[str, str]] | None,
        dict[str, str] | None,
        dict[str, str] | None,
        str | None,
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
            stored: Final = _recover_auth_from_session()

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

    def get_active_mcp_session() -> _McpServerSession | None:
        """Return the active MCP session captured during handler execution."""
        session: Final = active_mcp_session_var.get()
        if session is not None:
            return session
        return _get_current_session()

    def get_active_auth_context() -> MCPAuthenticatedUser | None:
        """Return auth context from ContextVar or session storage."""
        auth: Final = auth_context_var.get()
        if auth and isinstance(auth, MCPAuthenticatedUser):
            return auth

        stored: Final = _recover_auth_from_session()
        if stored is not None:
            return stored
        return None

    ########################################################
    ############ End of Auth Context Functions #############
    ########################################################

else:
    app = FastAPI()
