"""
Counts HTTP requests to LLM inference, MCP, and A2A endpoints.

Feeds two independent sinks off one classification:

- ``GatewayRequestSink`` receives every classified request with its status and
  is the source of truth for SGR (successful gateway requests) on the admin UI.
  Not license-gated (see litellm.proxy.db.gateway_request_tracking). It is not
  told which deployment served the request: it persists its counts, so every
  dimension it takes has to be one the proxy chooses.
- ``BillingRecorder`` receives 2xx requests only and exports them for
  enterprise metering (see litellm.proxy.enterprise_billing.billing_metrics).

Both are injected. When neither is present the middleware is a transparent
pass-through.
"""

import re
import threading
from collections.abc import Callable, Sequence
from enum import Enum
from typing import Final, Protocol, runtime_checkable

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import LiteLLMRoutes


class BillableCategory(str, Enum):
    LLM = "llm"
    MCP = "mcp"
    A2A = "a2a"


@runtime_checkable
class BillingRecorder(Protocol):
    def record(self, *, category: BillableCategory, route: str, status_code: int, model_id: str | None) -> None: ...


@runtime_checkable
class GatewayRequestSink(Protocol):
    """
    Records every classified request, 2xx or not, for the SGR dashboard.

    Distinct from BillingRecorder on three counts: this is not license-gated,
    it is not restricted to 2xx, and it takes no model id. The deployment that
    served a request is deliberately not part of what it records, because the
    dashboard aggregates by route and a per-deployment dimension would only
    multiply the rows it has to sum back together.
    """

    def record(self, *, category: BillableCategory, route: str, status_code: int) -> None: ...


_MODEL_ID_HEADER: Final = b"x-litellm-model-id"

# Ordered: a longer suffix that shares an ending with a shorter one must come
# first, e.g. "/chat/completions" before "/completions". This is the POST
# inference surface that writes a SpendLogs row on success, so the exported
# count lines up with the admin UI usage page for inference traffic. Billing
# is a deliberate lower bound on SpendLogs rows: management writes that also
# log (batch/file/fine-tuning creation, interaction cancel) and non-POST calls
# that log (passthrough reads) never bill, so drift only ever undercounts.
_LLM_ROUTE_SUFFIXES: Final[tuple[str, ...]] = (
    "/chat/completions",
    "/completions",
    "/embeddings",
    "/responses",
    "/rerank",
    "/moderations",
    "/images/generations",
    "/images/edits",
    "/images/variations",
    "/audio/transcriptions",
    "/audio/translations",
    "/audio/speech",
    "/videos",  # create; GET list is excluded by the POST gate
    "/remix",  # /v1/videos/{id}/remix
    "/ocr",
    "/search",  # /v1/search and /v1/vector_stores/{id}/search
    "/rag/query",
    "/rag/ingest",
    ":generateContent",  # Gemini-native /v1beta/models/{model}:generateContent
    ":streamGenerateContent",
)

# Exact paths only: a suffix match would also catch non-inference resources that
# share the ending, e.g. the OpenAI Assistants route /v1/threads/{id}/messages
# writes no SpendLogs row and must not bill, unlike Anthropic /v1/messages.
_LLM_ROUTE_EXACT: Final[tuple[str, ...]] = (
    "/v1/messages",
    "/interactions",  # Google Interactions create; /{id} reads and /cancel do not match
    "/v1beta/interactions",
    "/comprehendmedical",  # AWS-SDK-shaped passthrough: the operation rides in the X-Amz-Target header
)

# Provider passthrough prefixes (e.g. /bedrock/..., /vertex-ai/...) carry real
# inference calls that write SpendLogs rows, so they bill. Anchored to the
# routes enum so new providers are picked up without touching this module.
# /langfuse forwards observability traffic, not inference: it writes no
# SpendLogs row and must not bill.
_NON_BILLABLE_PASSTHROUGH_PREFIXES: Final = frozenset({"/langfuse"})
_PASSTHROUGH_PREFIXES: Final[tuple[str, ...]] = tuple(
    prefix
    for prefix in LiteLLMRoutes.mapped_pass_through_routes.value
    if prefix not in _NON_BILLABLE_PASSTHROUGH_PREFIXES
)


def _classify_llm_route(path: str) -> str | None:
    exact_match: Final = next((route for route in _LLM_ROUTE_EXACT if path == route), None)
    if exact_match is not None:
        return exact_match
    suffix_match = next((suffix for suffix in _LLM_ROUTE_SUFFIXES if path == suffix or path.endswith(suffix)), None)
    if suffix_match is not None:
        return suffix_match
    # Deep passthrough paths only: the bare prefix itself is not an inference call.
    return next((prefix for prefix in _PASSTHROUGH_PREFIXES if path.startswith(f"{prefix}/")), None)


_MCP_MANAGEMENT_PREFIX: Final = "/v1/mcp"
_MCP_DYNAMIC_TRANSPORT: Final = re.compile(r"/(?:toolset/)?[^/]+/mcp")
# The REST wrapper's tool-call endpoint executes a tool and fires the same MCP
# spend logging as the /mcp transport; its list/test siblings do not bill.
_MCP_REST_TOOL_CALL: Final = "/mcp-rest/tools/call"

_A2A_INVOKE_SUFFIX: Final = "/message/send"
_A2A_TRANSPORT_PREFIXES: Final[tuple[str, ...]] = ("/v1/a2a/", "/a2a/")
# Bare POST /a2a/{agent_id} carries the JSON-RPC method in the body, not the
# path. Only message/send and message/stream write a SpendLogs row there; the
# task RPCs (tasks/get, tasks/cancel, tasks/pushNotificationConfig/*, ...) are
# forwarded upstream and write none. A path-only classifier cannot separate
# them, so the bare route does not bill: counting a task RPC would overcount,
# while missing a bare-path message/send only undercounts, and undercounting is
# the sole direction this metric is allowed to drift. The /mcp transport is
# method-agnostic by contrast because its list path logs a SpendLogs row too.


def _classify_mcp_route(path: str) -> str | None:
    if path == _MCP_MANAGEMENT_PREFIX or path.startswith(f"{_MCP_MANAGEMENT_PREFIX}/"):
        return None
    if path == "/mcp" or path.startswith("/mcp/"):
        return "/mcp"
    if path == _MCP_REST_TOOL_CALL:
        return "/mcp"
    if _MCP_DYNAMIC_TRANSPORT.fullmatch(path) is not None:
        return "/mcp"
    return None


def _classify_a2a_route(path: str) -> str | None:
    if path.endswith(_A2A_INVOKE_SUFFIX) and any(path.startswith(prefix) for prefix in _A2A_TRANSPORT_PREFIXES):
        return "/a2a"
    return None


def classify_billable_request(path: str, method: str = "POST") -> tuple[BillableCategory, str] | None:
    """Map a request path to its (category, normalized route), or None if not billable."""
    normalized: Final = path.rstrip("/") or "/"

    mcp_route: Final = _classify_mcp_route(normalized)
    if mcp_route is not None:
        return (BillableCategory.MCP, mcp_route)

    a2a_route: Final = _classify_a2a_route(normalized)
    if a2a_route is not None:
        return (BillableCategory.A2A, a2a_route)

    # POST-only is a conservative gate: non-POST calls can still write a
    # SpendLogs row (passthrough reads, resource GETs) but must not bill, so
    # any classifier-vs-dashboard mismatch is an undercount, never an overcount.
    if method.upper() != "POST":
        return None

    llm_route: Final = _classify_llm_route(normalized)
    if llm_route is not None:
        return (BillableCategory.LLM, llm_route)
    return None


def _extract_model_id(headers: Sequence[tuple[bytes, bytes]]) -> str | None:
    return next(
        (value.decode("latin-1") for name, value in headers if name.lower() == _MODEL_ID_HEADER and value),
        None,
    )


class BillableRequestMetricsMiddleware:
    """
    Pure ASGI middleware that classifies each request once and fans the result
    out to the SGR sink (any status) and the billing recorder (2xx only).
    Modeled on InFlightRequestsMiddleware: it wraps `send`, reads the final
    status and the x-litellm-model-id header off the `http.response.start`
    message, and never blocks or fails the request path.
    """

    def __init__(
        self,
        app: ASGIApp,
        recorder: BillingRecorder | None = None,
        recorder_factory: Callable[[], BillingRecorder | None] | None = None,
        sink: GatewayRequestSink | None = None,
        sink_factory: Callable[[], GatewayRequestSink | None] | None = None,
    ) -> None:
        self.app = app
        self.recorder = recorder
        # The factory defers recorder construction to the first request, AFTER the
        # startup event has loaded the YAML config's environment_variables (license
        # and cert env vars). Building at import time captured recorder=None for
        # deployments configured that way. Resolved exactly once; the result
        # (including None) is cached.
        self._recorder_factory = recorder_factory
        self._resolved = recorder_factory is None
        self._resolve_lock = threading.Lock()
        # Resolved on the same schedule and for the same reason: the DB is not
        # connected at import time, so the sink cannot be built there either.
        self.sink = sink
        self._sink_factory = sink_factory
        self._sink_resolved = sink_factory is None
        self._sink_resolve_lock = threading.Lock()

    def _resolve_recorder(self) -> BillingRecorder | None:
        if self._resolved:
            return self.recorder
        # The lock keeps concurrent first requests from each building their own
        # MeterProvider (and leaking its background exporter thread).
        with self._resolve_lock:
            if not self._resolved:
                factory: Final = self._recorder_factory
                self.recorder = factory() if factory is not None else self.recorder
                self._resolved = True
        return self.recorder

    def _resolve_sink(self) -> GatewayRequestSink | None:
        if self._sink_resolved:
            return self.sink
        with self._sink_resolve_lock:
            if not self._sink_resolved:
                factory: Final = self._sink_factory
                self.sink = factory() if factory is not None else self.sink
                self._sink_resolved = True
        return self.sink

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        recorder: Final = self._resolve_recorder()
        sink: Final = self._resolve_sink()
        if recorder is None and sink is None:
            await self.app(scope, receive, send)
            return

        classification: Final = classify_billable_request(scope.get("path", ""), scope.get("method", "POST"))
        if classification is None:
            await self.app(scope, receive, send)
            return

        category, route = classification
        status_code = 0
        model_id: str | None = None

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code, model_id
            if message["type"] == "http.response.start":
                status_code = message["status"]
                model_id = _extract_model_id(message.get("headers", []))
            await send(message)

        await self.app(scope, receive, send_wrapper)

        if sink is not None:
            try:
                sink.record(category=category, route=route, status_code=status_code)
            except Exception:  # noqa: BLE001 -- metering must never fail a request that was already served
                verbose_proxy_logger.warning("gateway request metering failed for %s", route, exc_info=True)

        if recorder is not None and 200 <= status_code < 300:
            try:
                recorder.record(category=category, route=route, status_code=status_code, model_id=model_id)
            except Exception:  # noqa: BLE001 -- metering must never fail a request that was already served
                verbose_proxy_logger.warning("billable request metering failed for %s", route, exc_info=True)
