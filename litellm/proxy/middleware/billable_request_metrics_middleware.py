"""
Counts billable HTTP requests on enterprise deployments.

A billable request is an inbound request to an LLM inference, MCP, or A2A
endpoint that returns a 2xx status. The actual export happens in an injected
recorder (see litellm.proxy.enterprise_billing.billing_metrics); when no
recorder is injected (non-enterprise, or metering misconfigured) this
middleware is a transparent pass-through.
"""

import re
from enum import Enum
from typing import Callable, Optional, Protocol, Sequence, runtime_checkable

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from litellm.proxy._types import LiteLLMRoutes


class BillableCategory(str, Enum):
    LLM = "llm"
    MCP = "mcp"
    A2A = "a2a"


@runtime_checkable
class BillingRecorder(Protocol):
    def record(self, *, category: BillableCategory, route: str, status_code: int, model_id: Optional[str]) -> None: ...


_MODEL_ID_HEADER = b"x-litellm-model-id"

# Ordered: a longer suffix that shares an ending with a shorter one must come
# first, e.g. "/chat/completions" before "/completions". This is the inference
# surface that writes a SpendLogs row on success -- the same population the
# admin UI usage page counts -- so the collector and the UI report the same
# number. LLM routes are POST-only inference calls; GET reads on the same
# resources (list/status/content) are not billable and are excluded by the
# method gate in classify_billable_request.
_LLM_ROUTE_SUFFIXES: tuple[str, ...] = (
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
    "/messages",  # Anthropic /v1/messages (count_tokens does not end with /messages)
    "/videos",  # create; GET list is excluded by the POST gate
    "/remix",  # /v1/videos/{id}/remix
    "/ocr",
    "/search",  # /v1/search and /v1/vector_stores/{id}/search
    "/rag/query",
    "/rag/ingest",
    ":generateContent",  # Gemini-native /v1beta/models/{model}:generateContent
    ":streamGenerateContent",
)

# Provider passthrough prefixes (e.g. /bedrock/..., /vertex-ai/...) carry real
# inference calls that write SpendLogs rows, so they bill. Anchored to the
# routes enum so new providers are picked up without touching this module.
# /langfuse forwards observability traffic, not inference: it writes no
# SpendLogs row and must not bill.
_NON_BILLABLE_PASSTHROUGH_PREFIXES = frozenset({"/langfuse"})
_PASSTHROUGH_PREFIXES: tuple[str, ...] = tuple(
    prefix
    for prefix in LiteLLMRoutes.mapped_pass_through_routes.value
    if prefix not in _NON_BILLABLE_PASSTHROUGH_PREFIXES
)


def _classify_llm_route(path: str) -> Optional[str]:
    suffix_match = next((suffix for suffix in _LLM_ROUTE_SUFFIXES if path == suffix or path.endswith(suffix)), None)
    if suffix_match is not None:
        return suffix_match
    # Deep passthrough paths only: the bare prefix itself is not an inference call.
    return next((prefix for prefix in _PASSTHROUGH_PREFIXES if path.startswith(f"{prefix}/")), None)


_MCP_MANAGEMENT_PREFIX = "/v1/mcp"
_MCP_DYNAMIC_TRANSPORT = re.compile(r"/(?:toolset/)?[^/]+/mcp")

_A2A_INVOKE_SUFFIX = "/message/send"
_A2A_TRANSPORT_PREFIXES: tuple[str, ...] = ("/v1/a2a/", "/a2a/")


def _classify_mcp_route(path: str) -> Optional[str]:
    if path == _MCP_MANAGEMENT_PREFIX or path.startswith(f"{_MCP_MANAGEMENT_PREFIX}/"):
        return None
    if path == "/mcp" or path.startswith("/mcp/"):
        return "/mcp"
    if _MCP_DYNAMIC_TRANSPORT.fullmatch(path) is not None:
        return "/mcp"
    return None


def _classify_a2a_route(path: str) -> Optional[str]:
    if path.endswith(_A2A_INVOKE_SUFFIX) and any(path.startswith(prefix) for prefix in _A2A_TRANSPORT_PREFIXES):
        return "/a2a"
    return None


def classify_billable_request(path: str, method: str = "POST") -> Optional[tuple[BillableCategory, str]]:
    """Map a request path to its (category, normalized route), or None if not billable."""
    normalized = path.rstrip("/") or "/"

    mcp_route = _classify_mcp_route(normalized)
    if mcp_route is not None:
        return (BillableCategory.MCP, mcp_route)

    a2a_route = _classify_a2a_route(normalized)
    if a2a_route is not None:
        return (BillableCategory.A2A, a2a_route)

    # Inference calls are POSTs; GETs on these paths are reads (list videos,
    # fetch a response object), which write no SpendLogs row and must not bill.
    if method.upper() != "POST":
        return None

    llm_route = _classify_llm_route(normalized)
    if llm_route is not None:
        return (BillableCategory.LLM, llm_route)
    return None


def _extract_model_id(headers: Sequence[tuple[bytes, bytes]]) -> Optional[str]:
    return next(
        (value.decode("latin-1") for name, value in headers if name.lower() == _MODEL_ID_HEADER and value),
        None,
    )


class BillableRequestMetricsMiddleware:
    """
    Pure ASGI middleware that records one billable request per 2xx response to a
    billable endpoint. Modeled on InFlightRequestsMiddleware: it wraps `send`,
    reads the final status and the x-litellm-model-id header off the
    `http.response.start` message, and never blocks or fails the request path.
    """

    def __init__(
        self,
        app: ASGIApp,
        recorder: Optional[BillingRecorder] = None,
        recorder_factory: Optional[Callable[[], Optional[BillingRecorder]]] = None,
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

    def _resolve_recorder(self) -> Optional[BillingRecorder]:
        if not self._resolved:
            factory = self._recorder_factory
            self.recorder = factory() if factory is not None else self.recorder
            self._resolved = True
        return self.recorder

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        recorder = self._resolve_recorder()
        if recorder is None:
            await self.app(scope, receive, send)
            return

        classification = classify_billable_request(scope.get("path", ""), scope.get("method", "POST"))
        if classification is None:
            await self.app(scope, receive, send)
            return

        category, route = classification
        status_code = 0
        model_id: Optional[str] = None

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code, model_id
            if message["type"] == "http.response.start":
                status_code = message["status"]
                model_id = _extract_model_id(message.get("headers", []))
            await send(message)

        await self.app(scope, receive, send_wrapper)

        if 200 <= status_code < 300:
            recorder.record(category=category, route=route, status_code=status_code, model_id=model_id)
