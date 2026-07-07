"""
Counts billable HTTP requests on enterprise deployments.

A billable request is an inbound request to an LLM inference, MCP, or A2A
endpoint that returns a 2xx status. The actual export happens in an injected
recorder (see litellm.proxy.enterprise_billing.billing_metrics); when no
recorder is injected (non-enterprise, or metering misconfigured) this
middleware is a transparent pass-through.
"""

from enum import Enum
from typing import Callable, Optional, Protocol, Sequence, runtime_checkable

from starlette.types import ASGIApp, Message, Receive, Scope, Send


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
    ":generateContent",  # Gemini-native /v1beta/models/{model}:generateContent
    ":streamGenerateContent",
)


def _classify_llm_route(path: str) -> Optional[str]:
    return next((suffix for suffix in _LLM_ROUTE_SUFFIXES if path == suffix or path.endswith(suffix)), None)


def classify_billable_request(path: str, method: str = "POST") -> Optional[tuple[BillableCategory, str]]:
    """Map a request path to its (category, normalized route), or None if not billable."""
    normalized = path.rstrip("/") or "/"

    if normalized == "/mcp" or normalized.startswith("/mcp/"):
        return (BillableCategory.MCP, "/mcp")
    if normalized == "/v1/mcp" or normalized.startswith("/v1/mcp/"):
        return (BillableCategory.MCP, "/v1/mcp")
    if normalized == "/v1/a2a" or normalized.startswith("/v1/a2a/"):
        return (BillableCategory.A2A, "/v1/a2a")
    if normalized == "/a2a" or normalized.startswith("/a2a/"):
        return (BillableCategory.A2A, "/a2a")

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
