"""
Counts billable HTTP requests on enterprise deployments.

A billable request is an inbound request to an LLM inference, MCP, or A2A
endpoint that returns a 2xx status. The actual export happens in an injected
recorder (see litellm.proxy.enterprise_billing.billing_metrics); when no
recorder is injected (non-enterprise, or metering misconfigured) this
middleware is a transparent pass-through.
"""

from enum import Enum
from typing import Optional, Protocol, Sequence, Tuple, runtime_checkable

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
# first, e.g. "/chat/completions" before "/completions".
_LLM_ROUTE_SUFFIXES: Tuple[str, ...] = (
    "/chat/completions",
    "/completions",
    "/embeddings",
    "/responses",
    "/rerank",
    "/moderations",
    "/images/generations",
    "/audio/transcriptions",
    "/audio/translations",
    "/audio/speech",
)


def _classify_llm_route(path: str) -> Optional[str]:
    return next((suffix for suffix in _LLM_ROUTE_SUFFIXES if path == suffix or path.endswith(suffix)), None)


def classify_billable_request(path: str) -> Optional[Tuple[BillableCategory, str]]:
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

    llm_route = _classify_llm_route(normalized)
    if llm_route is not None:
        return (BillableCategory.LLM, llm_route)
    return None


def _extract_model_id(headers: Sequence[Tuple[bytes, bytes]]) -> Optional[str]:
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

    def __init__(self, app: ASGIApp, recorder: Optional[BillingRecorder] = None) -> None:
        self.app = app
        self.recorder = recorder

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        recorder = self.recorder
        if recorder is None or scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        classification = classify_billable_request(scope.get("path", ""))
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
