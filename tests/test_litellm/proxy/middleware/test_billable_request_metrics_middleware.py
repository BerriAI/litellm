"""
Tests for BillableRequestMetricsMiddleware and route classification.

These verify the metering gate (records only on 2xx to a billable endpoint),
correct category/route classification, model-id extraction, and that the
middleware is a transparent pass-through when no recorder is injected.
"""

import asyncio
import threading
from typing import List, Optional, Tuple

import pytest
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from starlette.testclient import TestClient

from litellm.proxy.middleware.billable_request_metrics_middleware import (
    BillableCategory,
    BillableRequestMetricsMiddleware,
    _extract_model_id,
    classify_billable_request,
)
from litellm.proxy.middleware.in_flight_requests_middleware import (
    InFlightRequestsMiddleware,
)


class FakeRecorder:
    def __init__(self) -> None:
        self.calls: List[dict] = []

    def record(self, *, category: BillableCategory, route: str, status_code: int, model_id: Optional[str]) -> None:
        self.calls.append(
            {"category": category, "route": route, "status_code": status_code, "model_id": model_id}
        )


def _make_app(recorder: Optional[FakeRecorder], status_code: int = 200, model_id: Optional[str] = None) -> Starlette:
    async def handler(request: Request) -> Response:
        headers = {"x-litellm-model-id": model_id} if model_id else {}
        return JSONResponse({}, status_code=status_code, headers=headers)

    paths = [
        "/v1/chat/completions",
        "/chat/completions",
        "/v1/embeddings",
        "/v1/completions",
        "/mcp",
        "/github/mcp",
        "/toolset/my-tools/mcp",
        "/v1/mcp/tools",
        "/v1/mcp/server",
        "/a2a/agent-1/message/send",
        "/v1/a2a/discover",
        "/health",
        "/ui",
    ]
    app = Starlette(routes=[Route(p, handler, methods=["GET", "POST"]) for p in paths])
    app.add_middleware(BillableRequestMetricsMiddleware, recorder=recorder)
    return app


# ── Structure ───────────────────────────────────────────────────────────────


def test_is_pure_asgi_not_base_http_middleware():
    assert not issubclass(BillableRequestMetricsMiddleware, BaseHTTPMiddleware)


# ── classify_billable_request ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "path,expected",
    [
        ("/v1/chat/completions", (BillableCategory.LLM, "/chat/completions")),
        ("/chat/completions", (BillableCategory.LLM, "/chat/completions")),
        ("/openai/deployments/gpt-4o/chat/completions", (BillableCategory.LLM, "/chat/completions")),
        ("/engines/gpt-4o/chat/completions", (BillableCategory.LLM, "/chat/completions")),
        ("/v1/completions", (BillableCategory.LLM, "/completions")),
        ("/completions", (BillableCategory.LLM, "/completions")),
        ("/v1/embeddings", (BillableCategory.LLM, "/embeddings")),
        ("/v1/responses", (BillableCategory.LLM, "/responses")),
        ("/v1/rerank", (BillableCategory.LLM, "/rerank")),
        ("/v1/audio/transcriptions", (BillableCategory.LLM, "/audio/transcriptions")),
        # Routes from the metering-bypass finding: authenticated inference
        # endpoints that must bill and previously classified as None.
        ("/v1/images/edits", (BillableCategory.LLM, "/images/edits")),
        ("/images/edits", (BillableCategory.LLM, "/images/edits")),
        ("/openai/deployments/dall-e/images/edits", (BillableCategory.LLM, "/images/edits")),
        ("/v1/images/variations", (BillableCategory.LLM, "/images/variations")),
        ("/v1/messages", (BillableCategory.LLM, "/v1/messages")),
        ("/interactions", (BillableCategory.LLM, "/interactions")),
        ("/v1beta/interactions", (BillableCategory.LLM, "/v1beta/interactions")),
        ("/v1/videos", (BillableCategory.LLM, "/videos")),
        ("/v1/videos/video_123/remix", (BillableCategory.LLM, "/remix")),
        ("/v1/ocr", (BillableCategory.LLM, "/ocr")),
        ("/v1beta/models/gemini-2.5-pro:generateContent", (BillableCategory.LLM, ":generateContent")),
        ("/v1beta/models/gemini-2.5-pro:streamGenerateContent", (BillableCategory.LLM, ":streamGenerateContent")),
        # SpendLogs-producing routes surfaced by the route-inventory audit
        ("/v1/search", (BillableCategory.LLM, "/search")),
        ("/v1/vector_stores/vs_1/search", (BillableCategory.LLM, "/search")),
        ("/v1/rag/query", (BillableCategory.LLM, "/rag/query")),
        ("/rag/ingest", (BillableCategory.LLM, "/rag/ingest")),
        # Provider passthrough carries real inference and writes SpendLogs
        ("/bedrock/model/anthropic.claude-v2/invoke", (BillableCategory.LLM, "/bedrock")),
        ("/vertex-ai/publishers/google/models/gemini:predict", (BillableCategory.LLM, "/vertex-ai")),
        ("/cohere/v2/chat", (BillableCategory.LLM, "/cohere")),
        # Passthrough inference bills under its provider prefix
        ("/anthropic/v1/messages", (BillableCategory.LLM, "/anthropic")),
        ("/mcp", (BillableCategory.MCP, "/mcp")),
        ("/mcp/", (BillableCategory.MCP, "/mcp")),
        ("/mcp/tools/list", (BillableCategory.MCP, "/mcp")),
        ("/github/mcp", (BillableCategory.MCP, "/mcp")),
        ("/github/mcp/", (BillableCategory.MCP, "/mcp")),
        ("/toolset/my-tools/mcp", (BillableCategory.MCP, "/mcp")),
        ("/github,slack/mcp", (BillableCategory.MCP, "/mcp")),
        # REST wrapper tool execution fires the same MCP spend logging as /mcp
        ("/mcp-rest/tools/call", (BillableCategory.MCP, "/mcp")),
        ("/a2a/agent-1/message/send", (BillableCategory.A2A, "/a2a")),
        ("/v1/a2a/agent-9/message/send", (BillableCategory.A2A, "/a2a")),
    ],
)
def test_classify_billable(path: str, expected: Tuple[BillableCategory, str]):
    assert classify_billable_request(path) == expected


@pytest.mark.parametrize(
    "path",
    [
        "/health",
        "/health/readiness",
        "/metrics",
        "/ui",
        "/",
        "/v1/models",
        "/key/generate",
        "/v1/files",
        # tokenization helper, not an inference call
        "/v1/messages/count_tokens",
        # OpenAI Assistants thread messages write no SpendLogs row
        "/v1/threads/thread_abc123/messages",
        "/threads/thread_abc123/messages",
        # Google Interactions reads and cancel are not inference calls
        "/interactions/int_123",
        "/v1beta/interactions/int_123",
        "/interactions/int_123/cancel",
        "/v1beta/interactions/int_123/cancel",
        # observability passthrough writes no SpendLogs row
        "/langfuse/api/public/ingestion",
        # a bare provider prefix is not an inference call
        "/bedrock",
        "/v1/mcp",
        "/v1/mcp/tools",
        "/v1/mcp/server",
        "/v1/mcp/server/health",
        "/v1/mcp/server/some-id",
        "/v1/mcp/server/register",
        "/v1/mcp/oauth/some-id/authorize",
        "/a2a/agent-1/.well-known/agent-card.json",
        "/v1/a2a/discover",
        "/.well-known/oauth-protected-resource/github/mcp",
        "/mcp-rest/tools/list",
        "/mcp-rest/test/connection",
        "/mcp-rest/test/tools/list",
    ],
)
def test_classify_non_billable_returns_none(path: str):
    assert classify_billable_request(path) is None


@pytest.mark.parametrize(
    "path",
    [
        "/v1/mcp/tools",
        "/v1/mcp/server",
        "/v1/mcp/server/register",
        "/v1/a2a/discover",
    ],
)
def test_classify_management_writes_are_not_billable(path: str):
    assert classify_billable_request(path, "POST") is None


@pytest.mark.parametrize(
    "path",
    [
        "/a2a/agent-1",
        "/a2a/agent-1/",
        "/v1/a2a/agent-1",
    ],
)
def test_classify_bare_a2a_route_is_not_billable(path: str):
    """
    The bare A2A route multiplexes JSON-RPC methods off the request body. Only
    message/send and message/stream write a SpendLogs row; tasks/get,
    tasks/cancel and the pushNotificationConfig RPCs are forwarded upstream and
    write none. Billing the path would count those task RPCs and push the metric
    above the dashboard's successful-request count, so it must stay unbilled.
    """
    assert classify_billable_request(path, "POST") is None


@pytest.mark.parametrize(
    "path",
    ["/v1/videos", "/v1/responses", "/v1/chat/completions", "/v1/messages"],
)
def test_classify_get_reads_are_not_billable(path: str):
    """GETs on inference resources (list videos, fetch a response) write no
    SpendLogs row and must not bill; only POST inference calls count."""
    assert classify_billable_request(path, "GET") is None


def test_classify_mcp_not_method_gated():
    assert classify_billable_request("/mcp/tools/list", "GET") == (BillableCategory.MCP, "/mcp")


def test_chat_completions_not_misclassified_as_plain_completions():
    """The /chat/completions suffix must win over /completions so the route label is correct."""
    category, route = classify_billable_request("/v1/chat/completions")
    assert route == "/chat/completions"


# ── _extract_model_id ─────────────────────────────────────────────────────────


def test_extract_model_id_present():
    headers = [(b"content-type", b"application/json"), (b"x-litellm-model-id", b"deploy-123")]
    assert _extract_model_id(headers) == "deploy-123"


def test_extract_model_id_case_insensitive():
    assert _extract_model_id([(b"X-LiteLLM-Model-Id", b"deploy-9")]) == "deploy-9"


def test_extract_model_id_absent():
    assert _extract_model_id([(b"content-type", b"application/json")]) is None


# ── Middleware recording behaviour ────────────────────────────────────────────


def test_records_once_on_2xx_llm_with_model_id():
    recorder = FakeRecorder()
    TestClient(_make_app(recorder, status_code=200, model_id="deploy-7")).post("/v1/chat/completions")
    assert recorder.calls == [
        {"category": BillableCategory.LLM, "route": "/chat/completions", "status_code": 200, "model_id": "deploy-7"}
    ]


def test_records_mcp_category():
    recorder = FakeRecorder()
    TestClient(_make_app(recorder)).post("/github/mcp")
    assert len(recorder.calls) == 1 and recorder.calls[0]["category"] == BillableCategory.MCP


def test_records_a2a_category():
    recorder = FakeRecorder()
    TestClient(_make_app(recorder)).post("/a2a/agent-1/message/send")
    assert len(recorder.calls) == 1 and recorder.calls[0]["category"] == BillableCategory.A2A


def test_does_not_record_mcp_management_read():
    recorder = FakeRecorder()
    TestClient(_make_app(recorder)).get("/v1/mcp/tools")
    assert recorder.calls == []


def test_does_not_record_mcp_management_write():
    recorder = FakeRecorder()
    TestClient(_make_app(recorder)).post("/v1/mcp/server")
    assert recorder.calls == []


def test_does_not_record_a2a_discovery():
    recorder = FakeRecorder()
    TestClient(_make_app(recorder)).post("/v1/a2a/discover")
    assert recorder.calls == []


def test_does_not_record_on_4xx():
    recorder = FakeRecorder()
    TestClient(_make_app(recorder, status_code=404)).post("/v1/chat/completions")
    assert recorder.calls == []


def test_does_not_record_on_5xx():
    recorder = FakeRecorder()
    TestClient(_make_app(recorder, status_code=503)).post("/v1/chat/completions")
    assert recorder.calls == []


def test_does_not_record_non_billable_path():
    recorder = FakeRecorder()
    TestClient(_make_app(recorder, status_code=200)).get("/health")
    assert recorder.calls == []


def test_no_model_id_when_header_absent():
    recorder = FakeRecorder()
    TestClient(_make_app(recorder, status_code=200, model_id=None)).post("/github/mcp")
    assert recorder.calls[0]["model_id"] is None


def test_passthrough_when_recorder_is_none():
    """Non-enterprise: middleware records nothing and does not break the response."""
    response = TestClient(_make_app(None, status_code=200)).post("/v1/chat/completions")
    assert response.status_code == 200


def test_record_raising_does_not_fail_the_request():
    """A broken exporter must never surface to the client: the response was
    already served when record() runs, so exceptions are swallowed and logged."""

    class ExplodingRecorder:
        def record(self, *, category, route, status_code, model_id):
            raise RuntimeError("exporter down")

    app = _make_app(None, status_code=200)
    app.user_middleware.clear()
    app.add_middleware(BillableRequestMetricsMiddleware, recorder=ExplodingRecorder())
    response = TestClient(app).post("/v1/chat/completions")
    assert response.status_code == 200


def test_non_http_scope_is_ignored():
    recorder = FakeRecorder()

    class _Inner:
        async def __call__(self, scope, receive, send):
            return None

    mw = BillableRequestMetricsMiddleware(_Inner(), recorder=recorder)
    asyncio.run(mw({"type": "lifespan"}, None, None))  # type: ignore[arg-type]
    assert recorder.calls == []


# ── lazy recorder factory ─────────────────────────────────────────────────────


def test_recorder_factory_not_called_at_init():
    """The factory must run on the first request, not at middleware construction:
    building at import time captured recorder=None before the YAML config's
    environment_variables loaded the license and cert env vars."""
    calls = []

    def factory():
        calls.append(1)
        return FakeRecorder()

    class _Inner:
        async def __call__(self, scope, receive, send):
            return None

    BillableRequestMetricsMiddleware(_Inner(), recorder_factory=factory)
    assert calls == []


def test_recorder_factory_resolved_once_on_first_request():
    recorder = FakeRecorder()
    calls = []

    def factory():
        calls.append(1)
        return recorder

    client = TestClient(_make_app_with_factory(factory, status_code=200))
    client.post("/v1/chat/completions")
    client.post("/v1/chat/completions")
    assert calls == [1]
    assert len(recorder.calls) == 2


def test_recorder_factory_returning_none_is_cached():
    calls = []

    def factory():
        calls.append(1)
        return None

    client = TestClient(_make_app_with_factory(factory, status_code=200))
    assert client.post("/v1/chat/completions").status_code == 200
    assert client.post("/v1/chat/completions").status_code == 200
    assert calls == [1]


def test_recorder_factory_resolved_once_under_concurrency():
    """Concurrent first requests must not each build a recorder: every extra
    build leaks a MeterProvider and its background exporter thread."""
    calls = []
    release = threading.Event()

    def slow_factory():
        calls.append(1)
        release.wait(timeout=2)
        return FakeRecorder()

    class _Inner:
        async def __call__(self, scope, receive, send):
            return None

    mw = BillableRequestMetricsMiddleware(_Inner(), recorder_factory=slow_factory)
    threads = [threading.Thread(target=mw._resolve_recorder) for _ in range(8)]
    for t in threads:
        t.start()
    release.set()
    for t in threads:
        t.join(timeout=5)
    assert calls == [1]


def _make_app_with_factory(factory, status_code: int) -> Starlette:
    async def handler(request: Request) -> Response:
        return JSONResponse({}, status_code=status_code)

    app = Starlette(routes=[Route("/v1/chat/completions", handler, methods=["POST"])])
    app.add_middleware(BillableRequestMetricsMiddleware, recorder_factory=factory)
    return app


# ── Shutdown ordering ───────────────────────────────────────────────────────


def test_record_runs_before_request_leaves_the_in_flight_tracker():
    """
    The count is recorded after the inner app returns. If this middleware sat
    outside InFlightRequestsMiddleware, a request could be seen as drained while
    its record() had not run, letting proxy_shutdown_event flush and stop the
    exporter underneath it. Nested inside, the in-flight count still covers it.
    """
    observed: List[int] = []

    class _CountingRecorder:
        def record(self, *, category: BillableCategory, route: str, status_code: int, model_id: Optional[str]) -> None:
            observed.append(InFlightRequestsMiddleware.get_count())

    async def inner(scope, receive, send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})

    stack = InFlightRequestsMiddleware(BillableRequestMetricsMiddleware(inner, recorder=_CountingRecorder()))
    assert TestClient(stack).post("/v1/chat/completions").status_code == 200

    assert observed == [1]
    assert InFlightRequestsMiddleware.get_count() == 0


def test_billable_middleware_is_registered_inside_the_in_flight_tracker():
    """Starlette makes the last-added middleware outermost, so the in-flight
    tracker must be registered after the billing middleware to wrap it."""
    from litellm.proxy.proxy_server import app as proxy_app

    classes = [middleware.cls for middleware in proxy_app.user_middleware]
    assert classes.index(InFlightRequestsMiddleware) < classes.index(BillableRequestMetricsMiddleware)
