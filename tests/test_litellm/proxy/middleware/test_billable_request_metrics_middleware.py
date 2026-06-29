"""
Tests for BillableRequestMetricsMiddleware and route classification.

These verify the metering gate (records only on 2xx to a billable endpoint),
correct category/route classification, model-id extraction, and that the
middleware is a transparent pass-through when no recorder is injected.
"""

import asyncio
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
        "/v1/mcp/tools",
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
        ("/mcp", (BillableCategory.MCP, "/mcp")),
        ("/mcp/", (BillableCategory.MCP, "/mcp")),
        ("/mcp/tools/list", (BillableCategory.MCP, "/mcp")),
        ("/v1/mcp", (BillableCategory.MCP, "/v1/mcp")),
        ("/v1/mcp/servers", (BillableCategory.MCP, "/v1/mcp")),
        ("/a2a", (BillableCategory.A2A, "/a2a")),
        ("/a2a/agent-1/message/send", (BillableCategory.A2A, "/a2a")),
        ("/v1/a2a", (BillableCategory.A2A, "/v1/a2a")),
        ("/v1/a2a/discover", (BillableCategory.A2A, "/v1/a2a")),
    ],
)
def test_classify_billable(path: str, expected: Tuple[BillableCategory, str]):
    assert classify_billable_request(path) == expected


@pytest.mark.parametrize(
    "path",
    ["/health", "/health/readiness", "/metrics", "/ui", "/", "/v1/models", "/key/generate", "/v1/files"],
)
def test_classify_non_billable_returns_none(path: str):
    assert classify_billable_request(path) is None


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
    TestClient(_make_app(recorder)).post("/v1/mcp/tools")
    assert len(recorder.calls) == 1 and recorder.calls[0]["category"] == BillableCategory.MCP


def test_records_a2a_category():
    recorder = FakeRecorder()
    TestClient(_make_app(recorder)).post("/a2a/agent-1/message/send")
    assert len(recorder.calls) == 1 and recorder.calls[0]["category"] == BillableCategory.A2A


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
    TestClient(_make_app(recorder, status_code=200, model_id=None)).post("/v1/mcp/tools")
    assert recorder.calls[0]["model_id"] is None


def test_passthrough_when_recorder_is_none():
    """Non-enterprise: middleware records nothing and does not break the response."""
    response = TestClient(_make_app(None, status_code=200)).post("/v1/chat/completions")
    assert response.status_code == 200


def test_non_http_scope_is_ignored():
    recorder = FakeRecorder()

    class _Inner:
        async def __call__(self, scope, receive, send):
            return None

    mw = BillableRequestMetricsMiddleware(_Inner(), recorder=recorder)
    asyncio.run(mw({"type": "lifespan"}, None, None))  # type: ignore[arg-type]
    assert recorder.calls == []
