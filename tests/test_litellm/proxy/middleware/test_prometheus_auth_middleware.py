import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

import litellm
from litellm.proxy._types import SpecialHeaders
from litellm.proxy.middleware.prometheus_auth_middleware import PrometheusAuthMiddleware


# Fake auth functions to simulate valid and invalid auth behavior.
async def fake_valid_auth(request, api_key):
    # Simulate valid authentication: do nothing (i.e. pass)
    return


async def fake_invalid_auth(request, api_key):
    print("running fake invalid auth", request, api_key)
    # Simulate invalid auth by raising an exception.
    raise Exception("Invalid API key")


from litellm.proxy.auth.user_api_key_auth import user_api_key_auth


@pytest.fixture
def app_with_middleware():
    """Create a FastAPI app with the PrometheusAuthMiddleware and dummy endpoints."""
    app = FastAPI()
    # Add the PrometheusAuthMiddleware to the app.
    app.add_middleware(PrometheusAuthMiddleware)

    @app.get("/metrics")
    async def metrics():
        return {"msg": "metrics OK"}

    # Also allow /metrics/ (trailing slash)
    @app.get("/metrics/")
    async def metrics_slash():
        return {"msg": "metrics OK"}

    @app.get("/chat/completions")
    async def chat():
        return {"msg": "chat completions OK"}

    @app.get("/embeddings")
    async def embeddings():
        return {"msg": "embeddings OK"}

    return app


def test_valid_auth_metrics(app_with_middleware, monkeypatch):
    """
    Test that a request to /metrics (and /metrics/) with valid auth headers passes.
    """
    # Enable auth on metrics endpoints.
    litellm.require_auth_for_metrics_endpoint = True
    # Patch the auth function to simulate a valid authentication.
    monkeypatch.setattr(
        "litellm.proxy.middleware.prometheus_auth_middleware.user_api_key_auth",
        fake_valid_auth,
    )

    client = TestClient(app_with_middleware)
    headers = {SpecialHeaders.openai_authorization.value: "valid"}

    # Test for /metrics (no trailing slash)
    response = client.get("/metrics", headers=headers)
    assert response.status_code == 200, response.text
    assert response.json() == {"msg": "metrics OK"}

    # Test for /metrics/ (with trailing slash)
    response = client.get("/metrics/", headers=headers)
    assert response.status_code == 200, response.text
    assert response.json() == {"msg": "metrics OK"}


def test_invalid_auth_metrics(app_with_middleware, monkeypatch):
    """
    Test that a request to /metrics with invalid auth headers fails with a 401.
    """
    litellm.require_auth_for_metrics_endpoint = True
    # Patch the auth function to simulate a failed authentication.
    monkeypatch.setattr(
        "litellm.proxy.middleware.prometheus_auth_middleware.user_api_key_auth",
        fake_invalid_auth,
    )

    client = TestClient(app_with_middleware)
    headers = {SpecialHeaders.openai_authorization.value: "invalid"}

    response = client.get("/metrics", headers=headers)
    assert response.status_code == 401, response.text
    assert "Unauthorized access to metrics endpoint" in response.text


def test_no_auth_metrics_when_disabled(app_with_middleware, monkeypatch):
    """
    Test that when require_auth_for_metrics_endpoint is False, requests to /metrics
    bypass the auth check.
    """
    litellm.require_auth_for_metrics_endpoint = False

    # To ensure auth is not run, patch the auth function with one that will raise if called.
    def should_not_be_called(*args, **kwargs):
        raise Exception("Auth should not be called")

    monkeypatch.setattr(
        "litellm.proxy.middleware.prometheus_auth_middleware.user_api_key_auth",
        should_not_be_called,
    )

    client = TestClient(app_with_middleware)
    response = client.get("/metrics")
    assert response.status_code == 200, response.text
    assert response.json() == {"msg": "metrics OK"}


def test_non_metrics_requests_pass_through(app_with_middleware):
    """
    Test that non-metrics endpoints pass through the middleware unaffected.
    """
    litellm.require_auth_for_metrics_endpoint = True

    client = TestClient(app_with_middleware)

    response = client.get("/chat/completions")
    assert response.status_code == 200, response.text
    assert response.json() == {"msg": "chat completions OK"}

    response = client.get("/embeddings")
    assert response.status_code == 200, response.text
    assert response.json() == {"msg": "embeddings OK"}


def test_non_metrics_requests_dont_trigger_auth(app_with_middleware, monkeypatch):
    """
    Test that non-metrics requests never trigger auth, even when auth is enabled
    and the auth function would reject the request.
    """
    litellm.require_auth_for_metrics_endpoint = True

    def should_not_be_called(*args, **kwargs):
        raise Exception("Auth should not be called for non-metrics requests")

    monkeypatch.setattr(
        "litellm.proxy.middleware.prometheus_auth_middleware.user_api_key_auth",
        should_not_be_called,
    )

    client = TestClient(app_with_middleware)

    response = client.get("/chat/completions")
    assert response.status_code == 200, response.text
    assert response.json() == {"msg": "chat completions OK"}

    response = client.get("/embeddings")
    assert response.status_code == 200, response.text
    assert response.json() == {"msg": "embeddings OK"}


@pytest.mark.asyncio
async def test_downstream_asgi_app_receives_http_request_after_auth_reads_body(
    monkeypatch,
):
    """
    Regression test: the real user_api_key_auth calls _read_request_body()
    (user_api_key_auth.py, line ~1591), which drains the ASGI `receive`
    channel. The middleware must not pass the already-drained `receive`
    straight through to the downstream app, or a mounted ASGI sub-app at
    /metrics (created by prometheus_client.make_asgi_app) blocks on
    `await receive()` and the request hangs indefinitely.

    Reproduction: patch user_api_key_auth with a stub that reads the body
    (mirroring what the real function does for admin-role callers). Install
    a pure ASGI inner app that records any message it receives. Drive the
    middleware once with a minimal HTTP scope. With the bug present, the
    inner app times out waiting on receive() because the http.request
    message was already pulled by auth. With a fix in place (e.g. a
    receive-replay wrapper in the middleware), the inner app observes the
    http.request message and completes normally.
    """
    import asyncio

    litellm.require_auth_for_metrics_endpoint = True

    # Stub that mirrors real auth: it reads the request body, consuming the
    # ASGI receive channel. The existing fake_valid_auth skips this step
    # which is why the bug is not caught by the other tests in this file.
    async def fake_auth_that_reads_body(request, api_key):
        _ = await request.body()
        return

    monkeypatch.setattr(
        "litellm.proxy.middleware.prometheus_auth_middleware.user_api_key_auth",
        fake_auth_that_reads_body,
    )

    received_by_inner: list = []

    async def inner_app(scope, receive, send):
        # Pure ASGI inner app. Emulates what a mounted app like
        # prometheus_client.make_asgi_app() does: try to receive, then
        # respond. If receive() blocks, we abort after 2s so the test can
        # fail cleanly rather than hanging.
        try:
            msg = await asyncio.wait_for(receive(), timeout=2.0)
            received_by_inner.append(msg)
        except asyncio.TimeoutError:
            received_by_inner.append({"type": "<timeout>"})
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    middleware = PrometheusAuthMiddleware(inner_app)

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/metrics",
        "raw_path": b"/metrics",
        "query_string": b"",
        "headers": [(b"authorization", b"Bearer test-admin-key")],
        "scheme": "http",
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "http_version": "1.1",
    }

    # Queue exactly one http.request message, as a real GET would deliver.
    # After it is consumed, receive() would normally block.
    queued = [{"type": "http.request", "body": b"", "more_body": False}]

    async def receive():
        if queued:
            return queued.pop(0)
        # Block indefinitely — same as real ASGI when no more messages.
        await asyncio.Event().wait()

    sent: list = []

    async def send(message):
        sent.append(message)

    await asyncio.wait_for(middleware(scope, receive, send), timeout=5.0)

    # Core assertion: the inner app must see the http.request message.
    # With the current middleware, `received_by_inner` contains
    # {"type": "<timeout>"} because `receive` was drained by the auth
    # body-read.
    assert received_by_inner, "inner app did not run"
    assert received_by_inner[0].get("type") == "http.request", (
        f"downstream app did not observe http.request; saw {received_by_inner[0]}. "
        "Middleware must replay the consumed http.request message to downstream "
        "when require_auth_for_metrics_endpoint is enabled."
    )
