"""
Integration tests for Anthropic-shaped error responses on POST /v1/messages.

When a request to the Anthropic-compatible `/v1/messages` endpoint fails,
the response body must be Anthropic-shaped:

    {"type": "error", "error": {"type": <enum>, "message": <str>}}

and NOT the OpenAI-shaped ProxyException envelope
(`{"error": {message, type, param, code}}`), nor FastAPI's
HTTPException `{"detail": ...}` wrapper.

See litellm/proxy/anthropic_endpoints/endpoints.py:anthropic_response().
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.proxy_server import app
from litellm.proxy.utils import ProxyLogging

from fastapi.testclient import TestClient

client = TestClient(app)


@pytest.fixture
def setup_proxy(monkeypatch):
    """Wire a minimal proxy_logging_obj + auth override for /v1/messages."""
    proxy_logging_obj = ProxyLogging(
        user_api_key_cache=DualCache(default_in_memory_ttl=1)
    )
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging_obj
    )
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        api_key="test-key"
    )
    try:
        yield
    finally:
        app.dependency_overrides.clear()


def _make_request(monkeypatch, raise_exc: Exception):
    """POST /v1/messages with base_process_llm_request stubbed to raise."""

    async def _raise(*args, **kwargs):
        raise raise_exc

    monkeypatch.setattr(
        ProxyBaseLLMRequestProcessing, "base_process_llm_request", _raise
    )
    return client.post(
        "/v1/messages",
        json={
            "model": "all-anthropic/claude-opus-4-6",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
        },
        headers={"Authorization": "Bearer test-key"},
    )


def test_rate_limit_passthrough_preserves_upstream_type(setup_proxy, monkeypatch):
    """Upstream Anthropic rate_limit_error JSON is passed through, not re-typed."""
    upstream = (
        'AnthropicException - {"type":"error","error":'
        '{"type":"rate_limit_error","message":"slow down"}}'
    )
    exc = litellm.RateLimitError(
        message=upstream, llm_provider="anthropic", model="claude-opus-4-6"
    )
    resp = _make_request(monkeypatch, exc)

    assert resp.status_code == 429
    body = resp.json()
    # Top-level Anthropic envelope — NOT wrapped in {"detail": ...}.
    assert "detail" not in body
    assert body["type"] == "error"
    assert body["error"]["type"] == "rate_limit_error"
    assert body["error"]["message"] == "slow down"
    # OpenAI-only fields must be absent.
    assert "param" not in body["error"]
    assert "code" not in body["error"]


def test_bad_request_maps_to_invalid_request_error(setup_proxy, monkeypatch):
    exc = litellm.BadRequestError(
        message="missing max_tokens", llm_provider="anthropic", model="claude-opus-4-6"
    )
    resp = _make_request(monkeypatch, exc)

    assert resp.status_code == 400
    body = resp.json()
    assert "detail" not in body
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"


def test_generic_exception_maps_to_api_error(setup_proxy, monkeypatch):
    """A bare Exception (no status_code/message) → 500 api_error, clean message."""
    resp = _make_request(monkeypatch, Exception("boom"))

    assert resp.status_code == 500
    body = resp.json()
    assert "detail" not in body
    assert body["type"] == "error"
    assert body["error"]["type"] == "api_error"
    assert body["error"]["message"] == "boom"


def test_class_prefix_stripped_from_plain_message(setup_proxy, monkeypatch):
    """Router-side errors (no embedded JSON) get the litellm.X: prefix stripped."""
    exc = litellm.RateLimitError(
        message="No deployments available",
        llm_provider="anthropic",
        model="claude-opus-4-6",
    )
    resp = _make_request(monkeypatch, exc)

    body = resp.json()
    assert body["error"]["type"] == "rate_limit_error"
    # litellm.RateLimitError class prefix must not leak into the message.
    assert "litellm." not in body["error"]["message"]
    assert "No deployments available" in body["error"]["message"]


def test_litellm_headers_present_on_error(setup_proxy, monkeypatch):
    """Error responses still carry the x-litellm-* observability headers."""
    exc = litellm.BadRequestError(
        message="bad", llm_provider="anthropic", model="claude-opus-4-6"
    )
    resp = _make_request(monkeypatch, exc)
    header_keys = {k.lower() for k in resp.headers.keys()}
    # x-litellm-version is always emitted by get_custom_headers; its presence
    # proves the custom-header block runs on the error path (regression guard:
    # JSONResponse must carry headers=..., unlike the old ProxyException path).
    assert "x-litellm-version" in header_keys
