"""
Tests for the Vertex AI pass-through incoming-header allowlist.

Regression: ``X-Serverless-Authorization`` is the Google Cloud Run
identity-aware-proxy header. Operators that front Vertex AI behind a Cloud
Run service which requires custom auth send both ``Authorization`` (the
LiteLLM virtual key, consumed by the proxy) AND ``X-Serverless-Authorization``
(the upstream token, intended for the Cloud Run service in front of Vertex).

Pre-1.83 the pass-through did not strip headers, so this worked. Adding the
allowlist (LIT-2800) dropped ``X-Serverless-Authorization`` along with every
other unrecognised header, causing the upstream to reject the request with
``Request had invalid authentication credentials``. These tests pin both the
allowlist contents and the runtime behaviour of the helper that uses it.
"""

import os
import sys

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.constants import ALLOWED_VERTEX_AI_PASSTHROUGH_HEADERS
from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
    get_vertex_ai_allowed_incoming_headers,
)


class TestVertexAllowlistConstant:
    def test_serverless_authorization_in_allowlist(self):
        # ``X-Serverless-Authorization`` is the GCP Cloud Run IAP header. Keep it
        # in the allowlist so operators fronting Vertex behind Cloud Run can
        # pass it through to the upstream service.
        assert "x-serverless-authorization" in ALLOWED_VERTEX_AI_PASSTHROUGH_HEADERS

    def test_allowlist_uses_lowercase_keys(self):
        # The helper does case-insensitive lookups via the Starlette/FastAPI
        # Headers object, but stores lowercase. Pin the convention so a future
        # ``X-Serverless-Authorization`` entry does not silently fail to match.
        for h in ALLOWED_VERTEX_AI_PASSTHROUGH_HEADERS:
            assert h == h.lower(), f"allowlist entry must be lowercase: {h!r}"

    def test_litellm_virtual_key_not_in_allowlist(self):
        # ``Authorization`` carries the LiteLLM virtual key from the client and
        # MUST NOT be forwarded — the proxy puts a fresh Vertex Bearer token
        # in its place. If this ever lands in the allowlist, the customer's
        # virtual key gets sent to Google (info leak + immediate Google 401).
        assert "authorization" not in ALLOWED_VERTEX_AI_PASSTHROUGH_HEADERS


def _make_request(headers: dict) -> Request:
    """Build a real Starlette Request via a FastAPI TestClient round-trip,
    so headers behave exactly like in the proxy (case-insensitive,
    Starlette Headers object)."""
    captured: dict = {}
    app = FastAPI()

    @app.post("/_capture")
    async def _capture(request: Request):
        captured["request"] = request
        return JSONResponse({"ok": True})

    client = TestClient(app)
    r = client.post("/_capture", headers=headers, json={})
    assert r.status_code == 200
    return captured["request"]


class TestGetVertexAllowedIncomingHeaders:
    def test_forwards_serverless_authorization(self):
        # LIT-2800 regression: this header must reach the helper's output dict.
        req = _make_request(
            {
                "Content-Type": "application/json",
                "Authorization": "Bearer sk-litellm-virtual-key",
                "X-Serverless-Authorization": "Bearer cloud-run-iap-token",
                "X-Anything-Else": "should-be-dropped",
            }
        )
        forwarded = get_vertex_ai_allowed_incoming_headers(req)

        assert (
            forwarded.get("x-serverless-authorization")
            == "Bearer cloud-run-iap-token"
        )
        # The LiteLLM virtual key must NOT leak upstream.
        assert "authorization" not in forwarded
        # Anything not on the allowlist is dropped.
        assert "x-anything-else" not in forwarded

    def test_forwards_content_type_and_anthropic_beta(self):
        req = _make_request(
            {
                "Content-Type": "application/json",
                "anthropic-beta": "extended-cache-2024",
            }
        )
        forwarded = get_vertex_ai_allowed_incoming_headers(req)
        assert forwarded.get("content-type") == "application/json"
        assert forwarded.get("anthropic-beta") == "extended-cache-2024"

    def test_drops_arbitrary_headers(self):
        req = _make_request(
            {
                "Content-Type": "application/json",
                "X-User-Identifier": "abc",
                "Cookie": "session=secret",
            }
        )
        forwarded = get_vertex_ai_allowed_incoming_headers(req)
        # Only the explicitly allowed header passes through.
        assert set(forwarded.keys()) == {"content-type"}

    def test_serverless_authorization_missing_is_absent(self):
        # Missing header => not present in forwarded dict.
        req = _make_request({"Content-Type": "application/json"})
        forwarded = get_vertex_ai_allowed_incoming_headers(req)
        assert "x-serverless-authorization" not in forwarded
