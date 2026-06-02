"""
Tests for _build_sampling_request header forwarding.

Verifies that the synthetic FastAPI Request built for sampling sub-calls
correctly propagates the original MCP connection's headers and client IP
so that header-dependent guardrails, routing hooks, and trace correlation
function correctly.
"""

import pytest

from litellm.proxy._experimental.mcp_server.sampling_handler import (
    _build_sampling_request,
)


class TestBuildSamplingRequest:
    """Tests for the _build_sampling_request helper."""

    def test_should_include_content_type_by_default(self):
        """Even with no raw headers, content-type must be present."""
        req = _build_sampling_request()
        headers = dict(req.headers)
        assert headers.get("content-type") == "application/json"

    def test_should_forward_raw_headers(self):
        """Headers from the original MCP connection should be forwarded."""
        raw = {
            "x-litellm-tags": "tag1,tag2",
            "x-litellm-trace-id": "trace-abc-123",
            "user-agent": "MCP-Client/1.0",
            "authorization": "Bearer sk-test",
        }
        req = _build_sampling_request(raw_headers=raw)
        headers = dict(req.headers)

        assert headers.get("x-litellm-tags") == "tag1,tag2"
        assert headers.get("x-litellm-trace-id") == "trace-abc-123"
        assert headers.get("user-agent") == "MCP-Client/1.0"
        assert headers.get("authorization") == "Bearer sk-test"

    def test_should_skip_hop_by_hop_headers(self):
        """content-length and transfer-encoding should not be forwarded."""
        raw = {
            "content-length": "42",
            "transfer-encoding": "chunked",
            "x-custom": "keep-me",
        }
        req = _build_sampling_request(raw_headers=raw)
        headers = dict(req.headers)

        assert "content-length" not in headers
        assert "transfer-encoding" not in headers
        assert headers.get("x-custom") == "keep-me"

    def test_should_not_duplicate_content_type(self):
        """If raw_headers includes content-type, don't add it twice."""
        raw = {"content-type": "text/plain"}
        req = _build_sampling_request(raw_headers=raw)
        # Count how many content-type headers are present
        ct_count = sum(1 for k, _ in req.scope["headers"] if k == b"content-type")
        assert ct_count == 1

    def test_should_inject_client_ip_as_x_forwarded_for(self):
        """client_ip should be injected as x-forwarded-for."""
        req = _build_sampling_request(client_ip="10.0.0.42")
        headers = dict(req.headers)
        assert headers.get("x-forwarded-for") == "10.0.0.42"

    def test_should_not_override_existing_x_forwarded_for(self):
        """If raw_headers already has x-forwarded-for, don't override it."""
        raw = {"x-forwarded-for": "192.168.1.1"}
        req = _build_sampling_request(raw_headers=raw, client_ip="10.0.0.42")
        headers = dict(req.headers)
        assert headers.get("x-forwarded-for") == "192.168.1.1"

    def test_should_set_correct_path(self):
        """The synthetic request should have the sampling path."""
        req = _build_sampling_request()
        assert req.scope["path"] == "/mcp/sampling/createMessage"

    def test_should_not_use_hardcoded_localhost_4000(self):
        """Regression: the old code used ('localhost', 4000)."""
        req = _build_sampling_request()
        server = req.scope["server"]
        assert server != ("localhost", 4000)
