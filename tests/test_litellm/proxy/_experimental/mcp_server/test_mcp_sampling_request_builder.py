"""
Tests for _build_sampling_request header forwarding.

Verifies that the synthetic FastAPI Request built for sampling sub-calls
correctly propagates the original MCP connection's headers and client IP
so that header-dependent guardrails, routing hooks, and trace correlation
function correctly.
"""

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
        """Caller-supplied x-forwarded-for is stripped; resolved client_ip wins."""
        raw = {"x-forwarded-for": "192.168.1.1"}
        req = _build_sampling_request(raw_headers=raw, client_ip="10.0.0.42")
        headers = dict(req.headers)
        assert headers.get("x-forwarded-for") == "10.0.0.42"

    def test_should_set_correct_path(self):
        """The synthetic request should have the sampling path."""
        req = _build_sampling_request()
        assert req.scope["path"] == "/mcp/sampling/createMessage"

    def test_server_should_default_to_litellm_port(self):
        """Server tuple should use port 4000 (LiteLLM default), not 0."""
        req = _build_sampling_request()
        _host, _port = req.scope["server"]
        assert _port == 4000, f"Expected default LiteLLM port 4000, got {_port}"

    def test_should_populate_client_tuple_from_client_ip(self):
        """request.client.host must return the real client IP for
        IP-based routing and guardrails."""
        req = _build_sampling_request(client_ip="10.0.0.42")
        assert req.scope.get("client") is not None
        assert req.scope["client"][0] == "10.0.0.42"
        # Verify request.client.host works (Starlette Address)
        assert req.client is not None
        assert req.client.host == "10.0.0.42"

    def test_should_not_set_client_when_no_ip(self):
        """If no client_ip is provided, client should not be in scope."""
        req = _build_sampling_request()
        assert "client" not in req.scope

    def test_should_skip_all_hop_by_hop_headers(self):
        """All hop-by-hop headers must be filtered, not just content-length
        and transfer-encoding."""
        raw = {
            "content-length": "42",
            "transfer-encoding": "chunked",
            "connection": "keep-alive",
            "keep-alive": "timeout=5",
            "upgrade": "websocket",
            "te": "trailers",
            "trailer": "Expires",
            "x-custom": "keep-me",
        }
        req = _build_sampling_request(raw_headers=raw)
        headers = dict(req.headers)

        for hop_header in [
            "content-length",
            "transfer-encoding",
            "connection",
            "keep-alive",
            "upgrade",
            "te",
            "trailer",
        ]:
            assert (
                hop_header not in headers
            ), f"Hop-by-hop header '{hop_header}' should be filtered"
        assert headers.get("x-custom") == "keep-me"

    def test_should_forward_traceparent_header(self):
        """traceparent header must be forwarded for trace correlation."""
        raw = {
            "traceparent": "00-abcdef1234567890abcdef1234567890-1234567890abcdef-01",
        }
        req = _build_sampling_request(raw_headers=raw)
        headers = dict(req.headers)
        assert headers.get("traceparent") == (
            "00-abcdef1234567890abcdef1234567890-1234567890abcdef-01"
        )

    def test_should_forward_x_litellm_api_key(self):
        """x-litellm-api-key header must be forwarded for auth."""
        raw = {"x-litellm-api-key": "sk-proxy-key-123"}
        req = _build_sampling_request(raw_headers=raw)
        headers = dict(req.headers)
        assert headers.get("x-litellm-api-key") == "sk-proxy-key-123"
