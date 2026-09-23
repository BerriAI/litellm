from unittest.mock import Mock

import pytest
from fastapi import HTTPException, Request

from litellm.proxy._experimental.mcp_server.auth.admission import MCPAdmissionLimiter, admission_source


def test_client_and_worker_rate_budgets_recover_after_window(monkeypatch):
    monkeypatch.setenv("LITELLM_MCP_PUBLIC_RPM", "2")
    monkeypatch.setenv("LITELLM_MCP_PUBLIC_WORKER_RPM", "3")
    clock = Mock(return_value=0.0)
    limiter = MCPAdmissionLimiter(clock)
    for source in ("a", "a"):
        with limiter.admit(source):
            pass
    with pytest.raises(HTTPException) as client_error, limiter.admit("a"):
        pytest.fail("client budget must reject")
    assert client_error.value.status_code == 429
    assert client_error.value.headers == {"Retry-After": "60"}
    with limiter.admit("b"):
        pass
    clock.return_value = 59.1
    with pytest.raises(HTTPException) as worker_error, limiter.admit("c"):
        pytest.fail("worker budget must reject rotated sources")
    assert worker_error.value.headers == {"Retry-After": "1"}
    clock.return_value = 60.0
    with limiter.admit("a"), limiter.admit("c"):
        pass


def test_inflight_limits_release_on_failure_across_window_boundary(monkeypatch):
    monkeypatch.setenv("LITELLM_MCP_PUBLIC_MAX_IN_FLIGHT", "1")
    monkeypatch.setenv("LITELLM_MCP_PUBLIC_WORKER_MAX_IN_FLIGHT", "2")
    clock = Mock(return_value=0.0)
    limiter = MCPAdmissionLimiter(clock)
    with limiter.admit("a"):
        with pytest.raises(HTTPException) as client_error, limiter.admit("a"):
            pytest.fail("one client cannot occupy another permit")
        assert client_error.value.headers == {"Retry-After": "1"}
        with limiter.admit("b"):
            clock.return_value = 60.0
            with pytest.raises(HTTPException) as worker_error, limiter.admit("c"):
                pytest.fail("rotating sources cannot exceed active work")
            assert worker_error.value.status_code == 429
        with pytest.raises(RuntimeError, match="upstream"), limiter.admit("c"):
            raise RuntimeError("upstream")
    with limiter.admit("a"), limiter.admit("c"):
        pass


def test_source_capacity_never_evicts_live_budget(monkeypatch):
    monkeypatch.setenv("LITELLM_MCP_PUBLIC_MAX_SOURCES", "2")
    clock = Mock(return_value=0.0)
    limiter = MCPAdmissionLimiter(clock)
    with limiter.admit("a"):
        with limiter.admit("b"):
            pass
        for index in range(100):
            with pytest.raises(HTTPException) as error, limiter.admit(str(index)):
                pytest.fail("source spray must be bounded")
            assert error.value.status_code == 429
        clock.return_value = 60.0
        with limiter.admit("c"):
            pass
    with limiter.admit("a"):
        pass


@pytest.mark.parametrize("value", ["0", "-1", "invalid"])
def test_invalid_limits_are_not_silently_disabled(monkeypatch, value):
    monkeypatch.setenv("LITELLM_MCP_PUBLIC_RPM", value)
    with pytest.raises(ValueError, match=r"must be positive|invalid literal"):
        MCPAdmissionLimiter()


@pytest.mark.parametrize("peer,trusted,xff,expected", [
    ("198.51.100.1", [], "203.0.113.1", "198.51.100.1"),
    ("10.0.0.1", ["10.0.0.0/8"], "203.0.113.1, 198.51.100.1", "198.51.100.1"),
    ("2001:db8::1", [], "", "2001:db8::/64"),
    (None, [], "203.0.113.1", "unknown"),
])
def test_source_identity_ignores_untrusted_forwarded_addresses(monkeypatch, peer, trusted, xff, expected):
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "general_settings", {"use_x_forwarded_for": True, "mcp_trusted_proxy_ranges": trusted})
    request = Request({"type": "http", "client": (peer, 1234) if peer else None,
                       "headers": [(b"x-forwarded-for", xff.encode())]})
    assert admission_source(request) == expected
