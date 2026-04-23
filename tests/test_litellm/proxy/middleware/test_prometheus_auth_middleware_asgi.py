"""
Tests that PrometheusAuthMiddleware is a pure ASGI middleware (not BaseHTTPMiddleware).

BaseHTTPMiddleware wraps streaming responses with receive_or_disconnect per chunk,
which blocks the event loop and causes severe throughput degradation.
"""
from starlette.middleware.base import BaseHTTPMiddleware

from litellm.proxy.middleware.prometheus_auth_middleware import PrometheusAuthMiddleware


def test_is_not_base_http_middleware():
    """PrometheusAuthMiddleware must NOT inherit from BaseHTTPMiddleware."""
    assert not issubclass(PrometheusAuthMiddleware, BaseHTTPMiddleware), (
        "PrometheusAuthMiddleware should be a pure ASGI middleware, not BaseHTTPMiddleware. "
        "BaseHTTPMiddleware causes severe streaming performance degradation."
    )


def test_has_asgi_call_protocol():
    """PrometheusAuthMiddleware must implement the ASGI __call__ protocol."""
    assert "__call__" in PrometheusAuthMiddleware.__dict__, (
        "PrometheusAuthMiddleware must define __call__(self, scope, receive, send)"
    )
