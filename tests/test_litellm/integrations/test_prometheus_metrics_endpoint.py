"""
Regression tests for: GET /metrics returning 200 directly without a 307 redirect.

Before the fix, app.mount("/metrics", metrics_app) caused Starlette to serve the
ASGI app only at /metrics/ (trailing slash). A request to /metrics (no slash) got
a 307 Temporary Redirect to /metrics/ — two round trips instead of one.

The fix mounts at "/metrics/" AND adds app.add_route("/metrics", ...) so both
paths respond with 200 directly.

These tests build a minimal FastAPI/Starlette app and apply the same mount pattern
used in PrometheusLogger._mount_metrics_endpoint(), so they run without needing
the full proxy dependency stack.

Ref: https://github.com/BerriAI/litellm/issues/33676
"""

import fastapi
import pytest
from starlette.requests import Request
from starlette.responses import Response
from starlette.testclient import TestClient

_METRICS_BODY = b"# prometheus metrics\n"
_METRICS_CONTENT_TYPE = "text/plain; version=0.0.4"


async def _fake_metrics_app(scope, receive, send) -> None:
    """Minimal ASGI app mimicking prometheus_client.make_asgi_app().
    Used with app.mount() which calls it as a raw ASGI callable."""
    if scope["type"] == "http":
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [[b"content-type", _METRICS_CONTENT_TYPE.encode()]],
            }
        )
        await send({"type": "http.response.body", "body": _METRICS_BODY})


async def _fake_metrics_view(request: Request) -> Response:
    """Same content as _fake_metrics_app but as a Starlette view function.
    Used with app.add_route() which wraps the callable as a view (passes Request)."""
    return Response(content=_METRICS_BODY, media_type=_METRICS_CONTENT_TYPE)


def _make_app_with_old_mount() -> fastapi.FastAPI:
    """Reproduces the bug: mount only at '/metrics' (no trailing slash).
    Starlette redirects GET /metrics -> /metrics/ with a 307."""
    app = fastapi.FastAPI()
    app.mount("/metrics", _fake_metrics_app)
    return app


def _make_app_with_fixed_mount() -> fastapi.FastAPI:
    """Applies the fix: mount at '/metrics/' AND add a direct route for '/metrics'.
    app.add_route receives a view function (called with Request), while
    app.mount receives a raw ASGI callable (called with scope/receive/send)."""
    app = fastapi.FastAPI()
    app.mount("/metrics/", _fake_metrics_app)
    app.add_route("/metrics", _fake_metrics_view)
    return app


class TestBugReproduction:
    def test_old_mount_redirects_no_slash(self):
        """Confirm the bug: app.mount('/metrics', ...) causes a 307 on GET /metrics.
        This test documents the pre-fix behaviour and must keep passing (it tests the
        broken pattern, not the fix)."""
        client = TestClient(_make_app_with_old_mount(), raise_server_exceptions=True)
        response = client.get("/metrics", follow_redirects=False)
        assert response.status_code == 307, (
            "Expected the unfixed mount to issue a 307 redirect on GET /metrics. "
            "If this fails, Starlette may have changed its redirect behaviour."
        )


class TestFixedMountNoRedirect:
    def test_metrics_no_slash_returns_200_directly(self):
        """GET /metrics must return 200 without any redirect after the fix.

        Regression: before the fix a request to /metrics got a 307 to /metrics/
        because app.mount('/metrics', ...) only served the ASGI app at /metrics/.
        After the fix app.add_route('/metrics', ...) serves it directly.
        """
        client = TestClient(_make_app_with_fixed_mount(), raise_server_exceptions=True)
        response = client.get("/metrics", follow_redirects=False)

        assert response.status_code == 200, (
            f"Expected 200 on GET /metrics but got {response.status_code}. "
            "This means /metrics is still issuing a redirect instead of serving directly."
        )

    def test_metrics_with_slash_also_returns_200(self):
        """GET /metrics/ must still return 200 — the mount point itself must keep working."""
        client = TestClient(_make_app_with_fixed_mount(), raise_server_exceptions=True)
        response = client.get("/metrics/", follow_redirects=False)

        assert response.status_code == 200, (
            f"Expected 200 on GET /metrics/ but got {response.status_code}."
        )

    def test_metrics_no_slash_does_not_redirect(self):
        """GET /metrics must NOT return any 3xx redirect after the fix."""
        client = TestClient(_make_app_with_fixed_mount(), raise_server_exceptions=True)
        response = client.get("/metrics", follow_redirects=False)

        assert not (300 <= response.status_code < 400), (
            f"GET /metrics returned a {response.status_code} redirect after the fix. "
            "Prometheus scrapers may not follow redirects, causing metrics to appear unavailable."
        )

    def test_metrics_no_slash_returns_prometheus_body(self):
        """GET /metrics must return the metrics body, not an empty redirect response."""
        client = TestClient(_make_app_with_fixed_mount(), raise_server_exceptions=True)
        response = client.get("/metrics", follow_redirects=False)

        assert b"prometheus" in response.content, (
            "Response body does not look like Prometheus metrics output."
        )
