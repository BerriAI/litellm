"""
Tests for SecurityHeadersMiddleware.

Verifies anti-framing / content-type headers are present on every response and
that HSTS is opt-in via LITELLM_ENABLE_HSTS.
"""

from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from starlette.staticfiles import StaticFiles
from starlette.testclient import TestClient

from litellm.proxy.middleware.security_headers_middleware import (
    SecurityHeadersMiddleware,
)


def _make_client(handler):
    app = Starlette(routes=[Route("/", handler)])
    app.add_middleware(SecurityHeadersMiddleware)
    return TestClient(app)


async def _ok(request):
    return JSONResponse({"ok": True})


def test_is_pure_asgi_not_base_http_middleware():
    """BaseHTTPMiddleware degrades streaming; this must be pure ASGI."""
    assert not issubclass(SecurityHeadersMiddleware, BaseHTTPMiddleware)
    assert "__call__" in SecurityHeadersMiddleware.__dict__


def test_static_security_headers_present():
    resp = _make_client(_ok).get("/")
    assert resp.headers["x-frame-options"] == "DENY"
    assert resp.headers["content-security-policy"] == "frame-ancestors 'none'"
    assert resp.headers["x-content-type-options"] == "nosniff"


def test_hsts_absent_by_default(monkeypatch):
    monkeypatch.delenv("LITELLM_ENABLE_HSTS", raising=False)
    resp = _make_client(_ok).get("/")
    assert "strict-transport-security" not in resp.headers


def test_hsts_present_when_enabled(monkeypatch):
    monkeypatch.setenv("LITELLM_ENABLE_HSTS", "true")
    resp = _make_client(_ok).get("/")
    assert resp.headers["strict-transport-security"] == ("max-age=31536000; includeSubDomains")


def test_hsts_not_enabled_by_arbitrary_value(monkeypatch):
    monkeypatch.setenv("LITELLM_ENABLE_HSTS", "1")
    resp = _make_client(_ok).get("/")
    assert "strict-transport-security" not in resp.headers


def test_does_not_override_existing_header(monkeypatch):
    """A route that sets its own X-Frame-Options must win."""

    async def custom(request):
        return Response("hi", headers={"X-Frame-Options": "SAMEORIGIN"})

    resp = _make_client(custom).get("/")
    assert resp.headers["x-frame-options"] == "SAMEORIGIN"
    # other headers still applied
    assert resp.headers["x-content-type-options"] == "nosniff"


def test_response_without_a_cache_policy_is_not_stored():
    resp = _make_client(_ok).get("/")
    assert resp.headers["cache-control"] == "no-store"


def test_error_without_a_cache_policy_is_not_stored():
    async def denied(request):
        return JSONResponse({"error": "no"}, status_code=401)

    resp = _make_client(denied).get("/")
    assert resp.status_code == 401
    assert resp.headers["cache-control"] == "no-store"


def test_explicit_cache_policy_is_preserved():
    async def published(request):
        return JSONResponse({"keys": []}, headers={"Cache-Control": "public, max-age=60"})

    resp = _make_client(published).get("/")
    assert resp.headers["cache-control"] == "public, max-age=60"
    assert resp.headers["x-content-type-options"] == "nosniff"


def _static_client(directory):
    app = Starlette()
    app.mount("/ui", StaticFiles(directory=directory, html=True), name="ui")
    app.add_middleware(SecurityHeadersMiddleware)
    return TestClient(app)


def test_successful_non_html_static_file_keeps_its_cache_policy(tmp_path):
    (tmp_path / "app.js").write_text("console.log(1)\n")
    (tmp_path / "index.html").write_text('<html><script src="app.js"></script></html>\n')
    client = _static_client(tmp_path)

    asset = client.get("/ui/app.js")
    assert asset.status_code == 200
    assert asset.text == "console.log(1)\n"
    assert "text/html" not in asset.headers["content-type"]
    assert "no-store" not in asset.headers.get("cache-control", "")

    page = client.get("/ui/")
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert page.headers["cache-control"] == "no-store"

    missing = client.get("/ui/missing.js")
    assert missing.status_code == 404
    assert missing.headers["cache-control"] == "no-store"


def test_conditional_static_html_is_not_stored(tmp_path):
    (tmp_path / "index.html").write_text("<html>ok</html>\n")
    client = _static_client(tmp_path)

    page = client.get("/ui/")
    assert page.status_code == 200
    conditional = client.get("/ui/", headers={"If-None-Match": page.headers["etag"]})
    assert conditional.status_code == 304
    assert conditional.headers["cache-control"] == "no-store"


def test_conditional_static_javascript_is_not_stored(tmp_path):
    (tmp_path / "app.js").write_text("console.log(1)\n")
    client = _static_client(tmp_path)

    asset = client.get("/ui/app.js")
    assert asset.status_code == 200
    assert "no-store" not in asset.headers.get("cache-control", "")
    conditional = client.get("/ui/app.js", headers={"If-None-Match": asset.headers["etag"]})
    assert conditional.status_code == 304
    assert conditional.headers["cache-control"] == "no-store"
