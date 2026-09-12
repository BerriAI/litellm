"""
Tests for UiRelativeRedirectMiddleware (LIT-7455).

Regression coverage for: behind a TLS-terminating load balancer whose peer IP
isn't in uvicorn's FORWARDED_ALLOW_IPS, scope["scheme"] stays "http" even
though the original request was https, and the /ui mount's trailing-slash and
directory-index redirects bake that wrong scheme into an absolute Location.
These tests drive the real StaticFiles mount (not a mock) with a scope built
the same way that bug reports it -- scheme="http", Host header the client
actually used -- and assert the emitted Location no longer carries a scheme
or host at all, so the browser resolves it against its own (correct) origin.
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import RedirectResponse
from starlette.routing import Route

from litellm.proxy.middleware.ui_relative_redirect_middleware import (
    UiRelativeRedirectMiddleware,
)


def _ui_app(tmp_path):
    ui_root = tmp_path / "ui"
    ui_root.mkdir()
    (ui_root / "index.html").write_text("index")
    nested = ui_root / "teams"
    nested.mkdir()
    (nested / "index.html").write_text("teams")

    app = FastAPI()
    app.mount("/ui", StaticFiles(directory=str(ui_root), html=True), name="ui")
    app.add_middleware(UiRelativeRedirectMiddleware)
    return app


def _client_behind_untrusted_lb(app):
    """A TestClient that talks plain http to the app but claims an https Host,
    mirroring a request that reached an untrusted-peer uvicorn over the
    plaintext LB->container hop after the LB terminated TLS."""
    return TestClient(app, base_url="http://ui.example.com")


def test_is_pure_asgi_not_base_http_middleware():
    """BaseHTTPMiddleware buffers the whole response; this must be pure ASGI
    so it keeps streaming responses (e.g. the SSE proxy paths) intact."""
    assert not issubclass(UiRelativeRedirectMiddleware, BaseHTTPMiddleware)
    assert "__call__" in UiRelativeRedirectMiddleware.__dict__


def test_bare_ui_trailing_slash_redirect_is_relative(tmp_path):
    """The router's own /ui -> /ui/ redirect must no longer carry a scheme/host."""
    client = _client_behind_untrusted_lb(_ui_app(tmp_path))

    response = client.get("/ui", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/ui/"


def test_nested_route_directory_index_redirect_is_relative(tmp_path):
    """StaticFiles' own directory-index redirect (e.g. /ui/teams -> /ui/teams/)
    is a second, independent code path that must also be relativized."""
    client = _client_behind_untrusted_lb(_ui_app(tmp_path))

    response = client.get("/ui/teams", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/ui/teams/"


def test_relativized_redirect_is_actually_followable(tmp_path):
    """The whole point: a client that refuses to follow a scheme-downgraded
    absolute redirect must still be able to follow the relative one."""
    client = _client_behind_untrusted_lb(_ui_app(tmp_path))

    landed = client.get("/ui", follow_redirects=True)

    assert landed.status_code == 200
    assert landed.text == "index"


def test_query_string_and_fragment_preserved(tmp_path):
    ui_root = tmp_path / "ui"
    ui_root.mkdir()
    nested = ui_root / "mcp" / "oauth" / "callback"
    nested.mkdir(parents=True)
    (nested / "index.html").write_text("callback")

    app = FastAPI()
    app.mount("/ui", StaticFiles(directory=str(ui_root), html=True), name="ui")
    app.add_middleware(UiRelativeRedirectMiddleware)
    client = _client_behind_untrusted_lb(app)

    response = client.get(
        "/ui/mcp/oauth/callback?code=abc&state=xyz", follow_redirects=False
    )

    assert response.status_code == 307
    assert response.headers["location"] == "/ui/mcp/oauth/callback/?code=abc&state=xyz"


def test_paths_outside_ui_mount_are_untouched():
    """The middleware must not relativize redirects on unrelated routes."""

    async def redirect_elsewhere(request):
        return RedirectResponse(url="https://ui.example.com/somewhere-else", status_code=307)

    app = FastAPI(routes=[Route("/not-ui", redirect_elsewhere)])
    app.add_middleware(UiRelativeRedirectMiddleware)
    client = TestClient(app, base_url="https://ui.example.com")

    response = client.get("/not-ui", follow_redirects=False)

    assert response.headers["location"] == "https://ui.example.com/somewhere-else"


def test_cross_origin_ui_redirect_is_left_absolute():
    """A /ui redirect to a genuinely different host (e.g. an external IdP)
    must never be rewritten -- only same-origin Locations are in scope."""

    async def redirect_to_idp(request):
        return RedirectResponse(url="https://idp.example.com/authorize", status_code=307)

    app = FastAPI(routes=[Route("/ui/login", redirect_to_idp)])
    app.add_middleware(UiRelativeRedirectMiddleware)
    client = TestClient(app, base_url="https://ui.example.com")

    response = client.get("/ui/login", follow_redirects=False)

    assert response.headers["location"] == "https://idp.example.com/authorize"


def test_protocol_relative_path_is_left_untouched():
    """A Location whose path would become //host/... once scheme/netloc are
    dropped must be left alone: stripping them would make it protocol-relative
    and let the browser pick a different host."""

    async def redirect_weird_path(request):
        return RedirectResponse(
            url="https://ui.example.com//evil.example.com/x", status_code=307
        )

    app = FastAPI(routes=[Route("/ui/weird", redirect_weird_path)])
    app.add_middleware(UiRelativeRedirectMiddleware)
    client = TestClient(app, base_url="https://ui.example.com")

    response = client.get("/ui/weird", follow_redirects=False)

    assert response.headers["location"] == "https://ui.example.com//evil.example.com/x"


def test_userinfo_in_netloc_does_not_match_host_header():
    """scheme://user@host is netloc "user@host", which must not string-match
    the bare Host header -- confirms the comparison can't be tricked into
    treating a foreign-looking netloc as same-origin."""

    async def redirect_with_userinfo(request):
        return RedirectResponse(
            url="https://evil.example.com@ui.example.com/x", status_code=307
        )

    app = FastAPI(routes=[Route("/ui/x", redirect_with_userinfo)])
    app.add_middleware(UiRelativeRedirectMiddleware)
    client = TestClient(app, base_url="https://ui.example.com")

    response = client.get("/ui/x", follow_redirects=False)

    assert response.headers["location"] == "https://evil.example.com@ui.example.com/x"


def test_host_header_match_is_case_insensitive():
    async def redirect_upper(request):
        return RedirectResponse(url="HTTPS://UI.EXAMPLE.COM/ui/", status_code=307)

    app = FastAPI(routes=[Route("/ui", redirect_upper)])
    app.add_middleware(UiRelativeRedirectMiddleware)
    client = TestClient(app, base_url="https://ui.example.com")

    response = client.get("/ui", follow_redirects=False)

    assert response.headers["location"] == "/ui/"


def test_already_relative_location_is_unchanged():
    async def redirect_relative(request):
        return RedirectResponse(url="/ui/login", status_code=307)

    app = FastAPI(routes=[Route("/ui/x", redirect_relative)])
    app.add_middleware(UiRelativeRedirectMiddleware)
    client = TestClient(app, base_url="https://ui.example.com")

    response = client.get("/ui/x", follow_redirects=False)

    assert response.headers["location"] == "/ui/login"


def test_non_redirect_ui_response_is_unaffected(tmp_path):
    ui_root = tmp_path / "ui"
    ui_root.mkdir()
    (ui_root / "index.html").write_text("index")

    app = FastAPI()
    app.mount("/ui", StaticFiles(directory=str(ui_root), html=True), name="ui")
    app.add_middleware(UiRelativeRedirectMiddleware)
    client = _client_behind_untrusted_lb(app)

    response = client.get("/ui/")

    assert response.status_code == 200
    assert response.text == "index"
    assert "location" not in response.headers
