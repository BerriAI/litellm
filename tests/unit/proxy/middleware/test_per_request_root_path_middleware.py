"""Tests for PerRequestRootPathMiddleware (``SERVER_ROOT_PATHS``).

One deployment fronting several client-visible URL path prefixes: the matched
prefix becomes that request's ``root_path``, so Starlette route matching and
``request.base_url`` — and therefore every URL the proxy emits, the MCP OAuth
discovery documents among them — resolve under the prefix the client called.
"""

from pathlib import Path
from typing import Final

import pytest
from fastapi import FastAPI, Request, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

from litellm.proxy.middleware.per_request_root_path_middleware import (
    PerRequestRootPathMiddleware,
    StrippedRootPathMiddleware,
    get_request_root_path,
    get_server_root_paths,
    normalize_root_paths,
    normalize_server_root_path,
)


class TestNormalizeRootPaths:
    def test_strips_whitespace_and_trailing_slash(self):
        assert normalize_root_paths([" /tenant-a/ ", "/tenant-b"]) == (
            "/tenant-a",
            "/tenant-b",
        )

    def test_drops_empty_entries(self):
        assert normalize_root_paths(["", "  ", "/tenant-a"]) == ("/tenant-a",)

    def test_drops_entries_without_leading_slash(self):
        # A typo'd entry must not silently match nothing at request time.
        assert normalize_root_paths(["tenant-a", "/tenant-b"]) == ("/tenant-b",)

    def test_drops_bare_root(self):
        # "/" would turn every request into a root_path rewrite; a
        # root-mounted deployment needs no entry at all.
        assert normalize_root_paths(["/", "/tenant-a"]) == ("/tenant-a",)

    def test_dedupes(self):
        assert normalize_root_paths(["/t", "/t/", " /t "]) == ("/t",)

    def test_longest_first_for_nested_prefixes(self):
        # Longest-first ordering is what makes the most-specific nested
        # prefix win at match time.
        assert normalize_root_paths(["/t", "/t/deep"]) == ("/t/deep", "/t")


class TestGetServerRootPaths:
    def test_unset_env_is_empty(self, monkeypatch):
        monkeypatch.delenv("SERVER_ROOT_PATHS", raising=False)
        assert get_server_root_paths() == ()

    def test_empty_env_is_empty(self, monkeypatch):
        monkeypatch.setenv("SERVER_ROOT_PATHS", "")
        assert get_server_root_paths() == ()

    def test_comma_separated_entries(self, monkeypatch):
        monkeypatch.setenv("SERVER_ROOT_PATHS", "/tenant-a, /tenant-b/")
        assert get_server_root_paths() == ("/tenant-a", "/tenant-b")


def _capture_scope_middleware(root_paths):
    """Middleware wired to a downstream that records the scope it received."""
    captured = {}

    async def downstream(scope, receive, send):
        captured.update(scope)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    return PerRequestRootPathMiddleware(downstream, root_paths=root_paths), captured


async def _run(mw, scope):
    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        pass

    await mw(scope, receive, send)


class TestPerRequestRootPathMiddleware:
    @pytest.mark.asyncio
    async def test_matched_prefix_becomes_root_path_path_untouched(self):
        # Starlette's router strips root_path from the (unmodified) path at
        # match time, so the middleware must NOT rewrite scope["path"].
        mw, captured = _capture_scope_middleware(["/tenant-a"])
        await _run(mw, {"type": "http", "path": "/tenant-a/mcp/x", "method": "GET", "headers": []})
        assert captured["root_path"] == "/tenant-a"
        assert captured["path"] == "/tenant-a/mcp/x"

    @pytest.mark.asyncio
    async def test_exact_prefix_matches(self):
        mw, captured = _capture_scope_middleware(["/tenant-a"])
        await _run(mw, {"type": "http", "path": "/tenant-a", "method": "GET", "headers": []})
        assert captured["root_path"] == "/tenant-a"

    @pytest.mark.asyncio
    async def test_segment_boundary_prevents_sibling_match(self):
        # /tenant-ab must not match the /tenant-a prefix.
        mw, captured = _capture_scope_middleware(["/tenant-a"])
        await _run(mw, {"type": "http", "path": "/tenant-ab/mcp", "method": "GET", "headers": []})
        assert "root_path" not in captured

    @pytest.mark.asyncio
    async def test_unmatched_path_untouched(self):
        mw, captured = _capture_scope_middleware(["/tenant-a"])
        await _run(mw, {"type": "http", "path": "/chat/completions", "method": "GET", "headers": []})
        assert "root_path" not in captured
        assert captured["path"] == "/chat/completions"

    @pytest.mark.asyncio
    async def test_longest_nested_prefix_wins(self):
        mw, captured = _capture_scope_middleware(["/t", "/t/deep"])
        await _run(mw, {"type": "http", "path": "/t/deep/mcp", "method": "GET", "headers": []})
        assert captured["root_path"] == "/t/deep"

    @pytest.mark.asyncio
    async def test_matched_prefix_overrides_scalar_root_path(self):
        # FastAPI(root_path=SERVER_ROOT_PATH) stamps the scalar before the
        # middleware stack runs; a matched dynamic prefix wins for that
        # request (combining both mechanisms is warned about at startup).
        mw, captured = _capture_scope_middleware(["/tenant-a"])
        await _run(
            mw,
            {"type": "http", "path": "/tenant-a/mcp", "root_path": "/legacy", "method": "GET", "headers": []},
        )
        assert captured["root_path"] == "/tenant-a"

    @pytest.mark.asyncio
    async def test_unmatched_request_keeps_scalar_root_path(self):
        mw, captured = _capture_scope_middleware(["/tenant-a"])
        await _run(
            mw,
            {"type": "http", "path": "/legacy/mcp", "root_path": "/legacy", "method": "GET", "headers": []},
        )
        assert captured["root_path"] == "/legacy"

    @pytest.mark.asyncio
    async def test_websocket_scope_matched(self):
        mw, captured = _capture_scope_middleware(["/tenant-a"])
        await _run(mw, {"type": "websocket", "path": "/tenant-a/ws", "headers": []})
        assert captured["root_path"] == "/tenant-a"

    @pytest.mark.asyncio
    async def test_lifespan_scope_passes_through(self):
        called = {}

        async def downstream(scope, receive, send):
            called["scope"] = scope

        mw = PerRequestRootPathMiddleware(downstream, root_paths=["/tenant-a"])
        await _run(mw, {"type": "lifespan"})
        assert called["scope"] == {"type": "lifespan"}


def _routed_client(prefixes):
    app = FastAPI()

    @app.get("/where")
    def where(request: Request):
        return {
            "base_url": str(request.base_url),
            "root_path": request.scope.get("root_path", ""),
        }

    app.add_middleware(PerRequestRootPathMiddleware, root_paths=prefixes)
    return TestClient(app)


class TestEndToEndRouting:
    def test_two_prefixes_route_on_one_app(self):
        # The property a scalar SERVER_ROOT_PATH cannot provide: two
        # client-visible prefixes served by the same app, each request
        # reconstructing its own base URL.
        client = _routed_client(["/tenant-a", "/tenant-b"])

        resp_a = client.get("/tenant-a/where")
        resp_b = client.get("/tenant-b/where")

        assert resp_a.status_code == 200
        assert resp_a.json() == {
            "base_url": "http://testserver/tenant-a/",
            "root_path": "/tenant-a",
        }
        assert resp_b.status_code == 200
        assert resp_b.json() == {
            "base_url": "http://testserver/tenant-b/",
            "root_path": "/tenant-b",
        }

    def test_unprefixed_route_still_served(self):
        client = _routed_client(["/tenant-a"])
        resp = client.get("/where")
        assert resp.status_code == 200
        assert resp.json()["base_url"] == "http://testserver/"

    def test_unlisted_prefix_404s(self):
        client = _routed_client(["/tenant-a"])
        assert client.get("/tenant-c/where").status_code == 404


class TestGetRequestRootPath:
    """``get_request_root_path`` is the accessor that plumbs the middleware's
    resolved prefix to code that doesn't have scope in hand — the 401 challenge
    builders in ``mcp_server_manager`` / ``server`` and ``get_custom_url`` on the
    SSO callback path. Reading the SERVER_ROOT_PATH scalar there would emit URLs
    under a prefix the client didn't call, and stack a second prefix onto ones it
    did (the two review points this fixture pins)."""

    def test_falls_back_to_server_root_path_env_outside_a_request(self, monkeypatch):
        # Outside a request the middleware's ContextVar is unset. The scalar
        # env still owns the answer, so pre-middleware call sites (module-load
        # UI URL builders, background tasks) behave exactly as they did.
        monkeypatch.setenv("SERVER_ROOT_PATH", "/legacy")
        assert get_request_root_path() == "/legacy"

    def test_returns_empty_string_when_no_env_and_no_request(self, monkeypatch):
        monkeypatch.delenv("SERVER_ROOT_PATH", raising=False)
        assert get_request_root_path() == ""

    def test_returns_matched_prefix_inside_a_request(self, monkeypatch):
        # With both env vars set, a request matching a SERVER_ROOT_PATHS prefix
        # must see that prefix — not the SERVER_ROOT_PATH scalar — so the URL
        # it emits stays under the prefix the router will resolve it against.
        monkeypatch.setenv("SERVER_ROOT_PATH", "/legacy")

        seen: list[str] = []
        app = FastAPI()

        @app.get("/where")
        def where():
            seen.append(get_request_root_path())
            return {}

        app.add_middleware(PerRequestRootPathMiddleware, root_paths=["/tenant-a"])
        client = TestClient(app)

        assert client.get("/tenant-a/where").status_code == 200
        assert seen == ["/tenant-a"]

    def test_unmatched_request_falls_through_to_scope_scalar(self, monkeypatch):
        # A request the middleware saw but did not match keeps whatever
        # scope["root_path"] the app was mounted under (the scalar). The
        # ContextVar still reflects the effective per-request answer, so
        # emitted URLs and the router agree even on the fallback path.
        monkeypatch.setenv("SERVER_ROOT_PATH", "/legacy")
        seen: list[str] = []

        app = FastAPI(root_path="/legacy")

        @app.get("/where")
        def where():
            seen.append(get_request_root_path())
            return {}

        app.add_middleware(PerRequestRootPathMiddleware, root_paths=["/tenant-a"])
        client = TestClient(app)

        assert client.get("/legacy/where").status_code == 200
        assert seen == ["/legacy"]

    def test_each_request_sees_its_own_prefix(self, monkeypatch):
        # Two sequential requests through the same app must each see the
        # prefix they arrived under, so one tenant's client is never sent
        # the URL of another tenant's origin.
        monkeypatch.delenv("SERVER_ROOT_PATH", raising=False)
        seen: list[tuple[str, str]] = []
        app = FastAPI()

        @app.get("/where")
        def where(tag: str):
            seen.append((tag, get_request_root_path()))
            return {}

        app.add_middleware(PerRequestRootPathMiddleware, root_paths=["/tenant-a", "/tenant-b"])
        client = TestClient(app)
        assert client.get("/tenant-a/where?tag=a").status_code == 200
        assert client.get("/tenant-b/where?tag=b").status_code == 200
        assert seen == [("a", "/tenant-a"), ("b", "/tenant-b")]

    def test_context_var_reset_after_request(self, monkeypatch):
        # A ContextVar left set after the request finishes would poison the
        # module-load-time callers that read it lazily (they'd think they were
        # inside a request under the last-seen prefix).
        monkeypatch.setenv("SERVER_ROOT_PATH", "/legacy")

        app = FastAPI()

        @app.get("/where")
        def where():
            return {"prefix": get_request_root_path()}

        app.add_middleware(PerRequestRootPathMiddleware, root_paths=["/tenant-a"])
        client = TestClient(app)
        assert client.get("/tenant-a/where").json() == {"prefix": "/tenant-a"}
        # After the request finishes, the scalar-env fallback owns the answer
        # again — nothing was left stashed from the last request's scope.
        assert get_request_root_path() == "/legacy"


class TestNormalizeServerRootPath:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("9000", "/9000"),
            ("/9000/", "/9000"),
            (" /9000 ", "/9000"),
            ("//9000", "/9000"),
            ("/a/b/", "/a/b"),
            ("", ""),
            (" ", ""),
            ("/", ""),
        ],
    )
    def test_canonical_form(self, raw: str, expected: str):
        assert normalize_server_root_path(raw) == expected


def _static_app(tmp_path: Path, root_path: str) -> FastAPI:
    ui: Final = tmp_path / "ui"
    (ui / "login").mkdir(parents=True)
    (ui / "index.html").write_text("<html>home</html>")
    (ui / "login" / "index.html").write_text("<html>login</html>")
    static: Final = tmp_path / "next" / "static"
    static.mkdir(parents=True)
    (static / "a.js").write_text("console.log(1)")
    app: Final = FastAPI(root_path=root_path)

    @app.get("/where")
    def where(request: Request):
        return {"path": request.scope["path"], "base_url": str(request.base_url)}

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        await websocket.send_json({"path": websocket.scope["path"]})
        await websocket.close()

    app.mount("/_next", StaticFiles(directory=tmp_path / "next"), name="next")
    app.mount("/ui", StaticFiles(directory=ui, html=True), name="ui")
    app.add_middleware(StrippedRootPathMiddleware)
    return app


class TestStrippedRootPathMiddleware:
    @pytest.mark.parametrize(
        ("path", "body"),
        [
            ("/ui/", "home"),
            ("/ui/login/", "login"),
            ("/_next/static/a.js", "console.log(1)"),
            ("/9000/ui/", "home"),
            ("/9000/ui/login/", "login"),
            ("/9000/_next/static/a.js", "console.log(1)"),
        ],
    )
    def test_static_files_served_with_and_without_prefix(self, tmp_path: Path, path: str, body: str):
        response = TestClient(_static_app(tmp_path, "/9000")).get(path)
        assert response.status_code == 200
        assert body in response.text

    def test_stripped_ui_redirect_keeps_prefix(self, tmp_path: Path):
        response = TestClient(_static_app(tmp_path, "/9000")).get("/ui", follow_redirects=False)
        assert response.status_code == 307
        assert response.headers["location"] == "http://testserver/9000/ui/"

    @pytest.mark.parametrize("path", ["/where", "/9000/where"])
    def test_scope_path_is_never_double_prefixed(self, tmp_path: Path, path: str):
        response = TestClient(_static_app(tmp_path, "/9000")).get(path)
        assert response.json() == {"path": "/9000/where", "base_url": "http://testserver/9000/"}

    @pytest.mark.parametrize("raw", ["9000", "/9000/", "/9000"])
    @pytest.mark.parametrize("path", ["/ui/", "/9000/ui/"])
    def test_env_value_spellings_serve_ui(self, tmp_path: Path, monkeypatch, raw: str, path: str):
        from litellm.proxy.utils import get_server_root_path

        monkeypatch.setenv("SERVER_ROOT_PATH", raw)
        response = TestClient(_static_app(tmp_path, get_server_root_path())).get(path)
        assert response.status_code == 200
        assert "home" in response.text

    def test_without_root_path_requests_pass_through(self, tmp_path: Path):
        response = TestClient(_static_app(tmp_path, "")).get("/where")
        assert response.json() == {"path": "/where", "base_url": "http://testserver/"}

    def test_matched_per_request_prefix_is_not_doubled(self, tmp_path: Path):
        app: Final = _static_app(tmp_path, "/9000")
        app.add_middleware(PerRequestRootPathMiddleware, root_paths=["/tenant-a"])
        client: Final = TestClient(app)
        assert client.get("/tenant-a/where").json()["base_url"] == "http://testserver/tenant-a/"
        assert client.get("/where").json()["base_url"] == "http://testserver/9000/"
        assert client.get("/9000/where").json()["base_url"] == "http://testserver/9000/"

    def test_stripped_websocket_sees_prefixed_path(self, tmp_path: Path):
        with TestClient(_static_app(tmp_path, "/9000")).websocket_connect("/ws") as websocket:
            assert websocket.receive_json() == {"path": "/9000/ws"}
