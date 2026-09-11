import sys
from types import ModuleType

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from starlette.routing import Match

from litellm.proxy.route_priority import HOT_ROUTE_PATHS, hot_routes_first

FILLER_COUNT = 300


def _routes_scanned_before_dispatch(app: FastAPI, method: str, path: str) -> int:
    """Number of route.matches() calls Starlette's Router.app makes before it finds a full match."""
    scope = {"type": "http", "method": method, "path": path, "root_path": "", "headers": [], "query_string": b""}
    for i, route in enumerate(app.router.routes):
        match, _ = route.matches(dict(scope))
        if match == Match.FULL:
            return i + 1
    raise AssertionError(f"{method} {path} has no route")


def _hot_router() -> APIRouter:
    router = APIRouter()

    @router.get("/health/liveliness")
    @router.get("/health/liveness")
    async def liveliness():
        return "I'm alive!"

    @router.post("/v1/chat/completions")
    @router.post("/chat/completions")
    async def chat():
        return {"object": "chat.completion"}

    return router


def _app_with_filler_then_hot_routes() -> FastAPI:
    app = FastAPI()
    for i in range(FILLER_COUNT):

        @app.get(f"/filler/{i}")
        async def filler(i: int = i):
            return {"filler": i}

    app.include_router(_hot_router())
    return app


def test_hot_routes_first_puts_hot_routes_ahead_of_everything_else():
    app = _app_with_filler_then_hot_routes()
    assert _routes_scanned_before_dispatch(app, "GET", "/health/liveliness") > FILLER_COUNT

    app.router.routes = hot_routes_first(app.router.routes)

    hot_count = sum(1 for r in app.router.routes if getattr(r, "path", None) in HOT_ROUTE_PATHS)
    assert _routes_scanned_before_dispatch(app, "GET", "/health/liveliness") <= hot_count
    assert _routes_scanned_before_dispatch(app, "GET", "/health/liveness") <= hot_count
    assert _routes_scanned_before_dispatch(app, "POST", "/v1/chat/completions") <= hot_count
    assert _routes_scanned_before_dispatch(app, "POST", "/chat/completions") <= hot_count


def test_hot_routes_first_keeps_the_other_routes_in_order_and_dispatching():
    app = _app_with_filler_then_hot_routes()
    before = [r.path for r in app.router.routes if getattr(r, "path", "").startswith("/filler/")]

    app.router.routes = hot_routes_first(app.router.routes)

    after = [r.path for r in app.router.routes if getattr(r, "path", "").startswith("/filler/")]
    assert after == before
    client = TestClient(app)
    assert client.get("/health/liveliness").json() == "I'm alive!"
    assert client.get("/filler/7").json() == {"filler": 7}
    assert client.post("/v1/chat/completions").json() == {"object": "chat.completion"}
    assert client.get("/v1/chat/completions").status_code == 405
    assert client.get("/does/not/exist").status_code == 404


def test_hot_routes_first_is_idempotent():
    app = _app_with_filler_then_hot_routes()
    once = hot_routes_first(app.router.routes)
    assert hot_routes_first(once) == once


@pytest.mark.asyncio
async def test_lazy_loaded_hot_route_moves_to_the_front(monkeypatch):
    from litellm.proxy._lazy_features import LazyFeature, LazyFeatureMiddleware

    messages_router = APIRouter()

    @messages_router.post("/v1/messages")
    async def messages():
        return {"type": "message"}

    fake_module = ModuleType("fake_anthropic_endpoints")
    fake_module.router = messages_router
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)

    target_app = _app_with_filler_then_hot_routes()
    target_app.router.routes = hot_routes_first(target_app.router.routes)

    async def downstream(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    feat = LazyFeature(name="anthropic", module_path=fake_module.__name__, path_prefixes=("/v1/messages",))
    mw = LazyFeatureMiddleware(downstream, fastapi_app=target_app, features=(feat,))

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        pass

    await mw({"type": "http", "path": "/v1/messages", "method": "POST", "headers": []}, receive, send)

    hot_count = sum(1 for r in target_app.router.routes if getattr(r, "path", None) in HOT_ROUTE_PATHS)
    assert _routes_scanned_before_dispatch(target_app, "POST", "/v1/messages") <= hot_count
    assert TestClient(target_app).post("/v1/messages").json() == {"type": "message"}


def test_proxy_app_dispatches_liveness_and_chat_completions_before_the_rest():
    from litellm.proxy.proxy_server import app

    hot_count = sum(1 for r in app.router.routes if getattr(r, "path", None) in HOT_ROUTE_PATHS)
    assert hot_count >= 4
    assert len(app.router.routes) > 100
    assert _routes_scanned_before_dispatch(app, "GET", "/health/liveliness") <= hot_count
    assert _routes_scanned_before_dispatch(app, "GET", "/health/liveness") <= hot_count
    assert _routes_scanned_before_dispatch(app, "POST", "/v1/chat/completions") <= hot_count
    assert _routes_scanned_before_dispatch(app, "POST", "/chat/completions") <= hot_count
