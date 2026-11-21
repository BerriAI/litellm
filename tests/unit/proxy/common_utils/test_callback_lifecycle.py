import asyncio
from typing import Final

import pytest
from fastapi import APIRouter, FastAPI, WebSocket
from fastapi.responses import PlainTextResponse
from fastapi.testclient import TestClient

from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy import proxy_server
from litellm.proxy._types import ProxyException
from litellm.proxy.common_utils.callback_lifecycle import close_proxy_callbacks, setup_proxy_callbacks


class _RouteCallback(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.setups = 0
        self.closed = False

    async def async_setup_proxy(self) -> APIRouter:
        self.setups += 1
        router: Final = APIRouter()
        router.add_api_route("/plugin/status", lambda: {"ok": True}, methods=["GET"])
        return router

    async def async_close_proxy(self) -> None:
        self.closed = True


class _FailingCallback(CustomLogger):
    async def async_setup_proxy(self) -> None:
        raise RuntimeError("setup failed")

    async def async_close_proxy(self) -> None:
        raise RuntimeError("close failed")


class _HangingCallback(CustomLogger):
    async def async_close_proxy(self) -> None:
        await asyncio.Event().wait()


class _PlainRouteCallback(CustomLogger):
    async def async_setup_proxy(self) -> APIRouter:
        router: Final = APIRouter()
        router.add_route("/plugin/plain", lambda request: PlainTextResponse("open"))
        return router


class _WebsocketCallback(CustomLogger):
    async def async_setup_proxy(self) -> APIRouter:
        router: Final = APIRouter()

        @router.websocket("/plugin/ws")
        async def ws(websocket: WebSocket) -> None:
            await websocket.accept()

        return router


@pytest.mark.asyncio
async def test_callback_routes_are_served_only_to_a_valid_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(proxy_server, "master_key", "sk-the-master-key")
    app: Final = FastAPI()
    await setup_proxy_callbacks(app, [_RouteCallback()])
    client: Final = TestClient(app)

    with pytest.raises(ProxyException) as missing:
        client.get("/plugin/status")
    with pytest.raises(ProxyException) as invalid:
        client.get("/plugin/status", headers={"Authorization": "Bearer sk-not-a-key"})
    allowed: Final = client.get("/plugin/status", headers={"Authorization": "Bearer sk-the-master-key"})

    assert (str(missing.value.code), str(invalid.value.code), allowed.status_code, allowed.json()) == (
        "401",
        "401",
        200,
        {"ok": True},
    ), (missing.value.message, invalid.value.message, allowed.text)


@pytest.mark.parametrize("callback_type", [_PlainRouteCallback, _WebsocketCallback])
@pytest.mark.asyncio
async def test_routes_that_cannot_carry_key_auth_abort_startup_and_are_not_served(callback_type: type[CustomLogger]):
    app: Final = FastAPI()

    with pytest.raises(ValueError, match="cannot carry LiteLLM key auth"):
        await setup_proxy_callbacks(app, [callback_type()])

    assert {route.path for route in app.routes if route.path.startswith("/plugin")} == set()


@pytest.mark.asyncio
async def test_callback_listed_twice_is_set_up_and_closed_once_and_non_loggers_are_skipped():
    callback: Final = _RouteCallback()
    app: Final = FastAPI()

    await setup_proxy_callbacks(app, [callback, "langfuse", callback, print])
    await close_proxy_callbacks([callback, "langfuse", callback, print])

    assert (callback.setups, callback.closed) == (1, True)


@pytest.mark.asyncio
async def test_callback_without_hooks_adds_no_routes_and_new_routes_reach_the_openapi_schema():
    app: Final = FastAPI()
    await setup_proxy_callbacks(app, [CustomLogger()])
    paths_before: Final = set(app.openapi()["paths"])

    await setup_proxy_callbacks(app, [_RouteCallback()])

    assert (paths_before, set(app.openapi()["paths"])) == (set(), {"/plugin/status"})


@pytest.mark.asyncio
async def test_failed_setup_aborts_startup():
    with pytest.raises(RuntimeError, match="setup failed"):
        await setup_proxy_callbacks(FastAPI(), [_FailingCallback()])


@pytest.mark.asyncio
async def test_failing_and_hanging_cleanups_do_not_stop_other_callbacks_from_closing():
    healthy: Final = _RouteCallback()

    await close_proxy_callbacks([_FailingCallback(), _HangingCallback(), healthy], timeout=0.05)

    assert healthy.closed
