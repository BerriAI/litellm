from collections.abc import AsyncIterator
from typing import Final

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.testclient import TestClient
from starlette.types import Message, Receive, Scope, Send

from litellm.proxy.proxy_server import app as proxy_app
from litellm.proxy.spend_tracking.currency import Currency
from litellm.proxy.spend_tracking.currency_response import MONEY_ROUTES, CurrencyResponseDecorator


def _client(currency: Currency) -> TestClient:
    app: Final = FastAPI()
    app.add_middleware(CurrencyResponseDecorator, get_currency=lambda: currency)

    @app.get("/key/info")
    async def info() -> JSONResponse:
        return JSONResponse({"spend": 10, "count": 3, "nested": [{"max_budget": 20, "spend": None}, {"spend": 0}]})

    @app.get("/budget/list")
    async def budgets() -> JSONResponse:
        return JSONResponse([{"soft_budget": 5}, {"other": 4}])

    @app.get("/team/info")
    async def text() -> Response:
        return Response("some text", media_type="text/plain")

    @app.get("/user/info")
    async def malformed() -> Response:
        return Response("{bad", media_type="application/json")

    @app.get("/team/list")
    async def stream() -> StreamingResponse:
        async def chunks() -> AsyncIterator[bytes]:
            yield b'{"spend":'
            yield b"10}"

        return StreamingResponse(chunks(), media_type="application/json")

    @app.get("/other")
    async def other() -> JSONResponse:
        return JSONResponse({"spend": 10})

    @app.get("/user/list")
    async def empty() -> JSONResponse:
        return JSONResponse([])

    @app.get("/customer/info")
    async def error() -> JSONResponse:
        return JSONResponse({"error": "not found"}, status_code=404)

    return TestClient(app)


def test_money_routes_exist() -> None:
    paths: Final = {str(getattr(route, "path", "")) for route in proxy_app.routes}
    assert set(MONEY_ROUTES) <= paths


def test_nested_and_array_values_are_converted_without_changing_originals() -> None:
    client: Final = _client(Currency(code="EUR", usd_rate=0.5))
    response: Final = client.get("/key/info")
    assert response.json() == {
        "spend": 10,
        "spend_display": 5,
        "count": 3,
        "nested": [{"max_budget": 20, "max_budget_display": 10, "spend": None}, {"spend": 0, "spend_display": 0}],
    }
    assert response.headers["x-litellm-display-currency"] == "EUR"
    assert response.headers["x-litellm-display-currency-rate"] == "0.5"
    array: Final = client.get("/budget/list")
    assert array.json() == [{"soft_budget": 5, "soft_budget_display": 2.5}, {"other": 4}]
    assert array.headers["x-litellm-display-currency"] == "EUR"
    assert client.get("/user/list").json() == []
    assert client.get("/customer/info").json() == {"error": "not found"}


def test_usd_non_json_streaming_malformed_and_unlisted_responses_are_untouched() -> None:
    usd: Final = _client(Currency(code="USD", usd_rate=1))
    response: Final = usd.get("/key/info")
    assert response.json() == {"spend": 10, "count": 3, "nested": [{"max_budget": 20, "spend": None}, {"spend": 0}]}
    assert "x-litellm-display-currency" not in response.headers
    eur: Final = _client(Currency(code="EUR", usd_rate=0.5))
    for path, body in (
        ("/team/info", b"some text"),
        ("/user/info", b"{bad"),
        ("/team/list", b'{"spend":10}'),
        ("/other", b'{"spend":10}'),
    ):
        assert eur.get(path).content == body
        assert "x-litellm-display-currency" not in eur.get(path).headers


def test_display_values_are_rounded_without_changing_usd_amounts() -> None:
    client: Final = _client(Currency(code="EUR", usd_rate=0.92))
    info: Final = client.get("/key/info").json()
    assert info["spend"] == 10
    assert info["spend_display"] == 9.2
    assert info["nested"][0]["max_budget_display"] == 18.4
    assert client.get("/budget/list").json()[0]["soft_budget_display"] == 4.6


@pytest.mark.asyncio
async def test_usd_passes_the_original_send_through() -> None:
    async def send(message: Message) -> None:
        pass

    async def receive() -> Message:
        return {"type": "http.request"}

    async def app(scope: Scope, inner_receive: Receive, inner_send: Send) -> None:
        assert inner_send is send
        assert inner_receive is receive

    middleware: Final = CurrencyResponseDecorator(app, lambda: Currency(code="USD", usd_rate=1))
    await middleware({"type": "http", "method": "GET", "path": "/key/info"}, receive, send)
