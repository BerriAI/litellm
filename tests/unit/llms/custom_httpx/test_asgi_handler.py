import asyncio
from collections.abc import Mapping
from typing import Final

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse
from starlette.routing import Route

from litellm.llms.custom_httpx.asgi_handler import get_async_asgi_client


@pytest.mark.asyncio
async def test_cached_client_isolates_concurrent_apps_and_request_credentials() -> None:
    ready: Final = (asyncio.Event(), asyncio.Event())

    async def call(index: int) -> Mapping[str, object]:
        async def endpoint(request: Request) -> JSONResponse:
            ready[index].set()
            await ready[1 - index].wait()
            assert request.scope["root_path"] == f"/gateway-{index}"
            assert request.client == (f"192.0.2.{index + 1}", 4321)
            assert request.headers["authorization"] == f"Bearer key-{index}"
            return JSONResponse({"app": index}, headers={"set-cookie": f"session=app-{index}; Path=/"})

        app: Final = Starlette(routes=[Route("/child", endpoint, methods=["POST"])])
        with get_async_asgi_client(app, f"/gateway-{index}", (f"192.0.2.{index + 1}", 4321)) as client:
            response: Final = await client.post(
                f"https://proxy.test/gateway-{index}/child", headers={"authorization": f"Bearer key-{index}"},
            )
            assert response.status_code == 200
            assert not client.cookies
            with get_async_asgi_client(app) as reused:
                assert reused is client
            return response.json()

    results: Final = await asyncio.wait_for(asyncio.gather(call(0), call(1)), timeout=5)
    assert results == [{"app": 0}, {"app": 1}]


@pytest.mark.asyncio
async def test_internal_client_does_not_follow_redirects_or_environment_proxies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://unreachable.invalid:8080")

    async def endpoint(request: Request) -> RedirectResponse:
        return RedirectResponse("https://external.invalid/credentials")

    app: Final = Starlette(routes=[Route("/redirect", endpoint, methods=["POST"])])
    with get_async_asgi_client(app) as client:
        response: Final = await client.post("https://proxy.test/redirect", headers={"authorization": "Bearer fixture"})
        assert response.status_code == 307
        assert not response.history
