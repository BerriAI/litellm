from dataclasses import dataclass, field
from typing import Final, cast

import httpx
import pytest

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy.control_plane.container import HandlerTransport, _shared_http_handler
from litellm.proxy.control_plane.settings import ControlPlaneSettings


def test_shared_http_handler_does_not_follow_redirects() -> None:
    handler: Final = _shared_http_handler(
        ControlPlaneSettings.model_validate({"LITELLM_MASTER_KEY": "01234567890123456789012345678901"})
    )
    assert handler.follow_redirects is False


@dataclass
class _RecyclingHandler:  # mutable-ok: counts how often the pool is re-read
    """Stands in for AsyncHTTPHandler, whose client property hands out a fresh pool after cache eviction."""

    pools: list[httpx.AsyncClient] = field(default_factory=list)

    @property
    def client(self) -> httpx.AsyncClient:
        pool: Final = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"n": len(self.pools)}))
        )
        self.pools.append(pool)
        return pool


@pytest.mark.asyncio
async def test_transport_rereads_the_handlers_pool_on_every_request() -> None:
    handler: Final = _RecyclingHandler()
    transport: Final = HandlerTransport(cast(AsyncHTTPHandler, handler))  # cast-ok: only the client property is used
    first: Final = await transport.request("GET", "https://dp.test/health", timeout=1.0)
    second: Final = await transport.request("GET", "https://dp.test/health", timeout=1.0)
    assert (first.json(), second.json()) == ({"n": 1}, {"n": 2}), "a pool closed by eviction must not be reused"


@dataclass
class _RedirectingHandler:  # mutable-ok: counts the hops a following pool would take
    hops: list[str] = field(default_factory=list)

    @property
    def client(self) -> httpx.AsyncClient:
        def route(request: httpx.Request) -> httpx.Response:
            self.hops.append(str(request.url))
            if request.url.host == "dp.test":
                return httpx.Response(302, headers={"Location": "http://127.0.0.1:9/secret"})
            return httpx.Response(200, json={"leaked": True})

        return httpx.AsyncClient(transport=httpx.MockTransport(route), follow_redirects=True)


@pytest.mark.asyncio
async def test_transport_never_follows_a_dataplane_redirect_even_on_a_following_pool() -> None:
    handler: Final = _RedirectingHandler()
    transport: Final = HandlerTransport(cast(AsyncHTTPHandler, handler))  # cast-ok: only the client property is used
    reply: Final = await transport.request("GET", "https://dp.test/v1/mcp/server/oauth/x/authorize", timeout=1.0)
    assert reply.status_code == 302 and handler.hops == ["https://dp.test/v1/mcp/server/oauth/x/authorize"]
