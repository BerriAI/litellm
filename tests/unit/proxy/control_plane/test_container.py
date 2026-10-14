from dataclasses import dataclass, field
from typing import cast

import httpx
import pytest

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.proxy.control_plane.container import HandlerTransport, _environment_region


@dataclass
class _RecyclingHandler:  # mutable-ok: counts how often the pool is re-read
    """Stands in for AsyncHTTPHandler, whose client property hands out a fresh pool after cache eviction."""

    pools: list[httpx.AsyncClient] = field(default_factory=list)

    @property
    def client(self) -> httpx.AsyncClient:
        pool = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"n": len(self.pools)}))
        )
        self.pools.append(pool)
        return pool


@pytest.mark.asyncio
async def test_transport_rereads_the_handlers_pool_on_every_request() -> None:
    handler = _RecyclingHandler()
    transport = HandlerTransport(cast(AsyncHTTPHandler, handler))  # cast-ok: only the client property is used
    first = await transport.request("GET", "https://dp.test/health", timeout=1.0)
    second = await transport.request("GET", "https://dp.test/health", timeout=1.0)
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
    handler = _RedirectingHandler()
    transport = HandlerTransport(cast(AsyncHTTPHandler, handler))  # cast-ok: only the client property is used
    reply = await transport.request("GET", "https://dp.test/v1/mcp/server/oauth/x/authorize", timeout=1.0)
    assert reply.status_code == 302 and handler.hops == ["https://dp.test/v1/mcp/server/oauth/x/authorize"]


def test_the_s3_region_follows_the_aws_sdk_order_aws_region_then_default_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")
    monkeypatch.delenv("AWS_REGION_NAME", raising=False)
    assert _environment_region(BaseAWSLLM()) == "us-east-1"
    monkeypatch.delenv("AWS_REGION")
    assert _environment_region(BaseAWSLLM()) == "eu-west-1"
