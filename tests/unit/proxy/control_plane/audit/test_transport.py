from dataclasses import dataclass, field
from typing import cast

import httpx
import pytest

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy.control_plane.audit.s3 import HttpxObjectStore
from litellm.proxy.control_plane.audit.transport import HttpxPoster


@dataclass
class _RedirectingHandler:  # mutable-ok: counts the hops a following pool would take
    hops: list[str] = field(default_factory=list)

    @property
    def client(self) -> httpx.AsyncClient:
        def route(request: httpx.Request) -> httpx.Response:
            self.hops.append(str(request.url))
            if request.url.host == "sink.test":
                return httpx.Response(302, headers={"Location": "http://127.0.0.1:9/collect"})
            return httpx.Response(200, json={"leaked": True})

        return httpx.AsyncClient(transport=httpx.MockTransport(route), follow_redirects=True)


@pytest.mark.asyncio
async def test_the_poster_never_follows_a_redirect_even_on_a_following_pool() -> None:
    handler = _RedirectingHandler()
    poster = HttpxPoster(cast(AsyncHTTPHandler, handler))  # cast-ok: only the client property is used
    reply = await poster.post("https://sink.test/collect", b"[]", {"Authorization": "Splunk t"}, 1.0)
    assert reply.status_code == 302 and handler.hops == ["https://sink.test/collect"]


@pytest.mark.asyncio
async def test_the_object_store_never_follows_a_redirect_even_on_a_following_pool() -> None:
    handler = _RedirectingHandler()
    store = HttpxObjectStore(cast(AsyncHTTPHandler, handler))  # cast-ok: only the client property is used
    reply = await store.put("https://sink.test/k", b"{}", {"Authorization": "AWS4 x"}, 1.0)
    assert reply.status_code == 302 and handler.hops == ["https://sink.test/k"]
