import asyncio
import gc
from typing import TYPE_CHECKING, Final

import httpx
import pytest
from fastapi import Request
from fastapi.routing import APIRoute
from uvicorn.importer import import_from_string
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

if TYPE_CHECKING:
    from uvicorn._types import (
        ASGIReceiveCallable,
        ASGISendCallable,
        ASGISendEvent,
        HTTPDisconnectEvent,
        HTTPScope,
        Scope,
    )

from litellm.proxy.proxy_server import app as proxy_app
from litellm.proxy.rust_forward.mode import SUPERVISED_ENV, Supervised, supervised_environment
from litellm.proxy.rust_forward.peer import RustPeerMiddleware
from litellm.proxy.rust_forward.server import NO_TRUSTED_PROXIES
from litellm.proxy.server_listener import InheritedFd

Headers = tuple[tuple[bytes, bytes], ...]


async def _receive() -> "HTTPDisconnectEvent":
    return {"type": "http.disconnect"}


async def _send(message: "ASGISendEvent") -> None:
    return None


async def _seen_by_app(
    behind_rust: bool, client: tuple[str, int], headers: Headers, mode: Supervised
) -> tuple[tuple[str, int] | None, str, Headers]:
    seen: Final[asyncio.Future[tuple[tuple[str, int] | None, str, Headers]]] = (
        asyncio.get_running_loop().create_future()
    )

    async def probe(scope: "Scope", receive: "ASGIReceiveCallable", send: "ASGISendCallable") -> None:
        assert scope["type"] == "http"
        seen.set_result((scope["client"], scope["scheme"], tuple(scope["headers"])))

    proxy_headers_off: Final = mode.trusted_proxies == NO_TRUSTED_PROXIES
    direct: Final = probe if proxy_headers_off else ProxyHeadersMiddleware(probe, mode.trusted_proxies)
    app: Final = RustPeerMiddleware(probe, mode) if behind_rust else direct
    scope: Final[HTTPScope] = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "client": client,
        "server": ("127.0.0.1", 4000),
    }
    await app(scope, _receive, _send)
    return seen.result()


PEER: Final = ("203.0.113.7", 51234)
TRUSTED_PEER: Final = ("10.0.12.34", 40000)
FORWARDED: Final[Headers] = (
    (b"x-forwarded-for", b"198.51.100.1, 203.0.113.7"),
    (b"x-forwarded-proto", b"https"),
)


@pytest.mark.parametrize("peer", [PEER, TRUSTED_PEER, ("2001:db8::1", 443)])
@pytest.mark.parametrize("headers", [(), FORWARDED, ((b"x-forwarded-for", b"10.1.2.3"), (b"host", b"public.example"))])
@pytest.mark.parametrize(
    "mode",
    [
        Supervised(InheritedFd(17), "127.0.0.1"),
        Supervised(InheritedFd(17), "10.0.0.0/8"),
        Supervised(InheritedFd(17), "*"),
        Supervised(InheritedFd(17), NO_TRUSTED_PROXIES),
    ],
)
async def test_python_behind_rust_sees_what_a_direct_connection_would(
    peer: tuple[str, int], headers: Headers, mode: Supervised
) -> None:
    rendered: Final = f"[{peer[0]}]:{peer[1]}" if ":" in peer[0] else f"{peer[0]}:{peer[1]}"
    behind_rust: Final = await _seen_by_app(
        True, ("127.0.0.1", 60000), (*headers, (b"x-litellm-rust-peer", rendered.encode())), mode
    )
    direct: Final = await _seen_by_app(False, peer, headers, mode)
    assert behind_rust == direct


async def test_peer_header_from_anyone_but_the_rust_hop_is_dropped_and_ignored() -> None:
    client, _, headers = await _seen_by_app(
        True,
        ("192.0.2.9", 1000),
        ((b"x-litellm-rust-peer", b"10.0.12.34:1"),),
        Supervised(InheritedFd(17), "127.0.0.1"),
    )
    assert (client, headers) == (("192.0.2.9", 1000), ())


async def _peer_probe(request: Request) -> dict[str, object]:
    assert request.client is not None
    return {"client": [request.client.host, request.client.port], "peer": request.headers.get("x-litellm-rust-peer")}


@pytest.mark.parametrize("entry_point", ["litellm_cli.main:proxy_worker_app", "gateway.launch:worker_app"])
async def test_every_entry_point_hands_routes_the_peer_rust_saw(
    entry_point: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(SUPERVISED_ENV, supervised_environment(Supervised(InheritedFd(17), "127.0.0.1"))[SUPERVISED_ENV])
    monkeypatch.setattr(proxy_app.router, "lifespan_context", proxy_app.router.lifespan_context)
    monkeypatch.setattr(
        proxy_app.router, "routes", [APIRoute("/__rust_forward_peer_probe", _peer_probe), *proxy_app.router.routes]
    )
    try:
        served: Final = import_from_string(entry_point)()
    finally:
        gc.unfreeze()
    transport: Final = httpx.ASGITransport(app=served, client=("127.0.0.1", 60000))
    async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
        response: Final = await client.get(
            "/__rust_forward_peer_probe", headers={"x-litellm-rust-peer": "203.0.113.7:51234"}
        )
    assert response.status_code == 200, response.text
    assert response.json() == {"client": ["203.0.113.7", 51234], "peer": None}
