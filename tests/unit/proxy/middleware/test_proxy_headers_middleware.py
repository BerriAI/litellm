from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Final

import pytest
from fastapi import Request
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.testclient import TestClient
from starlette.websockets import WebSocket

from litellm.proxy.auth.auth_utils import _check_valid_ip, get_request_ip_address
from litellm.proxy.auth.ip_address_utils import IPAddressUtils
from litellm.proxy.auth.login_throttle import LoginThrottle
from litellm.proxy.auth.network import get_direct_client_ip, resolve_network_context
from litellm.proxy.auth.trusted_proxy_utils import require_trusted_proxy_request
from litellm.proxy.middleware.proxy_headers_middleware import TrustedProxyMiddleware


def client_for(settings: Callable[[], Mapping[str, object]], peer: str = "10.0.0.1") -> TestClient:
    async def endpoint(request: Request) -> JSONResponse:
        config: Final = dict(settings())
        try:
            require_trusted_proxy_request(request=request, general_settings=config, feature_name="test auth")
            identity_trusted = True
        except ValueError:
            identity_trusted = False
        throttle: Final = LoginThrottle.from_request(request, config, None)
        return JSONResponse(
            {
                "network": resolve_network_context(request).model_dump(),
                "ip": get_request_ip_address(request, use_x_forwarded_for=config.get("use_x_forwarded_for") is True),
                "allowed": _check_valid_ip(["203.0.113.9"], request, use_x_forwarded_for=True)[0],
                "peer": get_direct_client_ip(request),
                "scheme": request.url.scheme,
                "identity_trusted": identity_trusted,
                "login_ip": throttle.client_ip,
                "source_limit": throttle.source_limit,
                "mcp_ip": IPAddressUtils.get_mcp_client_ip(request, config),
                "secure_cookie": IPAddressUtils.is_request_https(request, config),
            }
        )

    app: Final = Starlette(routes=[Route("/", endpoint)])
    app.add_middleware(TrustedProxyMiddleware, settings=settings)
    return TestClient(app, client=(peer, 5555))


@pytest.mark.parametrize(
    "ranges,peer,xff,expected,trusted",
    [
        (None, "10.0.0.1", "203.0.113.9", "10.0.0.1", False),
        ([], "10.0.0.1", "203.0.113.9", "10.0.0.1", False),
        (["10.0.0.0/8"], "192.0.2.1", "203.0.113.9", "192.0.2.1", False),
        (["10.0.0.0/8"], "10.0.0.1", "198.51.100.4, 203.0.113.9, 10.0.0.5", "203.0.113.9", True),
        (["10.0.0.1/8"], "10.0.0.1", "203.0.113.9", "203.0.113.9", True),
        (["10.0.0.0/8"], "10.0.0.1", "10.1.1.1, 10.0.0.9", "10.1.1.1", True),
        (["10.0.0.0/8"], "10.0.0.1", "", "10.0.0.1", True),
        (["10.0.0.0/8"], "10.0.0.1", "203.0.113.9, invalid, 10.0.0.5", "invalid", True),
        (["invalid", "*"], "10.0.0.1", "203.0.113.9", "10.0.0.1", False),
        (["2001:db8::/32"], "2001:db8::1", "203.0.113.9, 2001:db8::2", "203.0.113.9", True),
        (["10.0.0.0/8"], "::ffff:10.0.0.1", "203.0.113.9", "::ffff:10.0.0.1", False),
        (["::ffff:10.0.0.0/104"], "::ffff:10.0.0.1", "203.0.113.9", "203.0.113.9", True),
    ],
)
def test_trust_chain_resolves_once_for_ip_consumers(
    ranges: list[str] | None,
    peer: str,
    xff: str,
    expected: str,
    trusted: bool,
) -> None:
    client: Final = client_for(lambda: {"trusted_proxy_ranges": ranges}, peer)
    result: Final = client.get("/", headers={"x-forwarded-for": xff, "x-forwarded-proto": "https"}).json()
    assert result["network"] == {"client_ip": expected, "host": "testserver", "via_trusted_proxy": trusted}
    assert result["ip"] == expected
    assert result["login_ip"] == expected
    assert result["peer"] == peer
    assert result["scheme"] == ("https" if trusted else "http")


def test_identity_headers_check_the_original_peer_not_the_resolved_client() -> None:
    client: Final = client_for(lambda: {"trusted_proxy_ranges": ["10.0.0.0/8"], "use_x_forwarded_for": True})
    result: Final = client.get("/", headers={"x-forwarded-for": "198.51.100.4, 203.0.113.9, 10.0.0.5"}).json()
    assert result["identity_trusted"] is True
    assert result["allowed"] is True
    assert result["ip"] == "203.0.113.9"


def test_untrusted_peer_cannot_supply_identity_or_an_allowlisted_ip() -> None:
    client: Final = client_for(
        lambda: {"trusted_proxy_ranges": ["10.0.0.0/8"], "use_x_forwarded_for": True},
        "192.0.2.1",
    )
    result: Final = client.get(
        "/", headers={"x-forwarded-for": "203.0.113.9", "x-litellm-rust-peer": "10.0.0.1"}
    ).json()
    assert result["identity_trusted"] is False
    assert result["allowed"] is False
    assert result["ip"] == "192.0.2.1"


@pytest.mark.parametrize("ranges,expected", [(None, "203.0.113.9"), ([], "10.0.0.1")])
def test_legacy_flag_only_trusts_raw_headers_when_ranges_are_unset(
    ranges: list[str] | None,
    expected: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    client: Final = client_for(lambda: {"trusted_proxy_ranges": ranges, "use_x_forwarded_for": True})
    assert client.get("/", headers={"x-forwarded-for": "203.0.113.9"}).json()["ip"] == expected
    assert client.get("/", headers={"x-forwarded-for": "203.0.113.9"}).json()["ip"] == expected
    assert caplog.text.count("deprecated for client-IP resolution") == 1


@pytest.mark.parametrize("mcp_ranges,expected", [(["10.0.0.0/8"], "198.51.100.4"), (["192.0.2.0/24"], "10.0.0.1")])
def test_mcp_keeps_its_own_trust_and_hop_policy(mcp_ranges: list[str], expected: str) -> None:
    client: Final = client_for(
        lambda: {
            "trusted_proxy_ranges": ["10.0.0.0/8"],
            "use_x_forwarded_for": True,
            "mcp_trusted_proxy_ranges": mcp_ranges,
            "mcp_xff_num_trusted_hops": 2,
        }
    )
    result: Final = client.get(
        "/", headers={"x-forwarded-for": "198.51.100.4, 203.0.113.9", "x-forwarded-proto": "https"}
    ).json()
    assert result["ip"] == "203.0.113.9"
    assert result["mcp_ip"] == expected
    assert result["secure_cookie"] is True


def test_mcp_without_forwarding_uses_original_peer_but_ui_cookies_use_resolved_scheme() -> None:
    client: Final = client_for(lambda: {"trusted_proxy_ranges": ["10.0.0.0/8"]})
    result: Final = client.get("/", headers={"x-forwarded-for": "203.0.113.9", "x-forwarded-proto": "https"}).json()
    assert result["ip"] == "203.0.113.9"
    assert result["mcp_ip"] == "10.0.0.1"
    assert result["secure_cookie"] is True


@pytest.mark.parametrize(
    "mcp_ranges,expected", [(["10.0.0.0/8"], "https://public.example"), (["192.0.2.0/24"], "http://testserver")]
)
def test_mcp_discovery_origin_uses_its_own_trust_policy(
    mcp_ranges: list[str],
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy.mcp_server.oauth_utils import get_request_base_url

    settings: Final = {
        "trusted_proxy_ranges": ["10.0.0.0/8"],
        "use_x_forwarded_for": True,
        "mcp_trusted_proxy_ranges": mcp_ranges,
    }
    monkeypatch.setattr(proxy_server, "general_settings", settings)
    monkeypatch.delenv("PROXY_BASE_URL", raising=False)

    async def endpoint(request: Request) -> JSONResponse:
        return JSONResponse({"origin": get_request_base_url(request), "scheme": request.url.scheme})

    app: Final = Starlette(routes=[Route("/", endpoint)])
    app.add_middleware(TrustedProxyMiddleware, settings=lambda: settings)
    client: Final = TestClient(app, client=("10.0.0.1", 5555))
    assert client.get(
        "/",
        headers={
            "x-forwarded-for": "203.0.113.9",
            "x-forwarded-proto": "https",
            "x-forwarded-host": "public.example",
        },
    ).json() == {"origin": expected, "scheme": "https"}


def test_config_refresh_does_not_reuse_stale_trust() -> None:
    configs: Final = iter(({"trusted_proxy_ranges": ["10.0.0.0/8"]}, {"trusted_proxy_ranges": []}))
    app: Final = Starlette(routes=[Route("/", lambda request: JSONResponse({"ip": request.client.host}))])
    app.add_middleware(TrustedProxyMiddleware, settings=lambda: next(configs))
    client: Final = TestClient(app, client=("10.0.0.1", 5555))
    assert client.get("/", headers={"x-forwarded-for": "203.0.113.9"}).json() == {"ip": "203.0.113.9"}
    assert client.get("/", headers={"x-forwarded-for": "203.0.113.9"}).json() == {"ip": "10.0.0.1"}


def test_websockets_resolve_client_and_scheme_after_lifespan_configuration() -> None:
    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        app.state.proxy_ranges = ["10.0.0.0/8"]
        yield

    async def endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_json({"client": websocket.scope["client"], "scheme": websocket.scope["scheme"]})
        await websocket.close()

    app: Final = Starlette(routes=[WebSocketRoute("/", endpoint)], lifespan=lifespan)
    app.add_middleware(TrustedProxyMiddleware, settings=lambda: {"trusted_proxy_ranges": app.state.proxy_ranges})
    with TestClient(app, client=("10.0.0.1", 5555)) as client:
        with client.websocket_connect(
            "/", headers={"x-forwarded-for": "203.0.113.9", "x-forwarded-proto": "https"}
        ) as ws:
            assert ws.receive_json() == {"client": ["203.0.113.9", 0], "scheme": "wss"}
