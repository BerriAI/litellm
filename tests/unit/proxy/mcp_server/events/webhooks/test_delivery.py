from collections.abc import Callable
from typing import Final

import httpx
import pytest

from litellm.proxy.mcp_server.events.webhooks.delivery import (
    CallbackResponse,
    PinnedUrl,
    failure_for_status,
    is_https_callback,
    pin,
    post_callback,
)
from litellm.proxy.mcp_server.outbound_credentials.result import Error, Ok
from litellm.types.mcp_server.events import CallbackFailure


def _resolver(*addresses: str) -> Callable[[str, int], object]:
    async def resolve(_host: str, _port: int) -> tuple[str, ...]:
        return addresses

    return resolve


@pytest.mark.parametrize(
    ("url", "https"),
    (
        ("https://hook.example/x", True),
        ("http://hook.example/x", False),
        ("https:///no-host", False),
        ("https://hook.example/x#frag", False),
        ("ftp://hook.example", False),
    ),
)
def test_only_https_urls_with_a_host_are_callbacks(url: str, https: bool) -> None:
    assert is_https_callback(url) is https


@pytest.mark.asyncio
async def test_pinning_connects_to_the_resolved_ip_and_keeps_the_hostname() -> None:
    pinned: Final = await pin("https://hook.example:8443/path?q=1", _resolver("93.184.216.34"))  # pyright: ignore[reportArgumentType]  # test resolver
    assert pinned == Ok(
        PinnedUrl(url="https://93.184.216.34:8443/path?q=1", host_header="hook.example:8443", hostname="hook.example")
    )


@pytest.mark.asyncio
async def test_pinning_brackets_ipv6_addresses() -> None:
    pinned: Final = await pin("https://hook.example/", _resolver("2606:2800:220:1:248:1893:25c8:1946"))  # pyright: ignore[reportArgumentType]  # test resolver
    assert isinstance(pinned, Ok) and pinned.ok.url == "https://[2606:2800:220:1:248:1893:25c8:1946]/"


@pytest.mark.parametrize(
    "addresses",
    (("10.0.0.1",), ("127.0.0.1",), ("169.254.169.254",), ("93.184.216.34", "192.168.1.5"), ("::1",), ("fc00::1",), ()),
)
@pytest.mark.asyncio
async def test_any_non_global_resolved_address_blocks_the_callback(addresses: tuple[str, ...]) -> None:
    assert await pin("https://rebind.example/", _resolver(*addresses)) == Error("connection_refused")  # pyright: ignore[reportArgumentType]  # test resolver


@pytest.mark.asyncio
async def test_an_allowlisted_host_may_resolve_to_a_private_address() -> None:
    pinned: Final = await pin(
        "https://internal.example/", _resolver("10.0.0.7"), lambda host, _: host == "internal.example"
    )  # pyright: ignore[reportArgumentType]  # test resolver
    assert isinstance(pinned, Ok) and pinned.ok.url == "https://10.0.0.7/"


@pytest.mark.asyncio
async def test_an_http_url_is_never_pinned() -> None:
    assert await pin("http://hook.example/", _resolver("93.184.216.34")) == Error("connection_refused")  # pyright: ignore[reportArgumentType]  # test resolver


@pytest.mark.asyncio
async def test_a_delivery_posts_to_the_pinned_ip_with_host_and_sni_and_follows_no_redirect() -> None:
    seen: Final[list[httpx.Request]] = []  # mutable-ok: records the transport's requests

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(302, headers={"location": "https://10.0.0.1/steal"})

    response: Final = await post_callback(
        "https://hook.example/in",
        b'{"a":1}',
        {"webhook-id": "msg_1"},
        resolve=_resolver("93.184.216.34"),  # pyright: ignore[reportArgumentType]  # test resolver
        transport=httpx.MockTransport(handler),
    )
    assert response == Ok(CallbackResponse(status=302, body=b""))
    assert len(seen) == 1
    assert seen[0].url.host == "93.184.216.34"
    assert seen[0].headers["host"] == "hook.example"
    assert seen[0].extensions["sni_hostname"] == "hook.example"
    assert seen[0].content == b'{"a":1}'


@pytest.mark.parametrize(
    ("error", "category"),
    (
        (httpx.ReadTimeout("slow"), "timeout"),
        (httpx.ConnectError("refused"), "connection_refused"),
        (httpx.ConnectError("[SSL: CERTIFICATE_VERIFY_FAILED]"), "tls_error"),
        (httpx.RemoteProtocolError("garbage"), "connection_refused"),
    ),
)
@pytest.mark.asyncio
async def test_transport_failures_map_to_fixed_categories(error: httpx.HTTPError, category: CallbackFailure) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise error

    response: Final = await post_callback(
        "https://hook.example/in",
        b"{}",
        {},
        resolve=_resolver("93.184.216.34"),  # pyright: ignore[reportArgumentType]  # test resolver
        transport=httpx.MockTransport(handler),
    )
    assert response == Error(category)


@pytest.mark.asyncio
async def test_a_large_response_body_is_read_only_up_to_the_cap() -> None:
    response: Final = await post_callback(
        "https://hook.example/in",
        b"{}",
        {},
        resolve=_resolver("93.184.216.34"),  # pyright: ignore[reportArgumentType]  # test resolver
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"x" * 200_000)),
    )
    assert isinstance(response, Ok) and len(response.ok.body) == 65536


@pytest.mark.parametrize(
    ("status", "failure"),
    ((200, None), (204, None), (299, None), (302, "http_4xx"), (404, "http_4xx"), (410, "http_4xx"), (503, "http_5xx")),
)
def test_status_categories(status: int, failure: CallbackFailure | None) -> None:
    assert failure_for_status(status) == failure
