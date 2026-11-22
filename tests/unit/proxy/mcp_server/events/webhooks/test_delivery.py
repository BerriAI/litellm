from collections.abc import AsyncIterator, Callable
from typing import Final

import anyio
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
        ("https://hook.example:notaport/x", False),
        ("https://hook.example:99999/x", False),
        ("https://hook.example:8443/x", True),
    ),
)
def test_only_https_urls_with_a_host_are_callbacks(url: str, https: bool) -> None:
    assert is_https_callback(url) is https


@pytest.mark.asyncio
async def test_pinning_connects_to_the_resolved_ip_and_keeps_the_hostname() -> None:
    pinned: Final = await pin("https://hook.example:8443/path?q=1", _resolver("93.184.216.34"))  # pyright: ignore[reportArgumentType]  # test resolver
    assert pinned == Ok(
        PinnedUrl(
            urls=("https://93.184.216.34:8443/path?q=1",), host_header="hook.example:8443", hostname="hook.example"
        )
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


@pytest.mark.parametrize(
    ("failing", "posted_to", "outcome"),
    (
        (
            {"[2606:2800:220:1:248:1893:25c8:1946]"},
            ["2606:2800:220:1:248:1893:25c8:1946", "93.184.216.34"],
            Ok(CallbackResponse(status=204, body=b"")),
        ),
        (
            {"[2606:2800:220:1:248:1893:25c8:1946]", "93.184.216.34"},
            ["2606:2800:220:1:248:1893:25c8:1946", "93.184.216.34"],
            Error("connection_refused"),
        ),
        (set(), ["2606:2800:220:1:248:1893:25c8:1946"], Ok(CallbackResponse(status=204, body=b""))),
    ),
)
@pytest.mark.asyncio
async def test_an_unreachable_address_falls_through_to_the_next_resolved_one(
    failing: set[str], posted_to: list[str], outcome: object
) -> None:
    attempts: Final[list[str]] = []  # mutable-ok: records each connection attempt

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request.url.host)
        host: Final = f"[{request.url.host}]" if ":" in request.url.host else request.url.host
        if host in failing:
            raise httpx.ConnectError("refused")
        return httpx.Response(204)

    response: Final = await post_callback(
        "https://hook.example/in",
        b"{}",
        {},
        resolve=_resolver("2606:2800:220:1:248:1893:25c8:1946", "93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946"),  # pyright: ignore[reportArgumentType]  # test resolver
        transport=httpx.MockTransport(handler),
    )
    assert (attempts, response) == (posted_to, outcome)


@pytest.mark.parametrize(
    "error", (httpx.ConnectError("[SSL: CERTIFICATE_VERIFY_FAILED]"), httpx.RemoteProtocolError("reset mid-body"))
)
@pytest.mark.asyncio
async def test_a_failure_after_connecting_never_retries_another_address(error: httpx.HTTPError) -> None:
    attempts: Final[list[str]] = []  # mutable-ok: records each connection attempt

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request.url.host)
        raise error

    await post_callback(
        "https://hook.example/in",
        b"{}",
        {},
        resolve=_resolver("93.184.216.34", "93.184.216.35"),  # pyright: ignore[reportArgumentType]  # test resolver
        transport=httpx.MockTransport(handler),
    )
    assert attempts == ["93.184.216.34"]


@pytest.mark.parametrize("url", ("https://hook.example:notaport/x", "https://hook.example:99999/x"))
@pytest.mark.asyncio
async def test_a_malformed_port_is_refused_without_resolving(url: str) -> None:
    assert await pin(url, _resolver("93.184.216.34")) == Error("connection_refused")  # pyright: ignore[reportArgumentType]  # test resolver


@pytest.mark.asyncio
async def test_a_connect_timeout_falls_through_to_the_next_address() -> None:
    attempts: Final[list[str]] = []  # mutable-ok: records each connection attempt

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request.url.host)
        if request.url.host == "93.184.216.34":
            raise httpx.ConnectTimeout("slow handshake")
        return httpx.Response(204)

    response: Final = await post_callback(
        "https://hook.example/in",
        b"{}",
        {},
        resolve=_resolver("93.184.216.34", "93.184.216.35"),  # pyright: ignore[reportArgumentType]  # test resolver
        transport=httpx.MockTransport(handler),
    )
    assert (attempts, response) == (["93.184.216.34", "93.184.216.35"], Ok(CallbackResponse(status=204, body=b"")))


@pytest.mark.asyncio
async def test_a_callback_that_never_finishes_hits_the_overall_deadline() -> None:
    async def slow_resolver(_host: str, _port: int) -> tuple[str, ...]:
        await anyio.sleep(3600)
        return ("93.184.216.34",)

    response: Final = await post_callback(
        "https://hook.example/in",
        b"{}",
        {},
        resolve=slow_resolver,
        deadline=0.01,  # pyright: ignore[reportArgumentType]  # test resolver
    )
    assert response == Error("timeout")


class _Streamed(httpx.AsyncByteStream):
    """A body that arrives over the wire in chunks, as a real socket delivers it."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for start in range(0, len(self._body), 4096):
            yield self._body[start : start + 4096]


@pytest.mark.asyncio
async def test_a_compressed_callback_body_is_capped_on_the_wire_and_never_inflated() -> None:
    import gzip

    inflated: Final = b"0" * (10 * 1024 * 1024)
    sent_encodings: Final[list[str | None]] = []  # mutable-ok: records what the gateway asked for

    def handler(request: httpx.Request) -> httpx.Response:
        sent_encodings.append(request.headers.get("accept-encoding"))
        return httpx.Response(200, stream=_Streamed(gzip.compress(inflated)), headers={"content-encoding": "gzip"})

    response: Final = await post_callback(
        "https://hook.example/in",
        b"{}",
        {},
        resolve=_resolver("93.184.216.34"),  # pyright: ignore[reportArgumentType]  # test resolver
        transport=httpx.MockTransport(handler),
    )
    assert sent_encodings == ["identity"]
    assert isinstance(response, Ok) and len(response.ok.body) <= 64 * 1024
    assert response.ok.body[:2] == b"\x1f\x8b"
