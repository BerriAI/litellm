from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Final

import httpcore2
import httpx2
import pytest

from litellm.mcp_client.guarded_transport import (
    CallerHeaders,
    GuardedNetworkBackend,
    GuardedTransport,
    Resolver,
    SocketOption,
)

PUBLIC_ADDRESS: Final = "93.184.215.14"


def _http_response(status: int, headers: Mapping[str, str], body: bytes = b"") -> tuple[bytes, ...]:
    header_lines: Final = "".join(f"{name}: {value}\r\n" for name, value in headers.items())
    return (f"HTTP/1.1 {status} X\r\n{header_lines}Content-Length: {len(body)}\r\n\r\n".encode(), body)


class _RecordingBackend(httpcore2.AsyncMockBackend):
    def __init__(self, buffer: Sequence[bytes]) -> None:
        super().__init__(list(buffer))
        self.hosts: tuple[str, ...] = ()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[SocketOption] | None = None,
    ) -> httpcore2.AsyncNetworkStream:
        self.hosts = (*self.hosts, host)
        return await super().connect_tcp(host, port, timeout, local_address, socket_options)


def _resolver(table: Mapping[str, tuple[str, ...]]) -> Resolver:
    async def resolve(host: str, port: int) -> tuple[str, ...]:
        return table[host]

    return resolve


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "addresses",
    [
        ("10.0.0.5",),
        ("127.0.0.1",),
        ("169.254.169.254",),
        ("192.168.1.10",),
        ("::1",),
        ("fd00::1",),
        ("::ffff:127.0.0.1",),
        (PUBLIC_ADDRESS, "10.0.0.5"),
        (),
    ],
    ids=["private", "loopback", "metadata", "lan", "ipv6_loopback", "ipv6_private", "mapped_loopback", "mixed", "none"],
)
async def test_a_host_resolving_to_a_blocked_address_is_refused_before_any_connection(addresses: tuple[str, ...]):
    connector: Final = _RecordingBackend(_http_response(200, {}))
    backend: Final = GuardedNetworkBackend(_resolver({"mcp.example.com": addresses}), connect=connector)

    with pytest.raises(httpcore2.ConnectError):
        await backend.connect_tcp("mcp.example.com", 443)

    assert connector.hosts == ()


@pytest.mark.asyncio
async def test_a_public_host_is_connected_at_the_address_that_was_checked():
    connector: Final = _RecordingBackend(_http_response(200, {}))
    backend: Final = GuardedNetworkBackend(_resolver({"mcp.example.com": (PUBLIC_ADDRESS,)}), connect=connector)

    await backend.connect_tcp("mcp.example.com", 443)

    assert connector.hosts == (PUBLIC_ADDRESS,)


class _UnreachableFirstBackend(_RecordingBackend):
    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[SocketOption] | None = None,
    ) -> httpcore2.AsyncNetworkStream:
        if not self.hosts:
            self.hosts = (host,)
            raise httpcore2.ConnectError("network is unreachable")
        return await super().connect_tcp(host, port, timeout, local_address, socket_options)


@pytest.mark.asyncio
async def test_an_unreachable_first_address_falls_back_to_the_next_checked_address():
    connector: Final = _UnreachableFirstBackend(_http_response(200, {}))
    backend: Final = GuardedNetworkBackend(
        _resolver({"mcp.example.com": ("2606:2800:21f:cb07:6820:80da:af6b:8b2c", PUBLIC_ADDRESS)}), connect=connector
    )

    await backend.connect_tcp("mcp.example.com", 443)

    assert connector.hosts == ("2606:2800:21f:cb07:6820:80da:af6b:8b2c", PUBLIC_ADDRESS)


def _guarded_client(
    allowed_hosts: tuple[str, ...],
    table: Mapping[str, tuple[str, ...]],
    connector: _RecordingBackend,
    registered_hosts: frozenset[str] = frozenset(),
) -> httpx2.AsyncClient:
    transport: Final = GuardedTransport(
        allowed_hosts,
        network_backend=GuardedNetworkBackend(_resolver(table), connect=connector),
        registered_hosts=registered_hosts,
    )
    return httpx2.AsyncClient(transport=transport, follow_redirects=True)


@pytest.mark.asyncio
async def test_an_allowlisted_public_host_is_reached():
    connector: Final = _RecordingBackend(_http_response(200, {}, b"ok"))
    client: Final = _guarded_client(("mcp.example.com",), {"mcp.example.com": (PUBLIC_ADDRESS,)}, connector)

    response: Final = await client.get("https://mcp.example.com/mcp")

    assert response.text == "ok"
    assert connector.hosts == (PUBLIC_ADDRESS,)


@pytest.mark.asyncio
async def test_a_host_missing_from_the_allowlist_opens_no_connection():
    connector: Final = _RecordingBackend(_http_response(200, {}))
    client: Final = _guarded_client(("mcp.example.com",), {"other.example.com": (PUBLIC_ADDRESS,)}, connector)

    with pytest.raises(httpx2.ConnectError):
        await client.get("https://other.example.com/mcp")

    assert connector.hosts == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "location",
    [
        "http://169.254.169.254/latest/meta-data",
        "https://internal.example.com/mcp",
        "https://elsewhere.example.com/mcp",
    ],
    ids=["metadata_ip", "allowlisted_host_on_a_private_address", "host_missing_from_the_allowlist"],
)
async def test_a_redirect_to_a_refused_destination_never_connects_there(location: str):
    connector: Final = _RecordingBackend(_http_response(302, {"Location": location}))
    client: Final = _guarded_client(
        ("mcp.example.com", "internal.example.com"),
        {
            "mcp.example.com": (PUBLIC_ADDRESS,),
            "internal.example.com": ("10.0.0.7",),
            "elsewhere.example.com": (PUBLIC_ADDRESS,),
        },
        connector,
    )

    with pytest.raises(httpx2.ConnectError):
        await client.get("https://mcp.example.com/mcp")

    assert connector.hosts == (PUBLIC_ADDRESS,)


@pytest.mark.asyncio
async def test_a_redirect_to_an_allowlisted_host_serving_a_registered_mcp_server_never_connects_there():
    connector: Final = _RecordingBackend(_http_response(302, {"Location": "https://registered.example.com/mcp"}))
    client: Final = _guarded_client(
        ("mcp.example.com", "registered.example.com"),
        {"mcp.example.com": (PUBLIC_ADDRESS,), "registered.example.com": ("93.184.215.15",)},
        connector,
        registered_hosts=frozenset({"registered.example.com"}),
    )

    with pytest.raises(httpx2.ConnectError, match="registered MCP server"):
        await client.get("https://mcp.example.com/mcp")

    assert connector.hosts == (PUBLIC_ADDRESS,)


class _WriteRecordingStream(httpcore2.AsyncMockStream):
    def __init__(self, buffer: list[bytes], record: Callable[[bytes], None]) -> None:
        super().__init__(buffer)
        self._record = record

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        self._record(buffer)


class _PerAddressBackend(httpcore2.AsyncMockBackend):
    def __init__(self, responses: Mapping[str, tuple[bytes, ...]]) -> None:
        super().__init__([])
        self._responses = responses
        self.sent: tuple[tuple[str, bytes], ...] = ()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[SocketOption] | None = None,
    ) -> httpcore2.AsyncNetworkStream:
        def record(buffer: bytes) -> None:
            self.sent = (*self.sent, (host, buffer))

        return _WriteRecordingStream([chunk for chunk in self._responses[host] if chunk], record)

    def sent_to(self, address: str) -> bytes:
        return b"".join(buffer for host, buffer in self.sent if host == address).lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("location", "reaches_second_address"),
    [("https://other.example.com/mcp", True), ("https://mcp.example.com/mcp/", False)],
    ids=["another_origin", "same_origin"],
)
async def test_the_callers_headers_follow_a_redirect_only_within_the_origin_they_were_sent_for(
    location: str, reaches_second_address: bool
):
    other_address: Final = "93.184.215.16"
    backend: Final = _PerAddressBackend(
        {
            PUBLIC_ADDRESS: (*_http_response(307, {"Location": location}), *_http_response(200, {}, b"ok")),
            other_address: _http_response(200, {}, b"ok"),
        }
    )
    transport: Final = GuardedTransport(
        ("mcp.example.com", "other.example.com"),
        network_backend=GuardedNetworkBackend(
            _resolver({"mcp.example.com": (PUBLIC_ADDRESS,), "other.example.com": (other_address,)}), connect=backend
        ),
        caller_headers=CallerHeaders(server_url="https://mcp.example.com/mcp", names=frozenset({"x-api-key"})),
    )
    client: Final = httpx2.AsyncClient(transport=transport, follow_redirects=True, headers={"X-Api-Key": "caller-key"})

    response: Final = await client.get("https://mcp.example.com/mcp")

    assert response.text == "ok"
    assert backend.sent_to(PUBLIC_ADDRESS).count(b"x-api-key: caller-key") == (1 if reaches_second_address else 2)
    assert (b"x-api-key" in backend.sent_to(other_address), bool(backend.sent_to(other_address))) == (
        False,
        reaches_second_address,
    )
