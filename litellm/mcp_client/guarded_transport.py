import socket
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Final, TypeAlias

import anyio
import httpcore2
import httpx2

from litellm.litellm_core_utils.url_utils import is_blocked_ip, is_url_destination_allowed_by_host
from litellm.llms.custom_httpx.http_handler import get_ssl_configuration

Resolver: TypeAlias = Callable[[str, int], Awaitable[Sequence[str]]]
SocketOption: TypeAlias = tuple[int, int, int] | tuple[int, int, bytes | bytearray] | tuple[int, int, None, int]


async def resolve_host(host: str, port: int) -> tuple[str, ...]:
    addresses: Final = await anyio.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return tuple(str(sockaddr[0]) for *_, sockaddr in addresses)


def _anyio_backend() -> httpcore2.AsyncNetworkBackend:
    backend: Final[object] = httpcore2.AnyIOBackend()
    return backend if isinstance(backend, httpcore2.AsyncNetworkBackend) else httpcore2.AsyncNetworkBackend()


class GuardedNetworkBackend(httpcore2.AsyncNetworkBackend):
    """Resolves each connection's host once, refuses it when any address is private, loopback, link-local or a cloud
    metadata address, and connects to the checked address so a second lookup cannot swap it. TLS still verifies the
    certificate against the hostname, since the connection pool starts TLS with the origin's host."""

    def __init__(self, resolve: Resolver = resolve_host, connect: httpcore2.AsyncNetworkBackend | None = None) -> None:
        self._resolve: Final = resolve
        self._connect: Final = connect or _anyio_backend()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[SocketOption] | None = None,
    ) -> httpcore2.AsyncNetworkStream:
        addresses: Final = tuple(await self._resolve(host, port))
        if not addresses or any(map(is_blocked_ip, addresses)):
            raise httpcore2.ConnectError(
                f"LiteLLM refused to connect to {host}, which resolves to a private, loopback, link-local or "
                "metadata address"
            )
        return await self._connect.connect_tcp(
            addresses[0], port, timeout=timeout, local_address=local_address, socket_options=socket_options
        )

    async def connect_unix_socket(
        self, path: str, timeout: float | None = None, socket_options: Iterable[SocketOption] | None = None
    ) -> httpcore2.AsyncNetworkStream:
        raise httpcore2.ConnectError("LiteLLM does not reach hosted MCP servers over a unix socket")

    async def sleep(self, seconds: float) -> None:
        await self._connect.sleep(seconds)


@dataclass(frozen=True, slots=True)
class CallerHeaders:
    server_url: str
    names: frozenset[str]


def _origin(url: httpx2.URL) -> tuple[str, str, int | None]:
    return url.scheme, url.host.lower().rstrip("."), url.port


def _without_headers(request: httpx2.Request, names: frozenset[str]) -> httpx2.Request:
    return httpx2.Request(
        request.method,
        request.url,
        headers=[(name, value) for name, value in request.headers.multi_items() if name.lower() not in names],
        stream=request.stream,
        extensions=request.extensions,
    )


class GuardedTransport(httpx2.AsyncHTTPTransport):
    """Sends a request only to an allowlisted host that serves no registered MCP server, redirects included, and opens
    every connection through a GuardedNetworkBackend. Environment proxies are never used, since the guard must see the
    real destination. A registered server's host is refused so a redirect cannot reach it past the key's MCP access,
    and the caller's headers reach only the origin of the server they were sent for."""

    def __init__(
        self,
        allowed_hosts: Sequence[str],
        network_backend: httpcore2.AsyncNetworkBackend | None = None,
        registered_hosts: frozenset[str] = frozenset(),
        caller_headers: CallerHeaders | None = None,
    ) -> None:
        ssl_context: Final = httpx2.create_ssl_context(verify=get_ssl_configuration(), trust_env=False)
        super().__init__(verify=ssl_context, trust_env=False)
        self._allowed_hosts: Final = list(allowed_hosts)
        self._registered_hosts: Final = registered_hosts
        self._caller_headers: Final = caller_headers
        self._pool = httpcore2.AsyncConnectionPool(
            ssl_context=ssl_context, network_backend=network_backend or GuardedNetworkBackend()
        )

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        if request.url.host.lower().rstrip(".") in self._registered_hosts:
            raise httpx2.ConnectError(
                f"LiteLLM refused to connect to {request.url.host}, which serves a registered MCP server that a "
                "request reaches only through the gateway",
                request=request,
            )
        if not is_url_destination_allowed_by_host(str(request.url), self._allowed_hosts):
            raise httpx2.ConnectError(
                f"LiteLLM refused to connect to {request.url.host}, which gateway_hosted_mcp_allowed_hosts does not "
                "list",
                request=request,
            )
        return await super().handle_async_request(self._for_destination(request))

    def _for_destination(self, request: httpx2.Request) -> httpx2.Request:
        caller_headers: Final = self._caller_headers
        if caller_headers is None or _origin(request.url) == _origin(httpx2.URL(caller_headers.server_url)):
            return request
        return _without_headers(request, caller_headers.names)
