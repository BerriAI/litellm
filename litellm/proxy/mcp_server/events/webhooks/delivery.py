import asyncio
import socket
import ssl
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Final, TypeAlias
from urllib.parse import ParseResult, urlparse, urlunparse

import anyio
import httpx

from litellm.litellm_core_utils.url_utils import (
    _format_host_header,  # pyright: ignore[reportPrivateUsage]  # the SSRF guard's own Host header rule
    _is_blocked_ip,  # pyright: ignore[reportPrivateUsage]  # the SSRF guard's own IP policy
    _is_host_allowlisted,  # pyright: ignore[reportPrivateUsage]  # honors user_url_allowed_hosts like validate_url
    _sockaddr_host,  # pyright: ignore[reportPrivateUsage]  # the SSRF guard's own address parser
)
from litellm.proxy.mcp_server.outbound_credentials.result import Error, Ok, Result
from litellm.types.mcp_server.events import CallbackFailure

DELIVERY_TIMEOUT_SECONDS: Final = 5.0
CALLBACK_DEADLINE_SECONDS: Final = 15.0
_MAX_RESPONSE_BYTES: Final = 65536
_MAX_ADDRESSES: Final = 3

Resolver: TypeAlias = Callable[[str, int], Awaitable[tuple[str, ...]]]


@dataclass(frozen=True, slots=True)
class PinnedUrl:
    urls: tuple[str, ...]
    host_header: str
    hostname: str

    @property
    def url(self) -> str:
        return self.urls[0]


@dataclass(frozen=True, slots=True)
class CallbackResponse:
    status: int
    body: bytes


async def _resolve(hostname: str, port: int) -> tuple[str, ...]:
    entries: Final = await asyncio.get_running_loop().getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
    return tuple(_sockaddr_host(entry[4]) for entry in entries)


def _port(parsed: ParseResult) -> int | None:
    """The URL's port; ValueError for a port that is not a number in 0..65535."""
    return parsed.port


def is_https_callback(url: str) -> bool:
    try:
        parsed: Final = urlparse(url)
        _port(parsed)
    except ValueError:
        return False
    return parsed.scheme == "https" and bool(parsed.hostname) and not parsed.fragment


async def pin(
    url: str, resolve: Resolver = _resolve, allowlisted: Callable[[str, int], bool] = _is_host_allowlisted
) -> Result[PinnedUrl, CallbackFailure]:
    if not is_https_callback(url):
        return Error("connection_refused")
    parsed: Final = urlparse(url)
    hostname: Final = parsed.hostname or ""
    port: Final = parsed.port or 443
    try:
        addresses: Final = await resolve(hostname, port)
    except (OSError, UnicodeError):
        return Error("connection_refused")
    if not addresses:
        return Error("connection_refused")
    if not allowlisted(hostname, port) and any(_is_blocked_ip(address) for address in addresses):
        return Error("connection_refused")
    distinct: Final = tuple(dict.fromkeys(addresses))[:_MAX_ADDRESSES]
    return Ok(
        PinnedUrl(
            urls=tuple(_pinned_url(parsed.scheme, address, parsed.port, parsed) for address in distinct),
            host_header=_format_host_header(hostname, port, 443),
            hostname=hostname,
        )
    )


def _pinned_url(scheme: str, address: str, port: int | None, parsed: ParseResult) -> str:
    ip_host: Final = f"[{address}]" if ":" in address else address
    netloc: Final = f"{ip_host}:{port}" if port is not None else ip_host
    return urlunparse((scheme, netloc, parsed.path or "/", parsed.params, parsed.query, ""))


def _verify_setting() -> bool | ssl.SSLContext | str:
    import litellm

    setting: Final = getattr(litellm, "ssl_verify", True)
    return setting if isinstance(setting, (bool, ssl.SSLContext, str)) else True


async def post_callback(
    url: str,
    body: bytes,
    headers: Mapping[str, str],
    resolve: Resolver = _resolve,
    transport: httpx.AsyncBaseTransport | None = None,
    deadline: float = CALLBACK_DEADLINE_SECONDS,
) -> Result[CallbackResponse, CallbackFailure]:
    """POST once, bounded end to end: DNS, every connection attempt and a slow body share one deadline."""
    try:
        with anyio.fail_after(deadline):
            return await _post_pinned(url, body, headers, resolve, transport)
    except TimeoutError:
        return Error("timeout")


async def _post_pinned(
    url: str,
    body: bytes,
    headers: Mapping[str, str],
    resolve: Resolver,
    transport: httpx.AsyncBaseTransport | None,
) -> Result[CallbackResponse, CallbackFailure]:
    pinned: Final = await pin(url, resolve)
    if isinstance(pinned, Error):
        return Error(pinned.error)
    async with httpx.AsyncClient(
        verify=_verify_setting(),
        follow_redirects=False,
        timeout=DELIVERY_TIMEOUT_SECONDS,
        trust_env=False,
        transport=transport,
    ) as client:
        for target in pinned.ok.urls:
            outcome = await _post_once(client, target, pinned.ok, body, headers)
            if outcome is not None:
                return outcome
        return Error("connection_refused")


async def _post_once(
    client: httpx.AsyncClient, target: str, pinned: PinnedUrl, body: bytes, headers: Mapping[str, str]
) -> Result[CallbackResponse, CallbackFailure] | None:
    """None when no connection was made, so nothing was sent and the next address may be tried."""
    try:
        async with client.stream(
            "POST",
            target,
            content=body,
            headers={**headers, "host": pinned.host_header},
            extensions={"sni_hostname": pinned.hostname},
        ) as response:
            received: Final = await _read_capped(response)
            return Ok(CallbackResponse(status=response.status_code, body=received))
    except httpx.ConnectTimeout:
        return None
    except httpx.TimeoutException:
        return Error("timeout")
    except httpx.ConnectError as error:
        return Error("tls_error") if isinstance(error.__cause__, ssl.SSLError) or "SSL" in str(error) else None
    except httpx.HTTPError:
        return Error("connection_refused")


async def _read_capped(response: httpx.Response) -> bytes:
    chunks: Final[list[bytes]] = []  # mutable-ok: bounded response buffer
    size = 0  # rebind-ok: running byte count of the capped read
    async for chunk in response.aiter_bytes():
        chunks.append(chunk[: _MAX_RESPONSE_BYTES - size])
        size += len(chunks[-1])
        if size >= _MAX_RESPONSE_BYTES:
            break
    return b"".join(chunks)


def failure_for_status(status: int) -> CallbackFailure | None:
    if 200 <= status < 300:
        return None
    return "http_5xx" if status >= 500 else "http_4xx"
