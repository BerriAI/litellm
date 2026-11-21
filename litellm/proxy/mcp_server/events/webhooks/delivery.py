import asyncio
import socket
import ssl
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Final
from urllib.parse import urlparse, urlunparse

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
_MAX_RESPONSE_BYTES: Final = 65536

Resolver = Callable[[str, int], Awaitable[tuple[str, ...]]]


@dataclass(frozen=True, slots=True)
class PinnedUrl:
    url: str
    host_header: str
    hostname: str


@dataclass(frozen=True, slots=True)
class CallbackResponse:
    status: int
    body: bytes


async def _resolve(hostname: str, port: int) -> tuple[str, ...]:
    entries: Final = await asyncio.get_running_loop().getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
    return tuple(_sockaddr_host(entry[4]) for entry in entries)


def is_https_callback(url: str) -> bool:
    parsed: Final = urlparse(url)
    return parsed.scheme == "https" and bool(parsed.hostname) and not parsed.fragment


async def pin(url: str, resolve: Resolver = _resolve) -> Result[PinnedUrl, CallbackFailure]:
    parsed: Final = urlparse(url)
    hostname: Final = parsed.hostname
    if parsed.scheme != "https" or not hostname:
        return Error("connection_refused")
    port: Final = parsed.port or 443
    try:
        addresses: Final = await resolve(hostname, port)
    except (OSError, UnicodeError):
        return Error("connection_refused")
    if not addresses:
        return Error("connection_refused")
    if not _is_host_allowlisted(hostname, port) and any(_is_blocked_ip(address) for address in addresses):
        return Error("connection_refused")
    address: Final = addresses[0]
    ip_host: Final = f"[{address}]" if ":" in address else address
    netloc: Final = f"{ip_host}:{parsed.port}" if parsed.port is not None else ip_host
    return Ok(
        PinnedUrl(
            url=urlunparse((parsed.scheme, netloc, parsed.path or "/", parsed.params, parsed.query, "")),
            host_header=_format_host_header(hostname, port, 443),
            hostname=hostname,
        )
    )


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
) -> Result[CallbackResponse, CallbackFailure]:
    pinned: Final = await pin(url, resolve)
    if isinstance(pinned, Error):
        return pinned
    try:
        async with httpx.AsyncClient(
            verify=_verify_setting(),
            follow_redirects=False,
            timeout=DELIVERY_TIMEOUT_SECONDS,
            trust_env=False,
            transport=transport,
        ) as client:
            async with client.stream(
                "POST",
                pinned.ok.url,
                content=body,
                headers={**headers, "host": pinned.ok.host_header},
                extensions={"sni_hostname": pinned.ok.hostname},
            ) as response:
                received: Final = await _read_capped(response)
                return Ok(CallbackResponse(status=response.status_code, body=received))
    except httpx.TimeoutException:
        return Error("timeout")
    except httpx.ConnectError as error:
        return Error(
            "tls_error" if isinstance(error.__cause__, ssl.SSLError) or "SSL" in str(error) else "connection_refused"
        )
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
