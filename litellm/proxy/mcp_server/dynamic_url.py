from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

import httpx2
import regex
from fastapi import HTTPException
from typing_extensions import assert_never

from litellm.proxy.mcp_server.exceptions import MCPServerListError
from litellm.proxy.mcp_server.faults.list_outcomes import ServerListFault
from litellm.types.mcp_server.mcp_server_manager import MCPDynamicURL, MCPServer

DYNAMIC_URL_HEADER: Final = "x-litellm-mcp-url"
_MAX_HOST_LENGTH: Final = 253
_HOST_MATCH_TIMEOUT_SECONDS: Final = 0.05
_ALLOWED_SCHEMES: Final = frozenset(("http", "https"))


@dataclass(frozen=True, slots=True)
class DynamicURLAccepted:
    url: str
    tag: Literal["accepted"] = "accepted"


@dataclass(frozen=True, slots=True)
class DynamicURLMissing:
    tag: Literal["missing"] = "missing"


@dataclass(frozen=True, slots=True)
class DynamicURLRejected:
    tag: Literal["rejected"] = "rejected"


DynamicURLResult: TypeAlias = DynamicURLAccepted | DynamicURLMissing | DynamicURLRejected


def dynamic_url_header(raw_headers: Mapping[str, str] | None) -> str | None:
    return next(
        (value for name, value in (raw_headers or {}).items() if name.lower() == DYNAMIC_URL_HEADER and value),
        None,
    )


def _host_is_allowed(host: str, policy: MCPDynamicURL) -> bool:
    if not host or len(host) > _MAX_HOST_LENGTH:
        return False
    try:
        return any(
            regex.fullmatch(pattern, host, timeout=_HOST_MATCH_TIMEOUT_SECONDS) is not None
            for pattern in policy.allowed_hosts_regex
        )
    except TimeoutError:
        return False


def _is_allowed_url(value: str, policy: MCPDynamicURL) -> bool:
    if "#" in value or "\\" in value:
        return False
    try:
        url: Final = httpx2.URL(value)
        host: Final = url.raw_host.decode("ascii")
    except (httpx2.InvalidURL, UnicodeDecodeError):
        return False
    return url.scheme in _ALLOWED_SCHEMES and not url.userinfo and _host_is_allowed(host, policy)


def resolve_dynamic_url(policy: MCPDynamicURL, raw_headers: Mapping[str, str] | None) -> DynamicURLResult:
    value: Final = dynamic_url_header(raw_headers)
    if value is None:
        return DynamicURLMissing()
    if not _is_allowed_url(value, policy):
        return DynamicURLRejected()
    return DynamicURLAccepted(url=value)


def server_for_request(server: MCPServer, raw_headers: Mapping[str, str] | None) -> MCPServer:
    """The server with its URL taken from this request when it opted into ``dynamic_url``.

    The registered server is never changed, so two requests naming different tenants cannot cross.
    Error details never echo the header, since the URL may carry a credential."""
    if server.dynamic_url is None:
        return server
    result: Final = resolve_dynamic_url(server.dynamic_url, raw_headers)
    match result:
        case DynamicURLAccepted(url=url):
            return server.model_copy(update={"url": url})
        case DynamicURLMissing():
            raise HTTPException(
                status_code=400,
                detail=f"MCP server '{server.name}' requires the {DYNAMIC_URL_HEADER} header",
            )
        case DynamicURLRejected():
            raise HTTPException(
                status_code=400,
                detail=(
                    f"The {DYNAMIC_URL_HEADER} header is not an allowed URL for MCP server '{server.name}'. "
                    "It must be an http or https URL without credentials or a fragment, "
                    "and its host must match the server's allowed_hosts_regex"
                ),
            )
        case _:
            assert_never(result)


def skips_dynamic_url_listing(server: MCPServer, raw_headers: Mapping[str, str] | None) -> bool:
    """True when a tool listing must return nothing for this server because the request names no URL.

    A header that names a URL outside the policy is a listing fault, never a silent empty list."""
    if server.dynamic_url is None:
        return False
    result: Final = resolve_dynamic_url(server.dynamic_url, raw_headers)
    match result:
        case DynamicURLMissing():
            return True
        case DynamicURLRejected():
            raise MCPServerListError(ServerListFault(tag="invalid_request", status_code=400), server.name)
        case DynamicURLAccepted():
            return False
        case _:
            assert_never(result)
