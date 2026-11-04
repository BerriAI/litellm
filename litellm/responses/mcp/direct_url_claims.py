from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from types import MappingProxyType
from typing import Final, Protocol, TypeAlias
from urllib.parse import urlsplit

from mcp.types import CallToolRequestParams, CallToolResult, InputRequiredResult
from mcp.types import Tool as MCPTool
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from litellm.exceptions import APIError
from litellm.litellm_core_utils.core_helpers import drops_unsupported_params
from litellm.litellm_core_utils.tool_fields import hosted_mcp_approval_error
from litellm.litellm_core_utils.url_utils import is_url_destination_allowed_by_host
from litellm.proxy.mcp_server.utils import add_server_prefix_to_name
from litellm.responses.mcp.tool_precheck import LITELLM_PROXY_MCP_SERVER_URL, proxy_path_mcp_name
from litellm.types.mcp import MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer

_DEFAULT_PORTS: Final = MappingProxyType({"http": 80, "https": 443})
_HEADERS: Final = TypeAdapter(Mapping[str, str])
_GATEWAY_OWNED_HEADERS: Final = frozenset(
    {
        "connection",
        "content-length",
        "forwarded",
        "host",
        "keep-alive",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "x-real-ip",
    }
)


class _ToolNames(BaseModel):
    model_config = ConfigDict(frozen=True)
    tool_names: tuple[str, ...]


_ALLOWED_TOOLS: Final[TypeAdapter[tuple[str, ...] | _ToolNames]] = TypeAdapter(tuple[str, ...] | _ToolNames)


@dataclass(frozen=True, slots=True)
class DirectMCPServer:
    server_label: str
    server_url: str
    headers: Mapping[str, str]
    allowed_tools: frozenset[str] | None


@dataclass(frozen=True, slots=True)
class HostedMCPClaims:
    """What the gateway runs for a request: `tools` with each registered match pointed at its gateway name and each
    direct claim removed, the direct servers it reaches itself, and the caller's label for every matched server name."""

    tools: tuple[Mapping[str, object], ...]
    direct: tuple[DirectMCPServer, ...]
    server_labels: Mapping[str, str]


class DirectMCPClient(Protocol):
    async def list_tools(self, raise_on_error: bool = False) -> Sequence[MCPTool]: ...

    async def call_tool(
        self, call_tool_request_params: CallToolRequestParams
    ) -> CallToolResult | InputRequiredResult: ...


DirectClientFactory: TypeAlias = Callable[[DirectMCPServer, Sequence[str]], DirectMCPClient]


@dataclass(frozen=True, slots=True)
class DirectTool:
    server: DirectMCPServer
    tool: MCPTool
    client: DirectMCPClient


@dataclass(frozen=True, slots=True)
class DirectListing:
    tools: tuple[MCPTool, ...]
    by_name: Mapping[str, DirectTool]
    server_tools: Mapping[str, tuple[MCPTool, ...]]


@dataclass(frozen=True, slots=True)
class _Unclaimed:
    tool: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _RegisteredClaim:
    tool: Mapping[str, object]
    server: MCPServer
    label: str


@dataclass(frozen=True, slots=True)
class _DirectClaim:
    tool: Mapping[str, object]
    server: DirectMCPServer


_Claim: TypeAlias = _Unclaimed | _RegisteredClaim | _DirectClaim


def normalized_mcp_url(url: str) -> str | None:
    """The form two MCP server URLs share when they name the same endpoint: lowercase scheme and host, no default port,
    fragment or trailing slash, and the path and query as sent. None for a URL the gateway never connects to."""
    parts: Final = urlsplit(url.strip())
    scheme: Final = parts.scheme.lower()
    if scheme not in _DEFAULT_PORTS or not parts.hostname or parts.username is not None or parts.password is not None:
        return None
    try:
        port: Final = parts.port
    except ValueError:
        return None
    host: Final = parts.hostname.rstrip(".")
    bracketed: Final = f"[{host}]" if ":" in host else host
    netloc: Final = bracketed if port in (None, _DEFAULT_PORTS[scheme]) else f"{bracketed}:{port}"
    query: Final = f"?{parts.query}" if parts.query else ""
    return f"{scheme}://{netloc}{parts.path.rstrip('/')}{query}"


def hosted_mcp_allowed_origins(entries: Sequence[str]) -> tuple[str, ...]:
    """The admin's hosted MCP allowlist with each entry that names no scheme read as https, so a bare host admits
    https on port 443 and `host:port` admits https on that port, never plain http or another port."""
    return tuple(entry.strip() if "://" in entry else f"https://{entry.strip()}" for entry in entries)


def unregistered_mcp_server(server: DirectMCPServer) -> MCPServer:
    """The server record the gateway's pre-call checks and guardrails see for a URL the request named. Its id is
    the URL, which no registered server's id can equal, so no key or team grant for a registered server applies."""
    return MCPServer(
        server_id=server.server_url,
        name=server.server_label,
        server_name=server.server_label,
        url=server.server_url,
        transport=MCPTransport.http,
    )


def server_names(server: MCPServer) -> tuple[str, ...]:
    return tuple(name for name in (server.alias, server.server_name, server.name, server.server_id) if name)


def gateway_name(server: MCPServer) -> str:
    return server.alias or server.server_name or server.server_id


def _text_field(tool: Mapping[str, object], key: str) -> str | None:
    value: Final = tool.get(key)
    return value if isinstance(value, str) and value else None


def _is_caller_header(name: str) -> bool:
    lowered: Final = name.lower()
    return lowered not in _GATEWAY_OWNED_HEADERS and not lowered.startswith("x-forwarded-")


def _caller_headers(tool: Mapping[str, object]) -> Mapping[str, str]:
    try:
        headers: Final = _HEADERS.validate_python(tool.get("headers") or {})
    except ValidationError:
        return MappingProxyType({})
    sent: Final = {name: value for name, value in headers.items() if _is_caller_header(name)}
    token: Final = _text_field(tool, "authorization")
    sets_authorization: Final = any(name.lower() == "authorization" for name in sent)
    bearer: Final = {} if token is None or sets_authorization else {"Authorization": f"Bearer {token}"}
    return MappingProxyType({**sent, **bearer})


def allowed_tool_names(tool: Mapping[str, object]) -> frozenset[str] | None:
    allowed: Final = tool.get("allowed_tools")
    if allowed is None:
        return None
    try:
        parsed: Final = _ALLOWED_TOOLS.validate_python(allowed)
    except ValidationError:
        return frozenset()
    return frozenset(parsed.tool_names if isinstance(parsed, _ToolNames) else parsed)


def _registered_matches(url: str, servers: Collection[MCPServer]) -> tuple[MCPServer, ...]:
    return tuple(server for server in servers if server.url and normalized_mcp_url(server.url) == url)


def _url_host(url: str | None) -> str | None:
    normalized: Final = None if url is None else normalized_mcp_url(url)
    return None if normalized is None else urlsplit(normalized).hostname


def _on_registered_host(url: str, servers: Collection[MCPServer]) -> bool:
    return _url_host(url) in {_url_host(server.url) for server in servers}


def _served_proxy_path(tool: Mapping[str, object], servers: Collection[MCPServer]) -> bool:
    name: Final = proxy_path_mcp_name(tool)
    return name is not None and any(name.lower() in map(str.lower, server_names(server)) for server in servers)


def _registered_claim(
    tool: Mapping[str, object], label: str, matches: tuple[MCPServer, ...]
) -> _RegisteredClaim | _Unclaimed:
    named: Final = (
        matches if len(matches) == 1 else tuple(server for server in matches if label in server_names(server))
    )
    return _RegisteredClaim(tool=tool, server=named[0], label=label) if len(named) == 1 else _Unclaimed(tool=tool)


def _names_a_reachable_url(tool: Mapping[str, object]) -> bool:
    server_url: Final = _text_field(tool, "server_url")
    return (
        tool.get("type") == "mcp"
        and server_url is not None
        and not server_url.startswith(LITELLM_PROXY_MCP_SERVER_URL)
        and normalized_mcp_url(server_url) is not None
    )


def direct_mcp_labels(tools: Iterable[Mapping[str, object]]) -> frozenset[str]:
    return frozenset(_text_field(tool, "server_label") or "unlabeled" for tool in tools if _names_a_reachable_url(tool))


def _claim(tool: Mapping[str, object], servers: Collection[MCPServer], allowed_hosts: Sequence[str]) -> _Claim:
    server_url: Final = _text_field(tool, "server_url")
    url: Final = None if server_url is None else normalized_mcp_url(server_url)
    if (
        tool.get("type") != "mcp"
        or server_url is None
        or url is None
        or server_url.startswith(LITELLM_PROXY_MCP_SERVER_URL)
        or _served_proxy_path(tool, servers)
    ):
        return _Unclaimed(tool=tool)
    label: Final = _text_field(tool, "server_label") or "unlabeled"
    matches: Final = _registered_matches(url, servers)
    if matches:
        return _registered_claim(tool, label, matches)
    if _on_registered_host(url, servers) or not is_url_destination_allowed_by_host(server_url, list(allowed_hosts)):
        return _Unclaimed(tool=tool)
    return _DirectClaim(
        tool=tool,
        server=DirectMCPServer(
            server_label=label,
            server_url=server_url,
            headers=_caller_headers(tool),
            allowed_tools=allowed_tool_names(tool),
        ),
    )


def _runs_without_approval(claim: _Claim) -> bool:
    return isinstance(claim, _Unclaimed) or claim.tool.get("require_approval") == "never"


def _gateway_tool(claim: _Claim) -> Mapping[str, object] | None:
    match claim:
        case _Unclaimed(tool=tool):
            return tool
        case _RegisteredClaim(tool=tool, server=server):
            return MappingProxyType(
                {**tool, "server_url": f"{LITELLM_PROXY_MCP_SERVER_URL}/mcp/{gateway_name(server)}"}
            )
        case _DirectClaim():
            return None


def _label_entries(claim: _RegisteredClaim) -> Mapping[str, str]:
    return MappingProxyType(dict.fromkeys(server_names(claim.server), claim.label))


def plan_hosted_mcp_claims(
    tools: Sequence[Mapping[str, object]],
    servers: Collection[MCPServer],
    allowed_hosts: Sequence[str],
    drop_params: object,
    model: str,
    route: str,
) -> HostedMCPClaims | None:
    """Claim each hosted MCP tool whose URL a registered server serves or whose host the admin allowlisted. A claimed
    tool that needs approval answers 400, or under drop_params is left for the hosted MCP strip to drop. None when
    nothing is claimed, so the request keeps its current path."""
    allowed_origins: Final = hosted_mcp_allowed_origins(allowed_hosts)
    claims: Final = tuple(_claim(tool, servers, allowed_origins) for tool in tools)
    needs_approval: Final = tuple(claim for claim in claims if not _runs_without_approval(claim))
    if needs_approval and not drops_unsupported_params(drop_params):
        raise hosted_mcp_approval_error(
            tuple(_text_field(claim.tool, "server_label") or "unlabeled" for claim in needs_approval),
            model=model,
            route=route,
        )
    runnable: Final = tuple(claim if _runs_without_approval(claim) else _Unclaimed(claim.tool) for claim in claims)
    if all(isinstance(claim, _Unclaimed) for claim in runnable):
        return None
    registered: Final = tuple(claim for claim in runnable if isinstance(claim, _RegisteredClaim))
    return HostedMCPClaims(
        tools=tuple(tool for tool in map(_gateway_tool, runnable) if tool is not None),
        direct=tuple(claim.server for claim in runnable if isinstance(claim, _DirectClaim)),
        server_labels=MappingProxyType(
            dict(chain.from_iterable(entries.items() for entries in map(_label_entries, registered)))
        ),
    )


def guarded_mcp_client(server: DirectMCPServer, allowed_hosts: Sequence[str]) -> DirectMCPClient:
    from litellm.mcp_client.client import MCPClient
    from litellm.mcp_client.guarded_transport import GuardedTransport

    return MCPClient(
        server_url=server.server_url,
        transport_type=MCPTransport.http,
        extra_headers=dict(server.headers),
        upstream_transport=GuardedTransport(hosted_mcp_allowed_origins(allowed_hosts)),
    )


def _listing_error(server: DirectMCPServer, error: Exception) -> APIError:
    return APIError(
        status_code=424,
        message=f"LiteLLM could not list the tools of the hosted MCP server `{server.server_label}`: {error}",
        llm_provider="mcp",
        model=server.server_label,
    )


async def _server_tools(server: DirectMCPServer, client: DirectMCPClient) -> tuple[MCPTool, ...]:
    try:
        listed: Final = await client.list_tools(raise_on_error=True)
    except Exception as error:
        raise _listing_error(server, error) from error
    return tuple(tool for tool in listed if server.allowed_tools is None or tool.name in server.allowed_tools)


def direct_tool_name(server_label: str, tool_name: str) -> str:
    return add_server_prefix_to_name(tool_name, server_label)


def _named_tools(
    server: DirectMCPServer, client: DirectMCPClient, tools: tuple[MCPTool, ...]
) -> tuple[tuple[str, MCPTool, DirectTool], ...]:
    return tuple(
        (direct_tool_name(server.server_label, tool.name), tool, DirectTool(server, tool, client)) for tool in tools
    )


async def list_direct_tools(
    servers: Sequence[DirectMCPServer], client_for: Callable[[DirectMCPServer], DirectMCPClient]
) -> DirectListing:
    clients: Final = tuple((server, client_for(server)) for server in servers)
    listed: Final = tuple([(server, client, await _server_tools(server, client)) for server, client in clients])
    direct_tools: Final = tuple(chain.from_iterable(_named_tools(*entry) for entry in listed))
    return DirectListing(
        tools=tuple(tool.model_copy(update={"name": name}) for name, tool, _ in direct_tools),
        by_name=MappingProxyType({name: direct_tool for name, _, direct_tool in direct_tools}),
        server_tools=MappingProxyType({server.server_label: tools for server, _, tools in listed}),
    )
