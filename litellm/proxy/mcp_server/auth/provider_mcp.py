from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, NoReturn

from pydantic import TypeAdapter, ValidationError

from litellm.proxy._types import ProxyErrorTypes, ProxyException, UserAPIKeyAuth
from litellm.proxy.mcp_server.auth.user_api_key_auth_mcp import MCPRequestHandler
from litellm.proxy.mcp_server.catalog import catalog_operation, global_manager
from litellm.responses.mcp.litellm_proxy_mcp_handler import LiteLLM_Proxy_MCP_Handler
from litellm.types.mcp_server.provider_registration import canonical_provider_mcp_url

_OBJECT: Final = TypeAdapter(dict[str, object])
_OBJECTS: Final = TypeAdapter(tuple[dict[str, object], ...])
_NAMES: Final = TypeAdapter(tuple[str, ...])


@dataclass(frozen=True, slots=True)
class ProviderMCPAttachment:
    url: str
    tools: frozenset[str]


@dataclass(frozen=True, slots=True)
class MCPAdmissionError:
    message: str


def _names(value: object) -> frozenset[str] | MCPAdmissionError:
    try:
        names: Final = _NAMES.validate_python(value)
    except ValidationError:
        return MCPAdmissionError("Provider MCP attachments require an explicit allowed_tools list")
    if any(not name.strip() or any(char in name for char in "*?[]") for name in names):
        return MCPAdmissionError("Provider MCP tool names must be exact names without wildcards")
    return frozenset(names)


def _attachment(url: object, tools: frozenset[str] | MCPAdmissionError) -> ProviderMCPAttachment | MCPAdmissionError:
    if isinstance(tools, MCPAdmissionError):
        return tools
    try:
        canonical: Final = canonical_provider_mcp_url(url)
    except ValueError:
        return MCPAdmissionError("Provider MCP attachments require a registered HTTPS server_url")
    return ProviderMCPAttachment(canonical, tools)


def _openai_attachment(tool: Mapping[str, object]) -> ProviderMCPAttachment | MCPAdmissionError:
    if tool.get("connector_id") is not None:
        return MCPAdmissionError("Provider MCP connector IDs are not supported by URL registrations")
    allowed: Final = tool.get("allowed_tools")
    names: Final = _OBJECT.validate_python(allowed).get("tool_names") if isinstance(allowed, Mapping) else allowed
    return _attachment(tool.get("server_url"), _names(names))


def _anthropic_attachment(
    server: Mapping[str, object], toolsets: tuple[Mapping[str, object], ...]
) -> ProviderMCPAttachment | MCPAdmissionError:
    matching: Final = tuple(tool for tool in toolsets if tool.get("mcp_server_name") == server.get("name"))
    if server.get("type") != "url" or not isinstance(server.get("name"), str):
        return MCPAdmissionError("Provider MCP servers must have type=url and a name")
    if len(matching) > 1 or (matching and server.get("tool_configuration") is not None):
        return MCPAdmissionError("Provider MCP servers require one unambiguous tool configuration")
    if matching:
        toolset: Final = matching[0]
        default: Final = _OBJECT.validate_python(toolset.get("default_config") or {})
        if default.get("enabled") is not False:
            return MCPAdmissionError("Provider MCP toolsets require default_config.enabled=false")
        configs: Final = TypeAdapter(dict[str, dict[str, object]]).validate_python(toolset.get("configs") or {})
        if any("enabled" in config and type(config["enabled"]) is not bool for config in configs.values()):
            return MCPAdmissionError("Provider MCP tool enabled settings must be booleans")
        enabled: Final = tuple(name for name, config in configs.items() if config.get("enabled") is True)
        return _attachment(server.get("url"), _names(enabled))
    config: Final = _OBJECT.validate_python(server.get("tool_configuration") or {})
    if config.get("enabled") is False:
        return _attachment(server.get("url"), frozenset())
    names: Final = _names(config.get("allowed_tools"))
    if names == frozenset():
        return MCPAdmissionError("Enabled legacy MCP connectors require a nonempty allowed_tools list")
    return _attachment(server.get("url"), names)


async def _attachments(
    body: Mapping[str, object], *, provider_bound: bool
) -> tuple[ProviderMCPAttachment | MCPAdmissionError, ...]:
    tools: Final = _OBJECTS.validate_python(body.get("tools") or ())
    mcp_tools: Final = tuple(tool for tool in tools if tool.get("type") == "mcp")
    gateway: Final = (
        (False,) * len(mcp_tools)
        if provider_bound
        else await LiteLLM_Proxy_MCP_Handler.routes_through_gateway(mcp_tools)
    )
    openai: Final = tuple(_openai_attachment(tool) for tool, routed in zip(mcp_tools, gateway) if not routed)
    servers: Final = _OBJECTS.validate_python(body.get("mcp_servers") or ())
    toolsets: Final = tuple(tool for tool in tools if tool.get("type") == "mcp_toolset")
    server_names: Final = tuple(server.get("name") for server in servers)
    if len(set(server_names)) != len(server_names) or any(
        tool.get("mcp_server_name") not in server_names for tool in toolsets
    ):
        return (MCPAdmissionError("Provider MCP server names must be unique and toolsets must reference them"),)
    return (*openai, *(_anthropic_attachment(server, toolsets) for server in servers))


def _raise_denied(error: MCPAdmissionError) -> NoReturn:
    raise ProxyException(
        message=error.message,
        type=ProxyErrorTypes.tool_access_denied,
        param="tools",
        code=403,
    )


@catalog_operation(global_manager)
async def authorize_provider_mcp(
    body: Mapping[str, object], auth: UserAPIKeyAuth, *, provider_bound: bool = False
) -> None:
    try:
        attachments: Final = await _attachments(body, provider_bound=provider_bound)
    except (ValidationError, TypeError):
        _raise_denied(MCPAdmissionError("Invalid provider MCP tool configuration"))
    if not attachments:
        return
    manager: Final = global_manager()
    registrations: Final = {server.url: server for server in manager.catalog.provider_registrations().values()}
    allowed_servers: Final = frozenset(await manager.get_allowed_mcp_servers(auth))
    for attachment in attachments:
        if isinstance(attachment, MCPAdmissionError):
            _raise_denied(attachment)
        registration = registrations.get(attachment.url)
        if registration is None or registration.server_id not in allowed_servers:
            _raise_denied(MCPAdmissionError("Provider MCP server is not registered or not permitted for this key/team"))
        permitted = await MCPRequestHandler.get_allowed_tools_for_server(registration.server_id, auth)
        ceiling = frozenset(registration.allowed_tools or ())
        allowed = ceiling if permitted is None else ceiling.intersection(permitted)
        if not attachment.tools <= allowed:
            _raise_denied(MCPAdmissionError("Provider MCP tools exceed the registered server or key/team permissions"))
