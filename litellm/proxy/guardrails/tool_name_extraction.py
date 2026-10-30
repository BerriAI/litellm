"""
Extract tool names from request body by route/call type.

Used by auth (check_tools_allowlist) and ToolPolicyGuardrail so tool-format
knowledge lives in one place. Uses guardrail translation handlers where available,
with standalone extractors for generate_content and MCP.
"""

from collections.abc import Callable, Iterator, Mapping
from typing import Final

from pydantic import BaseModel, ConfigDict, StrictBool, TypeAdapter, ValidationError

from litellm.litellm_core_utils.api_route_to_call_types import get_call_types_for_route
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.llms import load_guardrail_translation_mappings
from litellm.llms.base_llm.guardrail_translation.utils import list_field
from litellm.proxy.auth.native_mcp_policy import NativeMCPAccess, NativeMCPDenial, NativeMCPExtraction
from litellm.types.proxy.native_mcp_policy import MCPName, MCPToolScope, NativeMCPServerGrant
from litellm.types.utils import CallTypes

# Call types that have no guardrail translation handler; we use standalone extractors
STANDALONE_EXTRACTORS: Final[dict[str, Callable[[Mapping[str, object]], list[str]]]] = {}


def _extract_generate_content_tool_names(data: Mapping[str, object]) -> list[str]:
    """Google generateContent: tools[].functionDeclarations[].name"""
    names: Final[list[str]] = []
    for tool in list_field(data, "tools"):
        for decl in list_field(tool, "functionDeclarations"):
            if isinstance(decl, dict) and decl.get("name"):
                names.append(str(decl["name"]))
    return names


def _extract_mcp_tool_names(data: Mapping[str, object]) -> list[str]:
    """MCP call_tool: name or mcp_tool_name in body"""
    names: Final[list[str]] = []
    name: Final = data.get("name") or data.get("mcp_tool_name")
    if name:
        names.append(str(name))
    return names


def _register_standalone_extractors() -> None:
    if STANDALONE_EXTRACTORS:
        return
    STANDALONE_EXTRACTORS[CallTypes.generate_content.value] = _extract_generate_content_tool_names
    STANDALONE_EXTRACTORS[CallTypes.agenerate_content.value] = _extract_generate_content_tool_names
    STANDALONE_EXTRACTORS[CallTypes.call_mcp_tool.value] = _extract_mcp_tool_names


# Tool-capable call types (routes that can send tools in the request)
TOOL_CAPABLE_CALL_TYPES: Final = frozenset(
    {
        CallTypes.completion.value,
        CallTypes.acompletion.value,
        CallTypes.responses.value,
        CallTypes.aresponses.value,
        CallTypes.anthropic_messages.value,
        CallTypes.generate_content.value,
        CallTypes.agenerate_content.value,
        CallTypes.call_mcp_tool.value,
    }
)


def extract_request_tool_names(route: str, data: dict) -> list[str]:
    """
    Extract tool names from the request body for the given route.
    Uses guardrail translation handlers when available, else standalone extractors
    for generate_content and MCP. Returns [] for non-tool-capable routes or when
    no tools are present.
    """
    call_types: Final = get_call_types_for_route(route)
    if not call_types:
        return []
    _register_standalone_extractors()
    mappings: Final = load_guardrail_translation_mappings()
    for call_type in call_types:
        if not isinstance(call_type, CallTypes):
            continue
        if call_type.value not in TOOL_CAPABLE_CALL_TYPES:
            continue
        if call_type.value in STANDALONE_EXTRACTORS:
            return STANDALONE_EXTRACTORS[call_type.value](data)
        handler_cls = mappings.get(call_type)
        if handler_cls is not None:
            names = handler_cls().extract_request_tool_names(data)
            if names:
                return names
    return []


_MCP_ENTRIES: Final = TypeAdapter(tuple[Mapping[str, object], ...])
_MCP_NAMES: Final = TypeAdapter(tuple[MCPName, ...])


class _MCPToolConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    enabled: StrictBool = True
    allowed_tools: tuple[MCPName, ...] | None = None


class _MCPToolsetConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    default_config: _MCPToolConfig = _MCPToolConfig()
    configs: Mapping[str, _MCPToolConfig] = {}


def _mcp_entries(data: Mapping[str, object], field: str) -> tuple[Mapping[str, object], ...]:
    raw: Final = data.get(field)
    return () if raw is None else _MCP_ENTRIES.validate_python(raw)


def _mcp_request_tools(data: Mapping[str, object], depth: int = 0) -> Iterator[Mapping[str, object]]:
    if depth > 8:
        raise ValueError("MCP tool namespace nesting is too deep")
    for tool in _mcp_entries(data, "tools"):
        if tool.get("type") == "namespace":
            yield from _mcp_request_tools(tool, depth + 1)
        else:
            yield tool


def _mcp_scope(value: object) -> MCPToolScope:
    if value is None:
        return "*"
    filter_config: Final = as_str_mapping(value)
    if filter_config is not None:
        return _mcp_scope(filter_config.get("tool_names"))
    return _MCP_NAMES.validate_python(value) or "*"


def _mcp_access(name: object, url: object, scope: MCPToolScope) -> NativeMCPAccess:
    grant: Final = NativeMCPServerGrant.model_validate({"name": name, "url": url, "tools": scope})
    return NativeMCPAccess(grant.name, grant.url, grant.tools)


def _mcp_toolset_scope(tool: Mapping[str, object]) -> MCPToolScope:
    config: Final = _MCPToolsetConfig.model_validate(tool)
    if config.default_config.enabled:
        return "*"
    return _MCP_NAMES.validate_python(
        tuple(
            name
            for name, setting in config.configs.items()
            if setting.enabled and "enabled" in setting.model_fields_set
        )
    )


def _openai_mcp_access(tool: Mapping[str, object]) -> NativeMCPAccess:
    url: Final = tool.get("server_url")
    if _is_gateway_mcp_url(url) and isinstance(url, str):
        label: Final = tool.get("server_label")
        return NativeMCPAccess(label if isinstance(label, str) else "", url, (), "gateway")
    return _mcp_access(tool.get("server_label"), url, _mcp_scope(tool.get("allowed_tools")))


def _anthropic_mcp_access(server: Mapping[str, object], tools: tuple[Mapping[str, object], ...]) -> NativeMCPAccess:
    toolsets: Final = tuple(
        tool
        for tool in tools
        if tool.get("type") == "mcp_toolset" and tool.get("mcp_server_name") == server.get("name")
    )
    if len(toolsets) > 1:
        raise ValueError("Multiple MCP toolsets reference the same server")
    if toolsets and server.get("tool_configuration") is not None:
        raise ValueError("MCP toolset and legacy tool_configuration cannot be combined")
    if toolsets:
        return _mcp_access(server.get("name"), server.get("url"), _mcp_toolset_scope(toolsets[0]))
    config: Final = _MCPToolConfig.model_validate(server.get("tool_configuration") or {})
    return _mcp_access(
        server.get("name"), server.get("url"), _mcp_scope(config.allowed_tools) if config.enabled else ()
    )


def _mcp_choice_access(selected: Mapping[str, object], attachments: tuple[NativeMCPAccess, ...]) -> NativeMCPAccess:
    server: Final = next((access for access in attachments if access.server_name == selected.get("server_label")), None)
    if server is None:
        raise ValueError("MCP tool_choice references an unattached server")
    names: Final = () if selected.get("name") is None else _MCP_NAMES.validate_python((selected.get("name"),))
    return NativeMCPAccess(server.server_name, server.server_url, names, server.source)


def _mcp_choice_accesses(
    data: Mapping[str, object], attachments: tuple[NativeMCPAccess, ...]
) -> Iterator[NativeMCPAccess]:
    choice: Final = as_str_mapping(data.get("tool_choice")) or {}
    choices: Final = _mcp_entries(choice, "tools") if choice.get("type") == "allowed_tools" else (choice,)
    for selected in choices:
        if selected.get("type") != "mcp":
            continue
        yield _mcp_choice_access(selected, attachments)


def extract_request_mcp_accesses(data: Mapping[str, object]) -> NativeMCPExtraction:
    try:
        extra: Final = as_str_mapping(data.get("extra_body")) or {}
        tools: Final = (*_mcp_request_tools(data), *_mcp_request_tools(extra))
        servers: Final = (*_mcp_entries(data, "mcp_servers"), *_mcp_entries(extra, "mcp_servers"))
        hosted: Final = tuple(tool for tool in tools if tool.get("type") == "mcp")
        accesses: Final = (
            *(_openai_mcp_access(tool) for tool in hosted),
            *(_anthropic_mcp_access(server, tools) for server in servers),
        )
        names: Final = tuple(access.server_name for access in accesses if access.source == "remote")
        if len(set(names)) != len(names):
            return NativeMCPDenial("Duplicate native MCP server names are not allowed")
        orphaned: Final = tuple(
            tool for tool in tools if tool.get("type") == "mcp_toolset" and tool.get("mcp_server_name") not in names
        )
        if orphaned:
            return NativeMCPDenial("MCP toolset references a server missing from mcp_servers")
        return (*accesses, *_mcp_choice_accesses(data, accesses), *_mcp_choice_accesses(extra, accesses))
    except (ValidationError, ValueError):
        return NativeMCPDenial("Cannot validate the native MCP server identity or requested tool scope")


def _is_gateway_mcp_url(url: object) -> bool:
    from litellm.responses.mcp.tool_precheck import LITELLM_PROXY_MCP_SERVER_URL

    return isinstance(url, str) and url.startswith(LITELLM_PROXY_MCP_SERVER_URL)


def _response_mcp_access(
    item: Mapping[str, object], request_accesses: tuple[NativeMCPAccess, ...]
) -> NativeMCPExtraction:
    kind: Final = item.get("type")
    if kind not in ("mcp_call", "mcp_tool_use", "mcp_approval_request", "mcp_list_tools"):
        return ()
    name: Final = item.get("server_name") if kind == "mcp_tool_use" else item.get("server_label")
    matching: Final = tuple(access for access in request_accesses if access.server_name == name)
    if not matching:
        return NativeMCPDenial("Native MCP response references a server absent from the validated request")
    try:
        tool_names: Final = () if kind == "mcp_list_tools" else _MCP_NAMES.validate_python((item.get("name"),))
        return (NativeMCPAccess(matching[0].server_name, matching[0].server_url, tool_names, matching[0].source),)
    except ValidationError:
        return NativeMCPDenial("Native MCP response has no readable tool name")


def _provider_mcp_items(data: Mapping[str, object]) -> Iterator[Mapping[str, object]]:
    fields: Final = as_str_mapping(data.get("provider_specific_fields")) or {}
    yield from _mcp_entries(fields, "mcp_tool_uses")
    yield from _mcp_entries(fields, "tool_results")


def _response_items(data: Mapping[str, object]) -> Iterator[Mapping[str, object]]:
    yield from _mcp_entries(data, "output")
    yield from _mcp_entries(data, "content")
    yield from _provider_mcp_items(data)
    for choice in _mcp_entries(data, "choices"):
        for field in ("message", "delta"):
            message: Final = as_str_mapping(choice.get(field)) or {}
            yield from _provider_mcp_items(message)


def response_mcp_items(response: object) -> tuple[Mapping[str, object], ...] | NativeMCPDenial:
    data: Final = as_str_mapping(response.model_dump() if isinstance(response, BaseModel) else response)
    if data is None:
        return NativeMCPDenial("Cannot read the response for native MCP policy validation")
    try:
        return tuple(_response_items(data))
    except ValidationError:
        return NativeMCPDenial("Cannot read native MCP response items")


def extract_response_mcp_accesses(
    response: object, request_accesses: tuple[NativeMCPAccess, ...]
) -> NativeMCPExtraction:
    items: Final = response_mcp_items(response)
    if isinstance(items, NativeMCPDenial):
        return items
    call_ids: Final = tuple(item.get("id") for item in items if item.get("type") == "mcp_tool_use")
    if any(
        not isinstance(item.get("tool_use_id"), str) or item.get("tool_use_id") not in call_ids
        for item in items
        if item.get("type") == "mcp_tool_result"
    ):
        return NativeMCPDenial("Native MCP result references a call absent from the response")
    results: Final = tuple(_response_mcp_access(item, request_accesses) for item in items)
    failure: Final = next((result for result in results if isinstance(result, NativeMCPDenial)), None)
    if failure is not None:
        return failure
    from itertools import chain

    return tuple(chain.from_iterable(result for result in results if isinstance(result, tuple)))
