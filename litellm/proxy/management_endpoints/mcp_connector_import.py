"""
Convert Anthropic MCP connector definitions into LiteLLM MCP server create requests.

Two interchange shapes are accepted:
- the ``mcpServers`` mapping used by Claude Desktop / Claude Code config files
- the ``mcp_servers`` array used by the Anthropic Messages API MCP connector
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError

from litellm.proxy._types import MCPApprovalStatus, NewMCPServerRequest
from litellm.types.mcp import MCPAuth, MCPCredentials, MCPTransport


class MCPConnectorEntry(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str | None = None
    type: str | None = None
    url: str | None = None
    authorization_token: str | None = Field(
        default=None, validation_alias=AliasChoices("authorization_token", "authorizationToken")
    )
    headers: Mapping[str, str] | None = None
    command: str | None = None
    args: tuple[str, ...] = Field(default_factory=tuple)
    env: Mapping[str, str] = Field(default_factory=dict)
    description: str | None = None


class MCPConnectorImportRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    mcp_servers: Mapping[str, MCPConnectorEntry] | tuple[MCPConnectorEntry, ...] = Field(
        validation_alias=AliasChoices("mcp_servers", "mcpServers")
    )


@dataclass(frozen=True, slots=True)
class ConvertedConnector:
    name: str
    request: NewMCPServerRequest


@dataclass(frozen=True, slots=True)
class ConnectorConversionError:
    name: str
    error: str


class MCPConnectorImportResult(BaseModel):
    name: str
    server_id: str
    alias: str


class MCPConnectorImportSkipped(BaseModel):
    name: str
    reason: str


class MCPConnectorImportFailure(BaseModel):
    name: str
    error: str


class MCPConnectorImportResponse(BaseModel):
    imported: tuple[MCPConnectorImportResult, ...]
    skipped: tuple[MCPConnectorImportSkipped, ...]
    errors: tuple[MCPConnectorImportFailure, ...]


_INVALID_SERVER_NAME_CHARS: Final = re.compile(r"[^A-Za-z0-9_]")


def sanitize_connector_name(name: str) -> str:
    sanitized: Final = re.sub(r"_+", "_", _INVALID_SERVER_NAME_CHARS.sub("_", name.strip())).strip("_")
    return sanitized


_SSE_TYPES: Final = frozenset({"sse"})
_URL_TYPES: Final = frozenset({"url", "http", "streamable_http", "streamable-http", "sse", ""})


def _convert_entry(name: str, entry: MCPConnectorEntry) -> ConvertedConnector | ConnectorConversionError:
    sanitized_name: Final = sanitize_connector_name(name)
    if not sanitized_name:
        return ConnectorConversionError(name=name, error="Connector name is empty after sanitization.")

    if entry.url and entry.command:
        return ConnectorConversionError(name=name, error="Connector cannot have both a url and a command.")

    if entry.command:
        try:
            stdio_request: Final = NewMCPServerRequest(
                server_name=sanitized_name,
                alias=sanitized_name,
                description=entry.description,
                approval_status=MCPApprovalStatus.active,
                transport=MCPTransport.stdio,
                command=entry.command,
                args=entry.args,
                env=entry.env,
            )
        except ValidationError as e:
            return ConnectorConversionError(name=name, error=_first_validation_message(e))
        return ConvertedConnector(name=name, request=stdio_request)

    if not entry.url:
        return ConnectorConversionError(name=name, error="Connector must have either a url or a command.")

    entry_type: Final = (entry.type or "").lower()
    if entry_type not in _URL_TYPES:
        return ConnectorConversionError(name=name, error=f"Unsupported connector type '{entry.type}'.")

    transport: Final = MCPTransport.sse if entry_type in _SSE_TYPES else MCPTransport.http
    credentials: Final = _bearer_credentials(entry.authorization_token)
    try:
        remote_request: Final = NewMCPServerRequest(
            server_name=sanitized_name,
            alias=sanitized_name,
            description=entry.description,
            approval_status=MCPApprovalStatus.active,
            transport=transport,
            url=entry.url,
            auth_type=MCPAuth.bearer_token if entry.authorization_token else MCPAuth.none,
            credentials=credentials,
            static_headers=entry.headers,
        )
    except ValidationError as e:
        return ConnectorConversionError(name=name, error=_first_validation_message(e))
    return ConvertedConnector(name=name, request=remote_request)


def _bearer_credentials(token: str | None) -> MCPCredentials | None:
    if not token:
        return None
    credentials: Final[MCPCredentials] = {"auth_value": token}
    return credentials


def _first_validation_message(error: ValidationError) -> str:
    messages: Final = tuple(str(detail.get("msg", "")) for detail in error.errors())
    return messages[0] if messages else str(error)


def convert_connector_entries(
    payload: MCPConnectorImportRequest,
) -> tuple[ConvertedConnector | ConnectorConversionError, ...]:
    servers: Final = payload.mcp_servers
    if isinstance(servers, Mapping):
        return tuple(_convert_entry(name, entry) for name, entry in servers.items())
    return tuple(
        _convert_entry(entry.name or "", entry) if entry.name else _named_entry_error(index, entry)
        for index, entry in enumerate(servers)
    )


def _named_entry_error(index: int, entry: MCPConnectorEntry) -> ConnectorConversionError:
    return ConnectorConversionError(
        name=entry.url or f"entry {index}",
        error="Connector entries in list form must have a name.",
    )
