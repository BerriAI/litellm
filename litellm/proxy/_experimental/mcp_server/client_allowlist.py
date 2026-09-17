"""
Gateway-level allowlist of MCP client applications, matched against the
``clientInfo.name`` a client sends in its JSON-RPC ``initialize`` request. The
name is client-supplied, so this is a policy control and not a security boundary.
"""

import json
from dataclasses import dataclass
from typing import Final, Literal

from pydantic import TypeAdapter, ValidationError
from typing_extensions import ReadOnly, TypedDict

from litellm._logging import verbose_logger

MCP_ALLOWED_CLIENTS_SETTING: Final = "mcp_allowed_clients"

_ALLOWED_CLIENTS_ADAPTER: Final = TypeAdapter(list[str])


class MCPClientForbiddenBody(TypedDict):
    error: ReadOnly[Literal["Forbidden"]]
    details: ReadOnly[str]


@dataclass(frozen=True, slots=True)
class MCPClientRejection:
    client_name: str | None

    @property
    def details(self) -> str:
        if self.client_name is None:
            return (
                "MCP initialize request did not identify the client application (clientInfo.name). "
                f"This gateway only admits clients listed in {MCP_ALLOWED_CLIENTS_SETTING}."
            )
        return f"MCP client '{self.client_name}' is not listed in this gateway's {MCP_ALLOWED_CLIENTS_SETTING}."

    @property
    def response_body(self) -> MCPClientForbiddenBody:
        body: Final[MCPClientForbiddenBody] = {"error": "Forbidden", "details": self.details}
        return body


def parse_allowed_mcp_clients(raw_setting: object) -> frozenset[str] | None:
    """None when the setting is absent (not enforced). A malformed setting admits nobody."""
    if raw_setting is None:
        return None
    try:
        return frozenset(_ALLOWED_CLIENTS_ADAPTER.validate_python(raw_setting))
    except ValidationError:
        verbose_logger.warning(
            "%s is not a list of client names (%r); rejecting every MCP client until it is fixed",
            MCP_ALLOWED_CLIENTS_SETTING,
            raw_setting,
        )
        return frozenset()


def extract_mcp_client_name(body: bytes) -> str | None:
    try:
        data: Final = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    params: Final = data.get("params") if isinstance(data, dict) else None
    client_info: Final = params.get("clientInfo") if isinstance(params, dict) else None
    name: Final = client_info.get("name") if isinstance(client_info, dict) else None
    return name if isinstance(name, str) and name else None


def check_mcp_client_allowed(body: bytes, allowed_clients: frozenset[str] | None) -> MCPClientRejection | None:
    """None when the initialize is admitted, otherwise the rejection to send back as a 403."""
    if allowed_clients is None:
        return None
    client_name: Final = extract_mcp_client_name(body)
    if client_name is not None and client_name in allowed_clients:
        return None
    return MCPClientRejection(client_name=client_name)
