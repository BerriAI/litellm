"""
Gateway-level allowlist of MCP client applications, matched against the
``clientInfo.name`` a client sends in its JSON-RPC ``initialize`` request. The
name is client-supplied, so this is a policy control and not a security boundary.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

from pydantic import BaseModel, TypeAdapter, ValidationError
from typing_extensions import ReadOnly, TypedDict

from litellm._logging import verbose_logger
from litellm.constants import MCP_ALLOWLIST_PEEK_MAX_BYTES

MCP_ALLOWED_CLIENTS_SETTING: Final = "mcp_allowed_clients"

_ALLOWED_CLIENTS_ADAPTER: Final = TypeAdapter(list[str])
_GENERAL_SETTINGS_ADAPTER: Final = TypeAdapter(Mapping[str, object])


class _ClientInfo(BaseModel):
    name: str | None = None


class _InitializeParams(BaseModel):
    clientInfo: _ClientInfo | None = None


class _InitializeRequest(BaseModel):
    params: _InitializeParams | None = None


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


def oversized_unidentified_request_body() -> MCPClientForbiddenBody:
    body: Final[MCPClientForbiddenBody] = {
        "error": "Forbidden",
        "details": (
            f"While {MCP_ALLOWED_CLIENTS_SETTING} is set, this gateway reads at most "
            f"{MCP_ALLOWLIST_PEEK_MAX_BYTES} bytes of an MCP POST to find clientInfo.name before routing it; "
            "this request was larger than that and could not be identified."
        ),
    }
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


def allowed_mcp_clients_from_general_settings(general_settings: object) -> frozenset[str] | None:
    return parse_allowed_mcp_clients(
        _GENERAL_SETTINGS_ADAPTER.validate_python(general_settings).get(MCP_ALLOWED_CLIENTS_SETTING)
    )


def extract_mcp_client_name(body: bytes) -> str | None:
    try:
        request: Final = _InitializeRequest.model_validate_json(body)
    except ValidationError:
        return None
    client_info: Final = request.params.clientInfo if request.params is not None else None
    name: Final = client_info.name if client_info is not None else None
    return name if name else None


def check_mcp_client_allowed(body: bytes, allowed_clients: frozenset[str] | None) -> MCPClientRejection | None:
    if allowed_clients is None:
        return None
    client_name: Final = extract_mcp_client_name(body)
    if client_name is not None and client_name in allowed_clients:
        return None
    return MCPClientRejection(client_name=client_name)
