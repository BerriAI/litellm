"""
Gateway-level allowlist of MCP client applications (``general_settings.mcp_allowed_clients``).

Each entry pairs an admin-chosen ``alias`` (shown in the dashboard and logs) with the ``value`` that
identifies the client. Only the value is compared, exactly and case-sensitively.
A caller that authenticated with a JWT is identified by the claim named in
``litellm_jwtauth.mcp_client_id_jwt_field``, a value asserted by the identity provider.
Every other caller is identified by the header named in ``general_settings.mcp_client_id_header``,
which the client picks itself, so that source is a policy control rather than a security boundary.
While the allowlist is set, a caller with no usable identity source is rejected.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

from pydantic import TypeAdapter, ValidationError
from typing_extensions import ReadOnly, TypedDict

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.dot_notation_indexing import get_nested_value
from litellm.types.mcp import MCPAllowedClient

MCP_ALLOWED_CLIENTS_SETTING: Final = "mcp_allowed_clients"
MCP_CLIENT_ID_HEADER_SETTING: Final = "mcp_client_id_header"
MCP_CLIENT_ID_JWT_FIELD_SETTING: Final = "mcp_client_id_jwt_field"
_JWT_AUTH_SETTING: Final = "litellm_jwtauth"

_ALLOWED_CLIENTS_ADAPTER: Final[TypeAdapter[list[MCPAllowedClient]]] = TypeAdapter(list[MCPAllowedClient])
_OPTIONAL_NAME_ADAPTER: Final[TypeAdapter[str | None]] = TypeAdapter(str | None)
_OPTIONAL_MAPPING_ADAPTER: Final[TypeAdapter[dict[str, object] | None]] = TypeAdapter(dict[str, object] | None)
_NOBODY: Final[Mapping[str, str]] = MappingProxyType({})


class MCPClientForbiddenBody(TypedDict):
    error: ReadOnly[Literal["Forbidden"]]
    details: ReadOnly[str]


@dataclass(frozen=True, slots=True)
class MCPClientAllowlist:
    """``aliases_by_value`` maps each admitted identity value to the alias the admin gave it."""

    aliases_by_value: Mapping[str, str]
    jwt_field: str | None
    header: str | None


@dataclass(frozen=True, slots=True)
class MCPClientIdentity:
    client_id: str
    source: Literal["jwt", "header"]
    source_name: str

    @property
    def description(self) -> str:
        return f"'{self.client_id}' (from {'JWT claim' if self.source == 'jwt' else 'header'} '{self.source_name}')"


@dataclass(frozen=True, slots=True)
class MCPClientRejection:
    details: str

    @property
    def response_body(self) -> MCPClientForbiddenBody:
        body: Final[MCPClientForbiddenBody] = {"error": "Forbidden", "details": self.details}
        return body


def _unidentified_rejection(reason: str) -> MCPClientRejection:
    return MCPClientRejection(
        details=f"{reason} This gateway only admits client applications listed in {MCP_ALLOWED_CLIENTS_SETTING}."
    )


def parse_allowed_mcp_clients(raw_setting: object) -> Mapping[str, str] | None:
    """Value-to-alias mapping; None when the setting is absent (not enforced). A malformed setting admits nobody."""
    if raw_setting is None:
        return None
    try:
        clients: Final = _ALLOWED_CLIENTS_ADAPTER.validate_python(raw_setting)
    except ValidationError:
        verbose_logger.warning(
            "%s is not a list of {alias, value} entries (%r); rejecting every MCP client until it is fixed",
            MCP_ALLOWED_CLIENTS_SETTING,
            raw_setting,
        )
        return _NOBODY
    return MappingProxyType({client.value: client.alias for client in clients})


def _parse_optional_name(setting_name: str, raw_setting: object) -> str | None:
    try:
        name: Final = _OPTIONAL_NAME_ADAPTER.validate_python(raw_setting)
    except ValidationError:
        verbose_logger.warning("%s is not a string (%r); ignoring it", setting_name, raw_setting)
        return None
    return name or None


def _jwt_field_from_general_settings(general_settings: Mapping[str, object]) -> str | None:
    try:
        jwt_auth: Final = _OPTIONAL_MAPPING_ADAPTER.validate_python(general_settings.get(_JWT_AUTH_SETTING))
    except ValidationError:
        return None
    if jwt_auth is None:
        return None
    return _parse_optional_name(
        f"{_JWT_AUTH_SETTING}.{MCP_CLIENT_ID_JWT_FIELD_SETTING}", jwt_auth.get(MCP_CLIENT_ID_JWT_FIELD_SETTING)
    )


def load_mcp_client_allowlist(general_settings: Mapping[str, object]) -> MCPClientAllowlist | None:
    """None when ``mcp_allowed_clients`` is unset, which admits every client."""
    allowed_clients: Final = parse_allowed_mcp_clients(general_settings.get(MCP_ALLOWED_CLIENTS_SETTING))
    if allowed_clients is None:
        return None
    header: Final = _parse_optional_name(
        MCP_CLIENT_ID_HEADER_SETTING, general_settings.get(MCP_CLIENT_ID_HEADER_SETTING)
    )
    return MCPClientAllowlist(
        aliases_by_value=allowed_clients,
        jwt_field=_jwt_field_from_general_settings(general_settings),
        header=header.lower() if header is not None else None,
    )


def resolve_mcp_client_identity(
    allowlist: MCPClientAllowlist,
    jwt_claims: Mapping[str, object] | None,
    headers: Mapping[str, str],
) -> MCPClientIdentity | MCPClientRejection:
    """A JWT caller is identified by its configured claim alone, so a header can never override the IdP."""
    if jwt_claims is not None and allowlist.jwt_field is not None:
        claim: Final[object] = get_nested_value(data=jwt_claims, key_path=allowlist.jwt_field)
        if isinstance(claim, str) and claim:
            return MCPClientIdentity(client_id=claim, source="jwt", source_name=allowlist.jwt_field)
        return _unidentified_rejection(
            f"The JWT presented has no '{allowlist.jwt_field}' claim naming the client application."
        )
    if allowlist.header is None:
        configured: Final = (
            f"litellm_jwtauth.{MCP_CLIENT_ID_JWT_FIELD_SETTING} for JWT callers or {MCP_CLIENT_ID_HEADER_SETTING}"
        )
        return _unidentified_rejection(
            f"No client identity source is configured for this request; set {configured} in general_settings."
        )
    header_value: Final = headers.get(allowlist.header)
    if header_value:
        return MCPClientIdentity(client_id=header_value, source="header", source_name=allowlist.header)
    return _unidentified_rejection(f"The request has no '{allowlist.header}' header naming the client application.")


def check_mcp_client_allowed(
    allowlist: MCPClientAllowlist | None,
    jwt_claims: Mapping[str, object] | None,
    headers: Mapping[str, str],
) -> MCPClientRejection | None:
    if allowlist is None:
        return None
    identity: Final = resolve_mcp_client_identity(allowlist, jwt_claims, headers)
    if isinstance(identity, MCPClientRejection):
        return identity
    alias: Final = allowlist.aliases_by_value.get(identity.client_id)
    if alias is None:
        return MCPClientRejection(
            details=f"MCP client {identity.description} is not listed in this gateway's {MCP_ALLOWED_CLIENTS_SETTING}."
        )
    verbose_logger.debug("Admitted MCP client '%s' identified as %s", alias, identity.description)
    return None
