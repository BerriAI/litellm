from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, NoReturn, TypeAlias

from fastapi import HTTPException
from pydantic import SecretStr
from typing_extensions import assert_never

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.core_helpers import single_log_line
from litellm.proxy.mcp_server.auth_profiles import profile_credentials_error
from litellm.types.mcp import MCPTransport
from litellm.types.mcp_server.auth_profile import (
    RESERVED_PROFILE_ID,
    ClientCredentialsAuthProfile,
    MCPAuthProfile,
    StaticAuthProfile,
    profile_id_key,
)
from litellm.types.mcp_server.auth_profile_grants import ProfileGrants, UnreadableGrants
from litellm.types.mcp_server.mcp_server_manager import MCPServer

PROFILE_UNAVAILABLE: Final = "The MCP server's selected auth profile is unavailable."

_CLEARED_BY_PROFILE: Final = MappingProxyType(
    {
        "configured_token_url": None,
        "authorization_url": None,
        "configured_authorization_url": None,
        "registration_url": None,
        "configured_registration_url": None,
        "audience": None,
        "upstream_resource": None,
        "delegate_auth_to_upstream": False,
        "static_headers": None,
        "extra_headers": None,
        "env_vars": None,
        "oauth_identity_binding": None,
        "aws_access_key_id": None,
        "aws_secret_access_key": None,
        "aws_session_token": None,
        "aws_region_name": None,
        "aws_service_name": None,
        "aws_role_name": None,
        "aws_session_name": None,
        "client_private_key": None,
        "client_private_key_id": None,
        "token_exchange_endpoint": None,
        "id_jag_resource_token_endpoint": None,
        "id_jag_resource_client_id": None,
        "id_jag_resource_client_secret": None,
    }
)


class ProfileUnavailable(HTTPException):
    def __init__(self) -> None:
        super().__init__(status_code=503, detail=PROFILE_UNAVAILABLE)


@dataclass(frozen=True, slots=True)
class MissingProfile:
    server_id: str
    profile_id: str


@dataclass(frozen=True, slots=True)
class IncompleteProfile:
    server_id: str
    profile_id: str


@dataclass(frozen=True, slots=True)
class UnreadableProfiles:
    server_id: str
    profile_id: str


@dataclass(frozen=True, slots=True)
class UnsupportedProfileHost:
    server_id: str
    profile_id: str


@dataclass(frozen=True, slots=True)
class AmbiguousProfile:
    server_id: str
    profile_id: str


ProfileRefusal: TypeAlias = (
    MissingProfile | IncompleteProfile | UnreadableProfiles | UnsupportedProfileHost | AmbiguousProfile
)
UnusableProfile: TypeAlias = ProfileRefusal | UnreadableGrants


def _plain(secret: SecretStr | None) -> str | None:
    return secret.get_secret_value() if secret else None


def _has_credential(server_id: str, profile: MCPAuthProfile) -> bool:
    return profile_credentials_error(server_id, (profile,)) is None


def _takes_profiles(server: MCPServer) -> bool:
    return server.transport != MCPTransport.stdio and not server.is_byok and not server.spec_path


def select_auth_profile(server: MCPServer, carriers: Sequence[ProfileGrants]) -> MCPAuthProfile | ProfileRefusal | None:
    granted: Final = next((carrier[server.server_id] for carrier in carriers if server.server_id in carrier), None)
    if granted is None or profile_id_key(granted) == RESERVED_PROFILE_ID:
        return None
    if not _takes_profiles(server):
        return UnsupportedProfileHost(server.server_id, granted)
    if server.auth_profiles_unreadable:
        return UnreadableProfiles(server.server_id, granted)
    matches: Final = tuple(p for p in server.additional_auth or () if profile_id_key(p.id) == profile_id_key(granted))
    match matches:
        case ():
            return MissingProfile(server.server_id, granted)
        case (profile,):
            return (
                profile if _has_credential(server.server_id, profile) else IncompleteProfile(server.server_id, granted)
            )
        case _:
            return AmbiguousProfile(server.server_id, granted)


def _credential_fields(profile: MCPAuthProfile) -> Mapping[str, object]:
    match profile:
        case StaticAuthProfile():
            return {
                "auth_type": profile.auth_type,
                "authentication_token": _plain(profile.authentication_token),
                "oauth2_flow": None,
                "client_id": None,
                "client_secret": None,
                "token_url": None,
                "scopes": None,
                "token_endpoint_auth_method": None,
            }
        case ClientCredentialsAuthProfile():
            return {
                "auth_type": profile.auth_type,
                "authentication_token": None,
                "oauth2_flow": profile.oauth2_flow,
                "client_id": _plain(profile.client_id),
                "client_secret": _plain(profile.client_secret),
                "token_url": profile.token_url,
                "scopes": None if profile.scopes is None else list(profile.scopes),
                "token_endpoint_auth_method": profile.token_endpoint_auth_method,
            }
        case _:  # pragma: no cover - exhaustiveness guard, unreachable while the union holds
            assert_never(profile)  # pragma: no cover


def with_auth_profile(server: MCPServer, profile: MCPAuthProfile | None) -> MCPServer:
    if profile is None:
        return server
    return server.model_copy(
        update={**_credential_fields(profile), **_CLEARED_BY_PROFILE, "auth_profile_id": profile.id}
    )


def unusable_profile_detail(unusable: UnusableProfile) -> str:
    match unusable:
        case MissingProfile(server_id=server_id, profile_id=profile_id):
            return f"the auth profile '{profile_id}' granted for MCP server '{server_id}' does not exist"
        case IncompleteProfile(server_id=server_id, profile_id=profile_id):
            return f"the auth profile '{profile_id}' of MCP server '{server_id}' has no usable credential"
        case UnreadableProfiles(server_id=server_id, profile_id=profile_id):
            return f"the auth profiles of MCP server '{server_id}' could not be read to apply '{profile_id}'"
        case UnsupportedProfileHost(server_id=server_id, profile_id=profile_id):
            return f"MCP server '{server_id}' cannot apply the auth profile '{profile_id}' on this kind of host"
        case AmbiguousProfile(server_id=server_id, profile_id=profile_id):
            return f"more than one auth profile of MCP server '{server_id}' matches '{profile_id}'"
        case UnreadableGrants():
            return "the caller's MCP auth profile grants could not be read"
        case _:  # pragma: no cover - exhaustiveness guard, unreachable while the union holds
            assert_never(unusable)  # pragma: no cover


def raise_unusable_profile(unusable: UnusableProfile) -> NoReturn:
    verbose_proxy_logger.warning("MCP auth profile unavailable: %s", single_log_line(unusable_profile_detail(unusable)))
    raise ProfileUnavailable()
