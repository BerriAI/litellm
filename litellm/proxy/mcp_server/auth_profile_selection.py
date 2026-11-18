from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from contextvars import ContextVar
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NoReturn, Protocol, TypeAlias

from pydantic import TypeAdapter
from typing_extensions import assert_never

from litellm._logging import verbose_proxy_logger
from litellm.types.mcp_server.mcp_server_manager import MCPAuthProfile, MCPServer

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.proxy._types import UserAPIKeyAuth


_RECORDED_METADATA: Final = TypeAdapter(Mapping[str, object])


class AuthProfileGrantSource(Protocol):
    async def __call__(self, user_api_key_auth: "UserAPIKeyAuth | None") -> tuple[tuple[str, ...], ...]: ...


class AuthProfileOwnerSource(Protocol):
    async def __call__(self, profile_ids: AbstractSet[str]) -> Mapping[str, str | None] | None: ...


async def no_auth_profile_grants(user_api_key_auth: "UserAPIKeyAuth | None") -> tuple[tuple[str, ...], ...]:
    return ()


@dataclass(frozen=True, slots=True)
class AmbiguousProfile:
    server: str
    profile_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class UnloadedProfile:
    profile_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class UnreadableGrants:
    pass


ProfileSelection: TypeAlias = MCPAuthProfile | AmbiguousProfile | UnloadedProfile | None
UnusableProfile: TypeAlias = AmbiguousProfile | UnloadedProfile | UnreadableGrants
UNUSABLE_PROFILE: Final = (AmbiguousProfile, UnloadedProfile, UnreadableGrants)


def select_auth_profile(
    server: MCPServer, carriers: Sequence[Sequence[str]], loaded_profile_ids: AbstractSet[str]
) -> ProfileSelection:
    profiles: Final = MappingProxyType({profile.id.lower(): profile for profile in server.additional_auth or ()})
    return next(
        (
            selection
            for carrier in carriers
            if (selection := _carrier_selection(server, carrier, profiles, loaded_profile_ids)) is not None
        ),
        None,
    )


def _carrier_selection(
    server: MCPServer,
    carrier: Sequence[str],
    profiles: Mapping[str, MCPAuthProfile],
    loaded_profile_ids: AbstractSet[str],
) -> ProfileSelection:
    unloaded: Final = tuple(grant for grant in carrier if grant.lower() not in loaded_profile_ids)
    if unloaded:
        return UnloadedProfile(unloaded)
    mine: Final = tuple(
        profiles[grant] for grant in dict.fromkeys(grant.lower() for grant in carrier) if grant in profiles
    )
    if len(mine) > 1:
        return AmbiguousProfile(server.server_name or server.server_id, tuple(profile.id for profile in mine))
    return mine[0] if mine else None


def unusable_profile_detail(unusable: UnusableProfile) -> str:
    match unusable:
        case AmbiguousProfile(server=server, profile_ids=profile_ids):
            return f"the caller is granted several auth profiles of MCP server '{server}': {', '.join(profile_ids)}"
        case UnloadedProfile(profile_ids=profile_ids):
            return f"auth profiles {', '.join(profile_ids)} are not loaded on this worker yet"
        case UnreadableGrants():
            return "the caller's MCP auth profile grants could not be read"
        case _:
            assert_never(unusable)


def raise_unusable_profile(unusable: UnusableProfile) -> NoReturn:
    from litellm.proxy.mcp_server.outbound_credentials.adapter import raise_public
    from litellm.proxy.mcp_server.outbound_credentials.types import CredError

    raise_public(CredError.of_upstream_unavailable(unusable_profile_detail(unusable)))


def without_profile_credential_headers(
    server: MCPServer,
    headers: dict[str, str] | None,  # mutable-ok: the upstream client takes the caller headers as a dict
) -> dict[str, str] | None:  # mutable-ok: same dict type the upstream client takes
    if server.auth_profile_id is None or not headers:
        return headers
    from litellm.proxy.mcp_server.outbound_credentials.adapter import token_header

    occupied: Final = frozenset({"authorization", token_header(server).lower()})
    return {name: value for name, value in headers.items() if name.lower() not in occupied}


class AuthProfileContext(Protocol):
    @property
    def auth_profile_grants(self) -> AuthProfileGrantSource: ...

    @property
    def auth_profile_owners(self) -> AuthProfileOwnerSource: ...

    def loaded_auth_profile_ids(self) -> AbstractSet[str]: ...

    def auth_profiles_unread(self, server_id: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class _GrantRead:
    caller: "UserAPIKeyAuth | None"
    source: AuthProfileGrantSource
    outcome: tuple[tuple[str, ...], ...] | Exception


_REQUEST_GRANT_READ: Final[ContextVar[_GrantRead | None]] = ContextVar("mcp_auth_profile_grant_read", default=None)


async def _request_grants(
    user_api_key_auth: "UserAPIKeyAuth | None", source: AuthProfileGrantSource
) -> tuple[tuple[str, ...], ...] | Exception:
    remembered: Final = _REQUEST_GRANT_READ.get()
    if remembered is not None and remembered.caller is user_api_key_auth and remembered.source is source:
        return remembered.outcome
    outcome: Final = await _read_grants(user_api_key_auth, source)
    _REQUEST_GRANT_READ.set(_GrantRead(user_api_key_auth, source, outcome))
    return outcome


async def _read_grants(
    user_api_key_auth: "UserAPIKeyAuth | None", source: AuthProfileGrantSource
) -> tuple[tuple[str, ...], ...] | Exception:
    try:
        return await source(user_api_key_auth)
    except Exception as error:  # noqa: BLE001  # a server without profiles must not fail on an unreadable grant
        verbose_proxy_logger.warning("could not read the caller's MCP auth profile grants: %s", error)
        return error


async def resolve_effective_server(
    server: MCPServer, user_api_key_auth: "UserAPIKeyAuth | None", context: AuthProfileContext
) -> MCPServer | UnusableProfile:
    if server.auth_profile_id is not None:
        return server
    carriers: Final = await _request_grants(user_api_key_auth, context.auth_profile_grants)
    if isinstance(carriers, Exception) and (server.additional_auth or context.auth_profiles_unread(server.server_id)):
        return UnreadableGrants()
    if isinstance(carriers, Exception) or not any(carriers):
        return server
    selection: Final = await _settled_selection(
        server, carriers, context.loaded_auth_profile_ids(), context.auth_profile_owners
    )
    if isinstance(selection, (AmbiguousProfile, UnloadedProfile)):
        return selection
    return with_auth_profile(server, selection)


async def effective_server_for(
    server: MCPServer, user_api_key_auth: "UserAPIKeyAuth | None", context: AuthProfileContext
) -> MCPServer:
    resolved: Final = await resolve_effective_server(server, user_api_key_auth, context)
    if isinstance(resolved, UNUSABLE_PROFILE):
        raise_unusable_profile(resolved)
    return resolved


async def _settled_selection(
    server: MCPServer,
    carriers: Sequence[Sequence[str]],
    known_profile_ids: AbstractSet[str],
    owners: AuthProfileOwnerSource,
) -> ProfileSelection:
    selection: Final = select_auth_profile(server, carriers, known_profile_ids)
    if not isinstance(selection, UnloadedProfile):
        return selection
    unloaded: Final = frozenset(grant.lower() for grant in selection.profile_ids)
    stored: Final = await owners(unloaded)
    if stored is None or any(stored.get(profile_id) == server.server_id for profile_id in unloaded):
        return selection
    return await _settled_selection(server, carriers, known_profile_ids | unloaded, owners)


def with_auth_profile(server: MCPServer, profile: MCPAuthProfile | None) -> MCPServer:
    if profile is None:
        return server
    return server.model_copy(
        update=MappingProxyType(
            {
                "auth_type": profile.auth_type,
                "oauth2_flow": profile.oauth2_flow,
                "client_id": profile.client_id,
                "client_secret": profile.client_secret,
                "token_url": profile.token_url,
                "configured_token_url": None,
                "scopes": None if profile.scopes is None else list(profile.scopes),
                "token_endpoint_auth_method": profile.token_endpoint_auth_method,
                "authentication_token": profile.authentication_token,
                "authorization_url": None,
                "configured_authorization_url": None,
                "registration_url": None,
                "configured_registration_url": None,
                "audience": None,
                "delegate_auth_to_upstream": False,
                "auth_profile_id": profile.id,
                "static_headers": None,
                "extra_headers": None,
            }
        )
    )


def record_auth_profile(logging_obj: "LiteLLMLoggingObj | None", server: MCPServer) -> None:
    if logging_obj is None or server.auth_profile_id is None:
        return
    details: Final = logging_obj.model_call_details
    recorded: Final = _RECORDED_METADATA.validate_python(details.get("mcp_tool_call_metadata") or {})
    stamped: Final = {**recorded, "mcp_auth_profile": server.auth_profile_id, "mcp_auth_mode": server.auth_type}
    details["mcp_tool_call_metadata"] = stamped
