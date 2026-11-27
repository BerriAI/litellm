import asyncio
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, NamedTuple, NoReturn, Protocol, TypeAlias

from fastapi import HTTPException
from pydantic import SecretStr
from typing_extensions import assert_never

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.core_helpers import single_log_line
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.mcp_server.auth.auth_profile_grants import caller_auth_profile_grants
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


@dataclass(frozen=True, slots=True)
class RejectedToken:
    server_id: str
    profile_id: str
    is_outage: bool = False


ProfileRefusal: TypeAlias = (
    MissingProfile | IncompleteProfile | UnreadableProfiles | UnsupportedProfileHost | AmbiguousProfile
)
UnusableProfile: TypeAlias = ProfileRefusal | RejectedToken | UnreadableGrants
_REFUSALS: Final = (MissingProfile, IncompleteProfile, UnreadableProfiles, UnsupportedProfileHost, AmbiguousProfile)
_UNUSABLE: Final = (*_REFUSALS, RejectedToken, UnreadableGrants)


GrantOutcome: TypeAlias = tuple[ProfileGrants, ...] | UnreadableGrants


class GrantSource(Protocol):
    async def __call__(self, caller: UserAPIKeyAuth | None) -> GrantOutcome: ...


class Resolution(NamedTuple):
    server: MCPServer
    outcome: MCPServer | UnusableProfile


@dataclass(frozen=True, slots=True)
class ServerResolutions:
    outcomes: tuple[Resolution, ...]

    @property
    def usable(self) -> tuple[MCPServer, ...]:
        return tuple(r.outcome for r in self.outcomes if not isinstance(r.outcome, _UNUSABLE))

    @property
    def left_out(self) -> tuple[tuple[MCPServer, UnusableProfile], ...]:
        return tuple((r.server, r.outcome) for r in self.outcomes if isinstance(r.outcome, _UNUSABLE))


def _plain(secret: SecretStr | None) -> str | None:
    return secret.get_secret_value() if secret else None


def _has_credential(server_id: str, profile: MCPAuthProfile) -> bool:
    return profile_credentials_error(server_id, (profile,)) is None


def _has_profiles(server: MCPServer) -> bool:
    return bool(server.additional_auth) or server.auth_profiles_unreadable


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
        case RejectedToken(server_id=server_id, profile_id=profile_id):
            return f"the auth profile '{profile_id}' of MCP server '{server_id}' could not obtain a token"
        case UnreadableGrants():
            return "the caller's MCP auth profile grants could not be read"
        case _:  # pragma: no cover - exhaustiveness guard, unreachable while the union holds
            assert_never(unusable)  # pragma: no cover


def unusable_profile_status(unusable: UnusableProfile) -> int:
    """The status the default credential answers with for the same cause: 500 when the stored
    credential is wrong, 503 when a store or the token endpoint it depends on is down."""
    match unusable:
        case UnreadableProfiles() | UnreadableGrants() | RejectedToken(is_outage=True):
            return 503
        case _:
            return 500


class ProfileUnavailable(HTTPException):
    def __init__(self, unusable: UnusableProfile) -> None:
        super().__init__(status_code=unusable_profile_status(unusable), detail=unusable_profile_detail(unusable))


def raise_unusable_profile(unusable: UnusableProfile) -> NoReturn:
    verbose_proxy_logger.warning("MCP auth profile unavailable: %s", single_log_line(unusable_profile_detail(unusable)))
    raise ProfileUnavailable(unusable)


@dataclass(frozen=True, slots=True)
class _PendingRead:
    caller: UserAPIKeyAuth | None  # held so the ids in the memo key stay unique while the scope is open
    source: GrantSource
    task: asyncio.Task[GrantOutcome]


def _consume_task_exception(task: asyncio.Task[GrantOutcome]) -> None:
    if not task.cancelled():
        task.exception()


class _GrantMemo:
    def __init__(self) -> None:
        self.reads: dict[tuple[int, int], _PendingRead] = {}  # mutable-ok: the memo fills as reads start
        self.is_open = True

    async def read(self, caller: UserAPIKeyAuth | None, source: GrantSource) -> GrantOutcome:
        if not self.is_open:
            return await source(caller)
        key: Final = (id(caller), id(source))
        held: Final = self.reads.get(key)
        if held is not None:
            return await asyncio.shield(held.task)
        task: Final = asyncio.create_task(source(caller))
        task.add_done_callback(_consume_task_exception)
        started: Final = _PendingRead(caller, source, task)
        self.reads[key] = started
        return await asyncio.shield(started.task)

    def close(self) -> None:
        self.is_open = False
        self.reads.clear()


_MEMO: Final[ContextVar[_GrantMemo | None]] = ContextVar("mcp_auth_profile_grant_memo", default=None)


@contextmanager
def shared_grant_memo() -> Generator[None]:
    outer: Final = _MEMO.get()
    if outer is not None and outer.is_open:
        yield
        return
    memo: Final = _GrantMemo()
    _MEMO.set(memo)
    try:
        yield
    finally:
        memo.close()
        if _MEMO.get() is memo:
            _MEMO.set(outer)


async def _read_grants(caller: UserAPIKeyAuth | None, source: GrantSource) -> GrantOutcome:
    memo: Final = _MEMO.get()
    return await memo.read(caller, source) if memo is not None else await source(caller)


async def resolve_effective_server(
    server: MCPServer, caller: UserAPIKeyAuth | None, source: GrantSource = caller_auth_profile_grants
) -> MCPServer | UnusableProfile:
    if server.uses_auth_profile or not _has_profiles(server):
        return server
    carriers: Final = await _read_grants(caller, source)
    if isinstance(carriers, UnreadableGrants):
        return carriers
    selection: Final = select_auth_profile(server, carriers)
    if isinstance(selection, _REFUSALS):
        return selection
    effective: Final = with_auth_profile(server, selection)
    if not isinstance(selection, ClientCredentialsAuthProfile):
        return effective
    token_failure: Final = await _token_failure(effective, caller)
    if token_failure is None:
        return effective
    return RejectedToken(server.server_id, selection.id, is_outage=token_failure == "upstream_unavailable")


async def _token_failure(server: MCPServer, caller: UserAPIKeyAuth | None) -> str | None:
    from litellm.proxy.mcp_server.mcp_server_manager import (  # noqa: PLC0415  # lazy: the manager imports this module
        global_mcp_server_manager,
    )

    return await global_mcp_server_manager.profile_token_failure(server, caller)


async def resolve_usable_server(
    server: MCPServer, caller: UserAPIKeyAuth | None, source: GrantSource = caller_auth_profile_grants
) -> MCPServer:
    resolved: Final = await resolve_effective_server(server, caller, source)
    if isinstance(resolved, _UNUSABLE):
        raise_unusable_profile(resolved)
    return resolved


async def resolve_effective_servers(
    servers: Sequence[MCPServer], caller: UserAPIKeyAuth | None, source: GrantSource = caller_auth_profile_grants
) -> ServerResolutions:
    with shared_grant_memo():
        resolutions: Final = ServerResolutions(
            tuple([Resolution(server, await resolve_effective_server(server, caller, source)) for server in servers])
        )
    left_out: Final = resolutions.left_out
    if left_out:
        verbose_proxy_logger.warning(
            "MCP servers left out of this request: %s",
            "; ".join(f"{server.server_id}: {unusable_profile_detail(unusable)}" for server, unusable in left_out),
        )
    return resolutions
