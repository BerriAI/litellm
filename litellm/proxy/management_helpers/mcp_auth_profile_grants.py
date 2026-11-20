import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, TypeAlias, overload

from fastapi import HTTPException, status
from pydantic import TypeAdapter, ValidationError
from typing_extensions import assert_never

from litellm.types.mcp_server.mcp_server_manager import MCPServer

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


_MAX_GRANT_IDS: Final = 256
_GRANT_IDS: Final = TypeAdapter(tuple[str, ...])
_PERMISSION: Final = TypeAdapter(Mapping[str, object])


@dataclass(frozen=True, slots=True)
class RegisteredProfile:
    profile_id: str
    server_id: str


@dataclass(frozen=True, slots=True)
class UnknownGrant:
    profile_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AmbiguousGrant:
    server_profile_ids: Mapping[str, tuple[str, ...]]


GrantError: TypeAlias = UnknownGrant | AmbiguousGrant


def registered_auth_profiles(servers: Iterable[MCPServer]) -> Mapping[str, RegisteredProfile]:
    return MappingProxyType(dict(chain.from_iterable(map(_registered_profiles_of, servers))))


def _registered_profiles_of(server: MCPServer) -> Iterator[tuple[str, RegisteredProfile]]:
    for profile in server.additional_auth or ():
        yield profile.id.lower(), RegisteredProfile(profile_id=profile.id, server_id=server.server_id)


async def grantable_auth_profiles(
    requested: Iterable[str], prisma_client: "PrismaClient | None"
) -> Mapping[str, RegisteredProfile]:
    from litellm.proxy.mcp_server.mcp_server_manager import global_mcp_server_manager

    config: Final = registered_auth_profiles(global_mcp_server_manager.config_mcp_servers.values())
    wanted: Final = sorted({grant.lower() for grant in requested} - config.keys())
    if prisma_client is None or not wanted:
        return config
    from litellm.proxy.mcp_server.db import LOADABLE_MCP_SERVER_FILTER
    from litellm.repositories.table_repositories import MCPAuthProfileRepository

    rows: Final = await MCPAuthProfileRepository(prisma_client, use_writer=True).table.find_many(
        where={
            "profile_id_normalized": {"in": wanted},  # bounded-ok: the ids one grant names
            "server": {"is": LOADABLE_MCP_SERVER_FILTER},
        }
    )
    return MappingProxyType(
        {
            **config,
            **{row.profile_id_normalized: RegisteredProfile(row.profile_id, row.server_id) for row in rows},
        }
    )


def grant_error(requested: Sequence[str], profiles: Mapping[str, RegisteredProfile]) -> GrantError | None:
    unknown: Final = tuple(grant for grant in requested if grant.lower() not in profiles)
    if unknown:
        return UnknownGrant(unknown)
    granted: Final = tuple(dict.fromkeys(profiles[grant.lower()] for grant in requested))
    by_server: Final = {
        server_id: tuple(profile.profile_id for profile in granted if profile.server_id == server_id)
        for server_id in dict.fromkeys(profile.server_id for profile in granted)
    }
    ambiguous: Final = {server_id: ids for server_id, ids in by_server.items() if len(ids) > 1}
    return AmbiguousGrant(MappingProxyType(ambiguous)) if ambiguous else None


def describe_grant_error(error: GrantError) -> str:
    match error:
        case UnknownGrant(profile_ids=profile_ids):
            return f"Unknown MCP auth profile ids in mcp_auth_profiles: {', '.join(profile_ids)}"
        case AmbiguousGrant(server_profile_ids=by_server):
            named: Final = "; ".join(f"{server_id} ({', '.join(ids)})" for server_id, ids in by_server.items())
            return f"mcp_auth_profiles may name one profile per MCP server: {named}"
        case _:
            assert_never(error)


def canonical_auth_profile_grants(
    requested: Sequence[str], profiles: Mapping[str, RegisteredProfile]
) -> tuple[str, ...]:
    error: Final = grant_error(requested, profiles)
    if error is not None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail={"error": describe_grant_error(error)})
    return tuple(dict.fromkeys(profiles[grant.lower()].profile_id for grant in requested))


async def enforce_auth_profile_grant_write(
    object_permission: Mapping[str, object] | None,
    existing_object_permission_id: str | None,
    is_proxy_admin: bool,
    prisma_client: "PrismaClient | None",
) -> tuple[str, ...] | None:
    sent: Final = (object_permission or {}).get("mcp_auth_profiles")
    if sent is None:
        return None
    requested: Final = _grant_ids(sent)
    if is_proxy_admin:
        return canonical_auth_profile_grants(requested, await grantable_auth_profiles(requested, prisma_client))
    stored: Final = await _stored_auth_profile_grants(existing_object_permission_id, prisma_client)
    if _folded(requested) != _folded(stored):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": "Only a proxy admin can change mcp_auth_profiles."},
        )
    return stored


def _grant_ids(sent: object) -> tuple[str, ...]:
    refusal: Final = HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={"error": f"mcp_auth_profiles must be a list of MCP auth profile ids, at most {_MAX_GRANT_IDS}"},
    )
    try:
        requested: Final = _GRANT_IDS.validate_python(sent)
    except ValidationError as invalid:
        raise refusal from invalid
    if len(requested) > _MAX_GRANT_IDS:
        raise refusal
    return requested


def _folded(grants: Iterable[str]) -> frozenset[str]:
    return frozenset(grant.lower() for grant in grants)


async def _stored_auth_profile_grants(
    object_permission_id: str | None, prisma_client: "PrismaClient | None"
) -> tuple[str, ...]:
    if not object_permission_id or prisma_client is None:
        return ()
    from litellm.repositories.table_repositories import ObjectPermissionRepository

    stored: Final = await ObjectPermissionRepository(prisma_client, use_writer=True).table.find_unique(
        where={"object_permission_id": object_permission_id}
    )
    return () if stored is None else tuple(stored.mcp_auth_profiles)


@overload
async def with_canonical_auth_profile_grants(
    permission: Mapping[str, object],
    existing_object_permission_id: str | None,
    is_proxy_admin: bool,
    prisma_client: "PrismaClient | None",
) -> Mapping[str, object]: ...


@overload
async def with_canonical_auth_profile_grants(
    permission: object,
    existing_object_permission_id: str | None,
    is_proxy_admin: bool,
    prisma_client: "PrismaClient | None",
) -> Mapping[str, object] | None: ...


async def with_canonical_auth_profile_grants(
    permission: object,
    existing_object_permission_id: str | None,
    is_proxy_admin: bool,
    prisma_client: "PrismaClient | None",
) -> Mapping[str, object] | None:
    if permission is None:
        return None
    sent: Final = _PERMISSION.validate_python(permission)
    granted: Final = await enforce_auth_profile_grant_write(
        sent, existing_object_permission_id, is_proxy_admin, prisma_client
    )
    if granted is None:
        return sent
    if not is_proxy_admin and existing_object_permission_id is not None:  # writing a resend could restore a revoke
        return {key: value for key, value in sent.items() if key != "mcp_auth_profiles"}
    return {**sent, "mcp_auth_profiles": list(granted)}


def refuse_auth_profile_grants(
    carrier: Literal["user", "organization", "customer", "project"], object_permission: object
) -> None:
    sent: Final[object] = json.loads(object_permission) if isinstance(object_permission, str) else object_permission
    granted: Final[object] = (
        _PERMISSION.validate_python(sent).get("mcp_auth_profiles")
        if isinstance(sent, Mapping)
        else getattr(sent, "mcp_auth_profiles", None)
    )
    if granted:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": f"mcp_auth_profiles can be set on keys, teams and agents, not on a {carrier}."},
        )
