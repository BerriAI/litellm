import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, TypeAlias, overload

from fastapi import HTTPException, status
from pydantic import TypeAdapter, ValidationError
from typing_extensions import assert_never

from litellm._logging import verbose_proxy_logger
from litellm.types.mcp_server.mcp_server_manager import MCPServer

if TYPE_CHECKING:
    from prisma import models as prisma_models

    from litellm.proxy.utils import PrismaClient
    from litellm.repositories.prisma_protocols import TableActions


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


_HOLDER_INCLUDE: Final = MappingProxyType({"verification_tokens": True, "teams": True, "agents_table": True})


@dataclass(frozen=True, slots=True)
class GrantHolder:
    kind: Literal["key", "team", "agent"]
    name: str
    profile_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StillGranted:
    holders: tuple[GrantHolder, ...]


def _grant_holders(row: "prisma_models.LiteLLM_ObjectPermissionTable") -> Iterator[GrantHolder]:
    profile_ids: Final = tuple(row.mcp_auth_profiles)
    for token in row.verification_tokens or ():
        yield GrantHolder("key", token.key_alias or token.key_name or "key without alias", profile_ids)
    for team in row.teams or ():
        yield GrantHolder("team", team.team_alias or team.team_id, profile_ids)
    for agent in row.agents_table or ():
        yield GrantHolder("agent", agent.agent_name or agent.agent_id, profile_ids)


async def auth_profile_grant_holders(
    profile_ids: Sequence[str], permission_table: "TableActions[prisma_models.LiteLLM_ObjectPermissionTable]"
) -> tuple[GrantHolder, ...]:
    if not profile_ids:
        return ()
    rows: Final = await permission_table.find_many(
        where={"mcp_auth_profiles": {"has_some": list(profile_ids)}},  # bounded-ok: one server's profiles
        include=_HOLDER_INCLUDE,
    )
    return tuple(chain.from_iterable(map(_grant_holders, rows)))


def still_granted_error(refusal: StillGranted) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={
            "error": "MCP auth profiles are still granted. Remove the grants first.",
            "grants": [
                {"kind": holder.kind, "name": holder.name, "mcp_auth_profiles": list(holder.profile_ids)}
                for holder in refusal.holders
            ],
        },
    )


async def raise_if_auth_profiles_granted(profile_ids: Sequence[str], prisma_client: "PrismaClient | None") -> None:
    if prisma_client is None:
        return
    from litellm.repositories.table_repositories import ObjectPermissionRepository

    holders: Final = await auth_profile_grant_holders(
        profile_ids, ObjectPermissionRepository(prisma_client, use_writer=True).table
    )
    if holders:
        raise still_granted_error(StillGranted(holders))


async def raise_if_server_profiles_granted(server_id: str, prisma_client: "PrismaClient") -> None:
    from litellm.proxy.mcp_server.auth_profile_db import stored_profile_rows
    from litellm.proxy.mcp_server.mcp_server_manager import global_mcp_server_manager

    try:
        profile_ids: Final = tuple(await stored_profile_rows(prisma_client, server_id))
    except Exception as error:
        loaded: Final = global_mcp_server_manager.get_mcp_server_by_id(server_id)
        if loaded is not None and loaded.additional_auth:
            raise
        verbose_proxy_logger.warning(
            "could not read the MCP auth profiles of MCP server %s, removing it as it has none loaded: %s",
            server_id,
            error,
        )
        return
    await raise_if_auth_profiles_granted(profile_ids, prisma_client)


def _refused_holders(
    row: "prisma_models.LiteLLM_ObjectPermissionTable", profiles: Mapping[str, RegisteredProfile]
) -> Iterator[str]:
    error: Final = grant_error(tuple(row.mcp_auth_profiles), profiles)
    if error is None:
        return
    for holder in _grant_holders(row):
        yield f"{holder.kind} {holder.name}: {describe_grant_error(error)}"


async def _startup_grant_refusal(prisma_client: "PrismaClient") -> str | None:
    from litellm.repositories.table_repositories import ObjectPermissionRepository

    rows: Final = await ObjectPermissionRepository(prisma_client, use_writer=True).table.find_many(
        where={"NOT": [{"mcp_auth_profiles": {"is_empty": True}}]}, include=_HOLDER_INCLUDE
    )
    profiles: Final = await grantable_auth_profiles(
        chain.from_iterable(row.mcp_auth_profiles for row in rows), prisma_client
    )
    refused: Final = tuple(chain.from_iterable(_refused_holders(row, profiles) for row in rows))
    return None if not refused else f"MCP auth profile grants no longer match the MCP servers: {'; '.join(refused)}"


async def mcp_auth_profile_startup_refusal(
    config_servers: Iterable[MCPServer], prisma_client: "PrismaClient | None"
) -> str | None:
    if prisma_client is None:
        return None
    from litellm.proxy.mcp_server.auth_profile_db import startup_profile_refusal

    profile_refusal: Final = await startup_profile_refusal(config_servers, prisma_client)
    if profile_refusal is not None:
        return profile_refusal
    from prisma.errors import PrismaError

    from litellm.proxy._types import DB_CONNECTION_ERROR_TYPES

    try:
        return await _startup_grant_refusal(prisma_client)
    except (PrismaError, *DB_CONNECTION_ERROR_TYPES) as error:
        verbose_proxy_logger.warning(
            "Skipped the MCP auth profile grant check at startup, the database could not be read: %s", error
        )
        return None
