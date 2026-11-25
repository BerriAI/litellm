from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from typing import Final, TypeAlias

from pydantic import StrictStr, TypeAdapter, ValidationError

from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.types.mcp_server.auth_profile import RESERVED_PROFILE_ID, MCPAuthProfile, profile_id_key

_MAX_GRANTS: Final = 256
_GRANTS: Final[TypeAdapter[Mapping[str, str]]] = TypeAdapter(Mapping[StrictStr, StrictStr])


@dataclass(frozen=True, slots=True)
class MalformedGrants:
    pass


@dataclass(frozen=True, slots=True)
class TooManyGrants:
    count: int


@dataclass(frozen=True, slots=True)
class UnreachableServer:
    server_id: str


@dataclass(frozen=True, slots=True)
class UnknownServer:
    server_id: str


@dataclass(frozen=True, slots=True)
class UnknownProfile:
    server_id: str
    profile_id: str


@dataclass(frozen=True, slots=True)
class AmbiguousProfile:
    server_id: str
    profile_id: str


GrantError: TypeAlias = TooManyGrants | UnreachableServer | UnknownServer | UnknownProfile | AmbiguousProfile


def parse_grants(raw: object) -> Mapping[str, str] | MalformedGrants | TooManyGrants:
    entries: Final = as_str_mapping(raw)
    if entries is None:
        return MalformedGrants()
    count: Final = len(entries)
    if count > _MAX_GRANTS:
        return TooManyGrants(count)
    try:
        return _GRANTS.validate_python(entries)
    except ValidationError:
        return MalformedGrants()


def _canonical_profile(
    server_id: str, profile_id: str, profiles: Sequence[MCPAuthProfile] | None
) -> str | UnknownServer | UnknownProfile | AmbiguousProfile:
    if profiles is None:
        return UnknownServer(server_id)
    key: Final = profile_id_key(profile_id)
    if key == RESERVED_PROFILE_ID:
        return RESERVED_PROFILE_ID
    matches: Final = tuple(profile.id for profile in profiles if profile_id_key(profile.id) == key)
    match matches:
        case (canonical,):
            return canonical
        case ():
            return UnknownProfile(server_id, profile_id)
        case _:
            return AmbiguousProfile(server_id, profile_id)


def canonicalize_grants(
    requested: Mapping[str, str],
    profiles_by_server: Mapping[str, Sequence[MCPAuthProfile]],
    *,
    reachable: Set[str],
) -> Mapping[str, str] | GrantError:
    if len(requested) > _MAX_GRANTS:
        return TooManyGrants(len(requested))
    for server in requested:
        if server not in reachable:
            return UnreachableServer(server)
    resolved: Final = {
        server: _canonical_profile(server, profile_id, profiles_by_server.get(server))
        for server, profile_id in requested.items()
    }
    for result in resolved.values():
        if not isinstance(result, str):
            return result
    return {server: profile_id for server, profile_id in resolved.items() if isinstance(profile_id, str)}
