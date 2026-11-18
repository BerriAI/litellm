import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from pydantic import TypeAdapter, ValidationError
from typing_extensions import assert_never

from litellm.types.mcp import MCPAuth, MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPAuthProfile

PROFILE_ID_PATTERN: Final = re.compile(r"[A-Za-z0-9_-]{1,64}")
RESERVED_PROFILE_ID: Final = "default"
SUPPORTED_PROFILE_AUTH_TYPES: Final = frozenset(
    member.value
    for member in (
        MCPAuth.oauth2,
        MCPAuth.api_key,
        MCPAuth.bearer_token,
        MCPAuth.basic,
        MCPAuth.token,
        MCPAuth.authorization,
    )
)
_ENTRY: Final = TypeAdapter(Mapping[str, object])
_ENTRIES: Final = TypeAdapter(tuple[object, ...])
_KNOWN_AUTH_TYPES: Final = frozenset(member.value for member in MCPAuth)


@dataclass(frozen=True, slots=True)
class InvalidProfileId:
    server: str
    position: int
    value: object


@dataclass(frozen=True, slots=True)
class MalformedProfile:
    server: str
    position: int | None
    detail: str


@dataclass(frozen=True, slots=True)
class UnsupportedAuthType:
    server: str
    position: int
    profile_id: str
    auth_type: str


@dataclass(frozen=True, slots=True)
class UnsupportedOAuth2Flow:
    server: str
    position: int
    profile_id: str
    oauth2_flow: object


@dataclass(frozen=True, slots=True)
class DuplicateProfileId:
    profile_id: str
    server: str
    other_profile_id: str
    other_server: str


@dataclass(frozen=True, slots=True)
class UnsupportedProfileHost:
    server: str
    host: Literal["stdio", "BYOK", "OpenAPI"]


AuthProfileRefusal: TypeAlias = (
    InvalidProfileId
    | MalformedProfile
    | UnsupportedAuthType
    | UnsupportedOAuth2Flow
    | DuplicateProfileId
    | UnsupportedProfileHost
)


@dataclass(frozen=True, slots=True)
class StoredProfileId:
    profile_id: str
    server: str


@dataclass(frozen=True, slots=True)
class ValidatedProfiles:
    by_server: Mapping[str, tuple[MCPAuthProfile, ...]]


@dataclass(frozen=True, slots=True)
class _Claim:
    normalized: str
    profile_id: str
    server: str


def _as_entries(raw: object) -> tuple[object, ...] | None:
    if not isinstance(raw, (list, tuple)):
        return None
    return _ENTRIES.validate_python(raw)


def _validation_detail(error: ValidationError) -> str:
    return ", ".join(f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}" for item in error.errors())


def _parse_profile(server: str, position: int, raw: object) -> MCPAuthProfile | AuthProfileRefusal:
    try:
        entry: Final = _ENTRY.validate_python(raw)
    except ValidationError:
        return MalformedProfile(server, position, "a profile must be a mapping")
    profile_id: Final = entry.get("id")
    if (
        not isinstance(profile_id, str)
        or PROFILE_ID_PATTERN.fullmatch(profile_id) is None
        or profile_id.lower() == RESERVED_PROFILE_ID
    ):
        return InvalidProfileId(server, position, profile_id)
    auth_type: Final = entry.get("auth_type")
    kind: Final = auth_type.value if isinstance(auth_type, MCPAuth) else auth_type
    if isinstance(kind, str) and kind in _KNOWN_AUTH_TYPES and kind not in SUPPORTED_PROFILE_AUTH_TYPES:
        return UnsupportedAuthType(server, position, profile_id, kind)
    if kind == MCPAuth.oauth2.value and entry.get("oauth2_flow") != "client_credentials":
        return UnsupportedOAuth2Flow(server, position, profile_id, entry.get("oauth2_flow"))
    try:
        return MCPAuthProfile.model_validate(entry)
    except ValidationError as error:
        return MalformedProfile(server, position, _validation_detail(error))


def _parsed_profiles(submitted: Mapping[str, object]) -> Iterator[tuple[str, MCPAuthProfile | AuthProfileRefusal]]:
    for server, raw in submitted.items():
        if (entries := _as_entries(raw)) is None:
            yield server, MalformedProfile(server, None, "additional_auth must be a list of profiles")
            continue
        for position, entry in enumerate(entries):
            yield server, _parse_profile(server, position, entry)


def _first_refusal(results: Iterable[MCPAuthProfile | AuthProfileRefusal]) -> AuthProfileRefusal | None:
    for result in results:
        if not isinstance(result, MCPAuthProfile):
            return result
    return None


def _first_duplicate(claims: Sequence[_Claim]) -> DuplicateProfileId | None:
    first_index: Final = {claim.normalized: index for index, claim in reversed(tuple(enumerate(claims)))}
    return next(
        (
            DuplicateProfileId(
                claim.profile_id,
                claim.server,
                claims[first_index[claim.normalized]].profile_id,
                claims[first_index[claim.normalized]].server,
            )
            for index, claim in enumerate(claims)
            if first_index[claim.normalized] != index
        ),
        None,
    )


def validate_auth_profiles(
    submitted: Mapping[str, object],
    existing: Iterable[StoredProfileId],
) -> ValidatedProfiles | AuthProfileRefusal:
    parsed: Final = tuple(_parsed_profiles(submitted))
    refusal: Final = _first_refusal(result for _, result in parsed)
    if refusal is not None:
        return refusal
    by_server: Final = MappingProxyType(
        {
            server: tuple(result for owner, result in parsed if owner == server and isinstance(result, MCPAuthProfile))
            for server in submitted
        }
    )
    kept: Final = tuple(
        _Claim(stored.profile_id.lower(), stored.profile_id, stored.server)
        for stored in existing
        if stored.server not in by_server
    )
    submitted_claims: Final = tuple(
        _Claim(result.id.lower(), result.id, owner) for owner, result in parsed if isinstance(result, MCPAuthProfile)
    )
    return _first_duplicate(kept + submitted_claims) or ValidatedProfiles(by_server)


def _credential_fields(profile: MCPAuthProfile) -> Mapping[str, str | None]:
    if profile.auth_type == MCPAuth.oauth2:
        return {"client_id": profile.client_id, "client_secret": profile.client_secret, "token_url": profile.token_url}
    return {"authentication_token": profile.authentication_token}


def _missing_credential(profile: MCPAuthProfile) -> str | None:
    return next((field for field, value in _credential_fields(profile).items() if not value), None)


def profile_missing_credential(server: str, profiles: Sequence[MCPAuthProfile]) -> MalformedProfile | None:
    return next(
        (
            MalformedProfile(server, position, f"profile '{profile.id}' needs {field}")
            for position, profile in enumerate(profiles)
            if (field := _missing_credential(profile)) is not None
        ),
        None,
    )


def profile_host_refusal(
    server: str, *, transport: object, is_byok: object, spec_path: object
) -> UnsupportedProfileHost | None:
    if transport == MCPTransport.stdio:
        return UnsupportedProfileHost(server, "stdio")
    if is_byok:
        return UnsupportedProfileHost(server, "BYOK")
    if spec_path:
        return UnsupportedProfileHost(server, "OpenAPI")
    return None


def describe_refusal(refusal: AuthProfileRefusal) -> str:
    match refusal:
        case InvalidProfileId(server=server, position=position, value=value):
            return (
                f"additional_auth[{position}] of MCP server '{server}' has the invalid id {value!r}. "
                f"An id is a string of 1 to 64 characters from [A-Za-z0-9_-] and cannot be '{RESERVED_PROFILE_ID}'."
            )
        case MalformedProfile(server=server, position=position, detail=detail):
            where: Final = "additional_auth" if position is None else f"additional_auth[{position}]"
            return f"{where} of MCP server '{server}' is invalid. {detail}"
        case UnsupportedAuthType(server=server, position=position, profile_id=profile_id, auth_type=auth_type):
            return (
                f"additional_auth[{position}] '{profile_id}' of MCP server '{server}' uses auth_type '{auth_type}', "
                "which is not supported in an additional profile."
            )
        case UnsupportedOAuth2Flow(server=server, position=position, profile_id=profile_id, oauth2_flow=flow):
            return (
                f"additional_auth[{position}] '{profile_id}' of MCP server '{server}' has oauth2_flow {flow!r}. "
                "An oauth2 profile must set oauth2_flow client_credentials explicitly."
            )
        case DuplicateProfileId(
            profile_id=profile_id, server=server, other_profile_id=other_id, other_server=other_server
        ):
            return (
                f"Profile id '{profile_id}' of MCP server '{server}' collides with profile id '{other_id}' "
                f"of MCP server '{other_server}'. Profile ids are unique across all servers, ignoring case."
            )
        case UnsupportedProfileHost(server=server, host=host):
            return (
                f"MCP server '{server}' is a {host} server and cannot have additional_auth profiles. "
                "Profiles apply only to HTTP and SSE servers."
            )
        case _:
            assert_never(refusal)


def refusal_http_status(refusal: AuthProfileRefusal) -> Literal[400, 409]:
    match refusal:
        case DuplicateProfileId():
            return 409
        case (
            InvalidProfileId()
            | MalformedProfile()
            | UnsupportedAuthType()
            | UnsupportedOAuth2Flow()
            | UnsupportedProfileHost()
        ):
            return 400
        case _:
            assert_never(refusal)
