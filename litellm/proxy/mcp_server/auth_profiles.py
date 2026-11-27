from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

from pydantic import TypeAdapter, ValidationError
from typing_extensions import assert_never

from litellm.types.mcp_server.auth_profile import (
    ClientCredentialsAuthProfile,
    MCPAuthProfile,
    StaticAuthProfile,
    profile_id_key,
)

_MAX_PROFILES: Final = 256
_PROFILES: Final[TypeAdapter[tuple[MCPAuthProfile, ...]]] = TypeAdapter(tuple[MCPAuthProfile, ...])
_ITEMS: Final = TypeAdapter(tuple[object, ...])


@dataclass(frozen=True, slots=True)
class InvalidAuthProfiles:
    message: str


def parse_auth_profiles(server: str, raw: object) -> tuple[MCPAuthProfile, ...] | InvalidAuthProfiles:
    if not isinstance(raw, (tuple, list)):
        return InvalidAuthProfiles(f"additional_auth of MCP server '{server}' must be a list")
    items: Final = _ITEMS.validate_python(raw)
    if len(items) > _MAX_PROFILES:
        return InvalidAuthProfiles(f"additional_auth of MCP server '{server}' has more than {_MAX_PROFILES} profiles")

    try:
        profiles: Final = _PROFILES.validate_python(items)
    except ValidationError as exc:
        error: Final = exc.errors(include_input=False, include_context=False)[0]
        index, *path_parts = error["loc"]
        path: Final = "".join(f".{part}" for part in path_parts[1:])
        return InvalidAuthProfiles(
            f"additional_auth[{index}]{path} of MCP server '{server}' is invalid: {error['msg']}"
        )

    keys: Final = tuple(profile_id_key(profile.id) for profile in profiles)
    for position, key in enumerate(keys):
        first = keys.index(key)
        if first != position:
            return InvalidAuthProfiles(
                f"additional_auth[{position}] of MCP server '{server}' repeats profile id '{profiles[position].id}'"
                f" (first '{profiles[first].id}' at additional_auth[{first}])"
            )
    return profiles


def _missing_credentials(profile: MCPAuthProfile) -> tuple[str, ...]:
    match profile:
        case StaticAuthProfile():
            return () if profile.authentication_token else ("authentication_token",)
        case ClientCredentialsAuthProfile():
            fields: Final = (
                ("client_id", profile.client_id),
                ("client_secret", profile.client_secret),
                ("token_url", profile.token_url),
            )
            return tuple(name for name, value in fields if not value)
        case _:  # pragma: no cover - exhaustiveness guard, unreachable while the union holds
            assert_never(profile)  # pragma: no cover


def profile_credentials_error(server: str, profiles: Iterable[MCPAuthProfile]) -> InvalidAuthProfiles | None:
    for profile in profiles:
        missing = _missing_credentials(profile)
        if missing:
            return InvalidAuthProfiles(f"Profile '{profile.id}' of MCP server '{server}' requires {', '.join(missing)}")
    return None
