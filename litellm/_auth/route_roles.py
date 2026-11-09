from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from pydantic import TypeAdapter

from litellm._auth.failure import native_boundary
from litellm.rust_bridge import _native


@dataclass(frozen=True, slots=True)
class RoleFacts:
    user_present: bool
    role: str | None


@dataclass(frozen=True, slots=True)
class RoleDecision:
    effective: str
    user_role: str | None
    admin: bool


@dataclass(frozen=True, slots=True)
class ApiFacts:
    token_present: bool
    admin_read: bool
    user: RoleFacts


ApiDecision: TypeAlias = Literal["InvalidToken", "ReadAdmin", "NonAdmin", "Allow"]


@dataclass(frozen=True, slots=True)
class JwtFacts:
    role: str | None
    config_read: bool = False
    config_missing: bool = False


JwtDecision: TypeAlias = Literal["ReadAdmin", "ReadTeam", "DefaultTeam", "Check", "Deny"]


@dataclass(frozen=True, slots=True)
class InsideFacts:
    role: str | None
    user_id: str | None
    requested_user_id: str | None


@dataclass(frozen=True, slots=True)
class Entry:
    text: str | None
    python_type: str
    hash_failure: str | None = None
    templates: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class ListedFacts:
    route: str
    entry: Entry | None = None
    exhausted: bool = False
    template_index: int = 0
    compiled: str | None = None


@dataclass(frozen=True, slots=True)
class Failure:
    kind: Literal["TypeError", "AttributeError"]
    message: str


@dataclass(frozen=True, slots=True)
class EntryDecision:
    action: Literal["ReadEntry", "ReadTemplate", "Allow", "Deny", "Keep", "Expand", "Failure"]
    template_index: int
    failure: Failure | None


_ROLE: Final = TypeAdapter(RoleFacts)
_ROLE_RESULT: Final = TypeAdapter(RoleDecision)
_API: Final = TypeAdapter(ApiFacts)
_API_RESULT: Final = TypeAdapter[ApiDecision](ApiDecision)
_JWT: Final = TypeAdapter(JwtFacts)
_JWT_RESULT: Final = TypeAdapter[JwtDecision](JwtDecision)
_INSIDE: Final = TypeAdapter(InsideFacts)
_BOOL: Final = TypeAdapter(bool)
_ENTRY: Final = TypeAdapter(Entry)
_LISTED: Final = TypeAdapter(ListedFacts)
_ENTRY_RESULT: Final = TypeAdapter(EntryDecision)


@native_boundary
def role(facts: RoleFacts) -> RoleDecision:
    return _ROLE_RESULT.validate_python(_native.request_auth_route_role(_ROLE.dump_python(facts)))  # pyright: ignore[reportAny]  # Pydantic serializes frozen facts


@native_boundary
def api(facts: ApiFacts) -> ApiDecision:
    return _API_RESULT.validate_python(_native.request_auth_route_role_api(_API.dump_python(facts)))  # pyright: ignore[reportAny]  # Pydantic serializes frozen facts


@native_boundary
def jwt(facts: JwtFacts) -> JwtDecision:
    return _JWT_RESULT.validate_python(_native.request_auth_route_role_jwt(_JWT.dump_python(facts)))  # pyright: ignore[reportAny]  # Pydantic serializes frozen facts


@native_boundary
def inside(facts: InsideFacts) -> bool:
    return _BOOL.validate_python(_native.request_auth_route_role_inside(_INSIDE.dump_python(facts)))  # pyright: ignore[reportAny]  # Pydantic serializes frozen facts


@native_boundary
def listed(facts: ListedFacts) -> EntryDecision:
    return _ENTRY_RESULT.validate_python(_native.request_auth_route_role_listed(_LISTED.dump_python(facts)))  # pyright: ignore[reportAny]  # Pydantic serializes frozen facts


@native_boundary
def expand(facts: Entry) -> EntryDecision:
    return _ENTRY_RESULT.validate_python(_native.request_auth_route_role_expand(_ENTRY.dump_python(facts)))  # pyright: ignore[reportAny]  # Pydantic serializes frozen facts
