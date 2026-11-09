from dataclasses import asdict, dataclass
from typing import Final, Literal, TypeAlias

from pydantic import TypeAdapter

from litellm._auth.failure import native_boundary
from litellm.rust_bridge import _native


@dataclass(frozen=True, slots=True)
class OrganizationMembership:
    organization_id: str | None
    role: str | None


@dataclass(frozen=True, slots=True)
class OrganizationRouteFacts:
    route: str
    user_present: bool
    user_role: str | None
    memberships: tuple[OrganizationMembership, ...] | None
    organization_id: str | None
    org_admin_routes: tuple[str, ...]
    organization_id_present: bool = False
    identifier_pending: bool = False
    hash_failure: str | None = None
    request_pending: bool = False
    memberships_pending: bool = False


OrgRouteDenial: TypeAlias = Literal[
    "CreateOrganization",
    "NoMemberships",
    "MissingOrganization",
    "MissingRole",
    "WrongRole",
    "TeamMissingOrganization",
    "TeamWrongRole",
]


@dataclass(frozen=True, slots=True)
class OrganizationRouteDecision:
    denial: OrgRouteDenial | None
    organization_role: str | None
    organization_ids: tuple[str, ...]
    read_identifier: bool = False
    hash_failure: str | None = None
    read_request: bool = False
    read_memberships: bool = False


_DECISION: Final = TypeAdapter(OrganizationRouteDecision)


@native_boundary
def organization_route(facts: OrganizationRouteFacts) -> OrganizationRouteDecision:
    return _DECISION.validate_python(_native.request_auth_org_route(asdict(facts)))


@native_boundary
def user_is_org_admin(memberships: tuple[OrganizationMembership, ...], organizations: tuple[str, ...]) -> bool:
    return _native.request_auth_org_admin(tuple(asdict(item) for item in memberships), organizations)


@dataclass(frozen=True, slots=True)
class Identifier:
    text: str | None
    hash_failure: str | None = None


@dataclass(frozen=True, slots=True)
class AdminFacts:
    memberships: tuple[OrganizationMembership, ...] | None
    request_read: bool = False
    singular_present: bool = False
    plural_count: int = 0
    candidate_index: int = 0
    candidate: Identifier | None = None


@dataclass(frozen=True, slots=True)
class AdminDecision:
    action: Literal["ReadRequest", "ReadSingular", "ReadPlural", "Allow", "Deny", "HashFailure"]
    candidate_index: int
    plural_index: int
    hash_failure: str | None


_ADMIN: Final = TypeAdapter(AdminDecision)


@native_boundary
def admin(facts: AdminFacts) -> AdminDecision:
    return _ADMIN.validate_python(_native.request_auth_org_admin_raw(asdict(facts)))
