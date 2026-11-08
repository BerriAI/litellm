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


_DECISION: Final = TypeAdapter(OrganizationRouteDecision)


@native_boundary
def organization_route(facts: OrganizationRouteFacts) -> OrganizationRouteDecision:
    return _DECISION.validate_python(_native.request_auth_org_route(asdict(facts)))


@native_boundary
def user_is_org_admin(memberships: tuple[OrganizationMembership, ...], organizations: tuple[str, ...]) -> bool:
    return _native.request_auth_org_admin(tuple(asdict(item) for item in memberships), organizations)
