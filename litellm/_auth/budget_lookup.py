from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from pydantic import TypeAdapter

from litellm._auth.failure import native_boundary
from litellm.rust_bridge import _native


@dataclass(frozen=True, slots=True)
class OrganizationFacts:
    token_present: bool
    client_present: bool
    team_present: bool
    token_id: bool | None = None
    team_id: bool | None = None
    row_present: bool | None = None
    lookup_failed: bool = False


OrganizationAction: TypeAlias = Literal["Skip", "ReadTokenId", "ReadTeamId", "ReadOrganization", "UseOrganization"]


@dataclass(frozen=True, slots=True)
class MemberFacts:
    team_present: bool
    token_present: bool
    membership_loaded: bool
    team_id: bool | None = None
    user_id: bool | None = None
    limits_read: bool = False
    maximum_present: bool = False
    default_id_string: bool = False
    default_read: bool = False


MemberAction: TypeAlias = Literal[
    "Skip", "ReadTeamId", "ReadUserId", "ReadMembership", "ReadLimits", "ReadDefault", "UseLimits"
]
_ORGANIZATION: Final = TypeAdapter[OrganizationAction](OrganizationAction)
_MEMBER: Final = TypeAdapter[MemberAction](MemberAction)
_ORGANIZATION_FACTS: Final = TypeAdapter(OrganizationFacts)
_MEMBER_FACTS: Final = TypeAdapter(MemberFacts)
_ITEMS: Final = TypeAdapter(tuple[object, ...])
_EMAILS: Final = TypeAdapter(tuple[str, ...])
_METADATA: Final = TypeAdapter(Mapping[object, object])


def soft_email_value(raw: object) -> object:
    return _METADATA.validate_python(raw).get("soft_budget_alerting_emails") if isinstance(raw, dict) else None


@dataclass(frozen=True, slots=True)
class SoftFacts:
    limit: float | None
    spend: float | None
    token_present: bool


_SOFT: Final = TypeAdapter(SoftFacts)


@native_boundary
def read_soft_emails(facts: SoftFacts) -> bool:
    return _native.request_auth_budget_lookup_soft(_SOFT.dump_python(facts))  # pyright: ignore[reportAny]  # Pydantic serializes frozen facts


@native_boundary
def organization(facts: OrganizationFacts) -> OrganizationAction:
    return _ORGANIZATION.validate_python(
        _native.request_auth_budget_lookup_organization(_ORGANIZATION_FACTS.dump_python(facts))  # pyright: ignore[reportAny]  # Pydantic serializes frozen facts
    )


@native_boundary
def member(facts: MemberFacts) -> MemberAction:
    return _MEMBER.validate_python(_native.request_auth_budget_lookup_member(_MEMBER_FACTS.dump_python(facts)))  # pyright: ignore[reportAny]  # Pydantic serializes frozen facts


@native_boundary
def emails(raw: object) -> tuple[str, ...]:
    facts: Final = (
        {"Text": raw}
        if isinstance(raw, str)
        else {"List": tuple(value if isinstance(value, str) else None for value in _ITEMS.validate_python(raw))}
        if isinstance(raw, list)
        else "Other"
    )
    return _EMAILS.validate_python(_native.request_auth_budget_lookup_emails(facts))
